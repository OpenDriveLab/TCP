import time
import torch as th
import numpy as np
from collections import deque
from torch.nn import functional as F

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance

from .ppo_buffer import PpoBuffer


class PPO():
    def __init__(self, policy, env,
                 learning_rate: float = 1e-5,
                 n_steps_total: int = 8192,
                 batch_size: int = 256,
                 n_epochs: int = 20,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.9,
                 clip_range: float = 0.2,
                 clip_range_vf: float = None,
                 ent_coef: float = 0.05,
                 explore_coef: float = 0.05,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 target_kl: float = 0.01,
                 update_adv=False,
                 lr_schedule_step=None,
                 start_num_timesteps: int = 0):

        self.policy = policy
        self.env = env
        self.learning_rate = learning_rate
        self.n_steps_total = n_steps_total
        self.n_steps = n_steps_total//env.num_envs
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.explore_coef = explore_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.update_adv = update_adv
        self.lr_schedule_step = lr_schedule_step
        self.start_num_timesteps = start_num_timesteps
        self.num_timesteps = start_num_timesteps

        self._last_obs = None
        self._last_dones = None
        self.ep_stat_buffer = None

        self.buffer = PpoBuffer(self.n_steps, self.env.observation_space, self.env.action_space,
                                gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=self.env.num_envs)
        self.policy = self.policy.to(self.policy.device)

        model_parameters = filter(lambda p: p.requires_grad, self.policy.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'trainable parameters: {total_params/1000000:.2f}M')

    def collect_rollouts(self, env: VecEnv, callback: BaseCallback,
                         rollout_buffer: PpoBuffer, n_rollout_steps: int) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()

        self.action_statistics = []
        self.mu_statistics = []
        self.sigma_statistics = []

        while n_steps < n_rollout_steps:
            actions, values, log_probs, mu, sigma, _ = self.policy.forward(self._last_obs)
            self.action_statistics.append(actions)
            self.mu_statistics.append(mu)
            self.sigma_statistics.append(sigma)

            new_obs, rewards, dones, infos = env.step(actions)

            if callback.on_step() is False:
                return False

            # update_info_buffer
            for i in np.where(dones)[0]:
                self.ep_stat_buffer.append(infos[i]['episode_stat'])

            n_steps += 1
            self.num_timesteps += env.num_envs

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs, mu, sigma, infos)
            self._last_obs = new_obs
            self._last_dones = dones

        last_values = self.policy.forward_value(self._last_obs)
        rollout_buffer.compute_returns_and_advantage(last_values, dones=self._last_dones)

        return True

    def train(self):
        for param_group in self.policy.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

        entropy_losses, exploration_losses, pg_losses, value_losses, losses = [], [], [], [], []
        clip_fractions = []
        approx_kl_divs = []

        # train for gradient_steps epochs
        epoch = 0
        data_len = int(self.buffer.buffer_size * self.buffer.n_envs / self.batch_size)
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            self.buffer.start_caching(self.batch_size)
            # while self.buffer.sample_queue.qsize() < 3:
                # time.sleep(0.01)
            for i in range(data_len):

                if self.buffer.sample_queue.empty():
                    while self.buffer.sample_queue.empty():
                        # print(f'buffer_empty: {self.buffer.sample_queue.qsize()}')
                        time.sleep(0.01)
                rollout_data = self.buffer.sample_queue.get()

                values, log_prob, entropy_loss, exploration_loss, distribution = self.policy.evaluate_actions(
                    rollout_data.observations, rollout_data.actions, rollout_data.exploration_suggests,
                    detach_values=False)
                # Normalize advantage
                advantages = rollout_data.advantages
                # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                clip_fraction = th.mean((th.abs(ratio - 1) > self.clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(values - rollout_data.old_values,
                                                                     -self.clip_range_vf, self.clip_range_vf)
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                loss = policy_loss + self.vf_coef * value_loss \
                    + self.ent_coef * entropy_loss + self.explore_coef * exploration_loss

                losses.append(loss.item())
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                exploration_losses.append(exploration_loss.item())

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                with th.no_grad():
                    old_distribution = self.policy.action_dist.proba_distribution(
                        rollout_data.old_mu, rollout_data.old_sigma)
                    kl_div = th.distributions.kl_divergence(old_distribution.distribution, distribution)

                approx_kl_divs.append(kl_div.mean().item())

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                if self.lr_schedule_step is not None:
                    self.kl_early_stop += 1
                    if self.kl_early_stop >= self.lr_schedule_step:
                        self.learning_rate *= 0.5
                        self.kl_early_stop = 0
                break

            # update advantages
            if self.update_adv:
                self.buffer.update_values(self.policy)
                last_values = self.policy.forward_value(self._last_obs)
                self.buffer.compute_returns_and_advantage(last_values, dones=self._last_dones)

        explained_var = explained_variance(self.buffer.returns.flatten(), self.buffer.values.flatten())

        # Logs
        self.train_debug = {
            "train/entropy_loss": np.mean(entropy_losses),
            "train/exploration_loss": np.mean(exploration_losses),
            "train/policy_gradient_loss": np.mean(pg_losses),
            "train/value_loss": np.mean(value_losses),
            "train/last_epoch_kl": np.mean(approx_kl_divs),
            "train/clip_fraction": np.mean(clip_fractions),
            "train/loss": np.mean(losses),
            "train/explained_variance": explained_var,
            "train/clip_range": self.clip_range,
            "train/train_epoch": epoch,
            "train/learning_rate": self.learning_rate
        }

    def learn(self, total_timesteps, callback=None, seed=2021):
        # reset env seed
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        self.env.seed(seed)

        self.start_time = time.time()

        self.kl_early_stop = 0
        self.t_train_values = 0.0

        self.ep_stat_buffer = deque(maxlen=100)
        self._last_obs = self.env.reset()
        self._last_dones = np.zeros((self.env.num_envs,), dtype=np.bool)

        callback.init_callback(self)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            callback.on_rollout_start()
            t0 = time.time()
            self.policy = self.policy.train()
            continue_training = self.collect_rollouts(self.env, callback, self.buffer, self.n_steps)
            self.t_rollout = time.time() - t0
            callback.on_rollout_end()

            if continue_training is False:
                break

            t0 = time.time()
            self.train()
            self.t_train = time.time() - t0
            callback.on_training_end()

        return self

    def _get_init_kwargs(self):
        init_kwargs = dict(
            learning_rate=self.learning_rate,
            n_steps_total=self.n_steps_total,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            clip_range_vf=self.clip_range_vf,
            ent_coef=self.ent_coef,
            explore_coef=self.explore_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            target_kl=self.target_kl,
            update_adv=self.update_adv,
            lr_schedule_step=self.lr_schedule_step,
            start_num_timesteps=self.num_timesteps,
        )
        return init_kwargs

    def save(self, path: str) -> None:
        th.save({'policy_state_dict': self.policy.state_dict(),
                 'policy_init_kwargs': self.policy.get_init_kwargs(),
                 'train_init_kwargs': self._get_init_kwargs()},
                path)

    def get_env(self):
        return self.env
