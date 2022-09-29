from typing import Union, Dict, Tuple, Any
from functools import partial
import gym
import torch as th
import torch.nn as nn
import numpy as np

from roach.utils.config_utils import load_entry_point


class PpoPolicy(nn.Module):

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 policy_head_arch=[256, 256],
                 value_head_arch=[256, 256],
                 features_extractor_entry_point=None,
                 features_extractor_kwargs={},
                 distribution_entry_point=None,
                 distribution_kwargs={}):

        super(PpoPolicy, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor_entry_point = features_extractor_entry_point
        self.features_extractor_kwargs = features_extractor_kwargs
        self.distribution_entry_point = distribution_entry_point
        self.distribution_kwargs = distribution_kwargs

        if th.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.optimizer_class = th.optim.Adam
        self.optimizer_kwargs = {'eps': 1e-5}

        features_extractor_entry_point = features_extractor_entry_point.replace("agents.rl_birdview","roach")
        features_extractor_class = load_entry_point(features_extractor_entry_point)
        self.features_extractor = features_extractor_class(observation_space, **features_extractor_kwargs)

        distribution_entry_point = distribution_entry_point.replace("agents.rl_birdview","roach")
        distribution_class = load_entry_point(distribution_entry_point)
        self.action_dist = distribution_class(int(np.prod(action_space.shape)), **distribution_kwargs)

        if 'StateDependentNoiseDistribution' in distribution_entry_point:
            self.use_sde = True
            self.sde_sample_freq = 4
        else:
            self.use_sde = False
            self.sde_sample_freq = None

        # best_so_far
        # self.net_arch = [dict(pi=[256, 128, 64], vf=[128, 64])]
        self.policy_head_arch = list(policy_head_arch)
        self.value_head_arch = list(value_head_arch)
        self.activation_fn = nn.ReLU
        self.ortho_init = False
        self._build()

    def reset_noise(self, n_envs: int = 1) -> None:
        assert self.use_sde, 'reset_noise() is only available when using gSDE'
        self.action_dist.sample_weights(self.dist_sigma, batch_size=n_envs)

    def _build(self) -> None:
        last_layer_dim_pi = self.features_extractor.features_dim
        policy_net = []
        for layer_size in self.policy_head_arch:
            policy_net.append(nn.Linear(last_layer_dim_pi, layer_size))
            policy_net.append(self.activation_fn())
            last_layer_dim_pi = layer_size
        self.policy_head = nn.Sequential(*policy_net).to(self.device)
        # mu->alpha/mean, sigma->beta/log_std (nn.Module, nn.Parameter)
        self.dist_mu, self.dist_sigma = self.action_dist.proba_distribution_net(last_layer_dim_pi)
        last_layer_dim_vf = self.features_extractor.features_dim
        value_net = []
        for layer_size in self.value_head_arch:
            value_net.append(nn.Linear(last_layer_dim_vf, layer_size))
            value_net.append(self.activation_fn())
            last_layer_dim_vf = layer_size
        value_net.append(nn.Linear(last_layer_dim_vf, 1))
        self.value_head = nn.Sequential(*value_net).to(self.device)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # feature_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                # self.features_extractor: np.sqrt(2),
                self.policy_head: np.sqrt(2),
                self.value_head: np.sqrt(2)
                # self.action_net: 0.01,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)

    def _get_features(self, birdview: th.Tensor, state: th.Tensor) -> th.Tensor:
        """
        :param birdview: th.Tensor (num_envs, frame_stack*channel, height, width)
        :param state: th.Tensor (num_envs, state_dim)
        """
        birdview = birdview.float() / 255.0
        features = self.features_extractor(birdview, state)
        return features

    def _get_action_dist_from_features(self, features: th.Tensor):
        latent_pi = self.policy_head(features)
        mu = self.dist_mu(latent_pi)
        if isinstance(self.dist_sigma, nn.Parameter):
            sigma = self.dist_sigma
        else:
            sigma = self.dist_sigma(latent_pi)
        return self.action_dist.proba_distribution(mu, sigma), mu.detach().cpu().numpy(), sigma.detach().cpu().numpy()

    def evaluate_actions(self, obs_dict: Dict[str, th.Tensor], actions: th.Tensor, exploration_suggests,
                         detach_values=False):
        features = self._get_features(**obs_dict)

        if detach_values:
            detached_features = features.detach()
            values = self.value_head(detached_features)
        else:
            values = self.value_head(features)

        distribution, mu, sigma = self._get_action_dist_from_features(features)
        actions = self.scale_action(actions)
        log_prob = distribution.log_prob(actions)
        return values.flatten(), log_prob, distribution.entropy_loss(), \
            distribution.exploration_loss(exploration_suggests), distribution.distribution

    def evaluate_values(self, obs_dict: Dict[str, th.Tensor]):
        features = self._get_features(**obs_dict)
        values = self.value_head(features)
        distribution, mu, sigma = self._get_action_dist_from_features(features)
        return values.flatten(), distribution.distribution

    def forward(self, obs_dict: Dict[str, np.ndarray], deterministic: bool = False, clip_action: bool = False, only_feature: bool = False, feature_input: np.ndarray = None):
        '''
        used in collect_rollouts(), do not clamp actions
        '''
        with th.no_grad():
            if feature_input is None:
                obs_tensor_dict = dict([(k, th.as_tensor(v).to(self.device).unsqueeze(0)) for k, v in obs_dict.items()])
                features = self._get_features(**obs_tensor_dict)
            else:
                features = th.tensor(feature_input).to(self.device)
            if only_feature:
                return features.cpu().numpy()
            values = self.value_head(features)
            distribution, mu, sigma = self._get_action_dist_from_features(features)
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
        actions = actions.cpu().numpy()
        actions = self.unscale_action(actions)
        if clip_action:
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        values = values.cpu().numpy().flatten()
        log_prob = log_prob.cpu().numpy()
        features = features.cpu().numpy()
        return actions, values, log_prob, mu, sigma, features

    def forward_value(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        with th.no_grad():
            obs_tensor_dict = dict([(k, th.as_tensor(v).to(self.device)) for k, v in obs_dict.items()])
            features = self._get_features(**obs_tensor_dict)
            values = self.value_head(features)
        values = values.cpu().numpy().flatten()
        return values

    def forward_policy(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        with th.no_grad():
            obs_tensor_dict = dict([(k, th.as_tensor(v).to(self.device)) for k, v in obs_dict.items()])
            features = self._get_features(**obs_tensor_dict)
            distribution, mu, sigma = self._get_action_dist_from_features(features)
        return mu, sigma

    def scale_action(self, action: th.Tensor, eps=1e-7) -> th.Tensor:
        # input action \in [a_low, a_high]
        # output action \in [d_low+eps, d_high-eps]
        d_low, d_high = self.action_dist.low, self.action_dist.high  # scalar

        if d_low is not None and d_high is not None:
            a_low = th.as_tensor(self.action_space.low.astype(np.float32)).to(action.device)
            a_high = th.as_tensor(self.action_space.high.astype(np.float32)).to(action.device)
            action = (action-a_low)/(a_high-a_low) * (d_high-d_low) + d_low
            action = th.clamp(action, d_low+eps, d_high-eps)
        return action

    def unscale_action(self, action: np.ndarray, eps=0.0) -> np.ndarray:
        # input action \in [d_low, d_high]
        # output action \in [a_low+eps, a_high-eps]
        d_low, d_high = self.action_dist.low, self.action_dist.high  # scalar

        if d_low is not None and d_high is not None:
            # batch_size = action.shape[0]
            a_low, a_high = self.action_space.low, self.action_space.high
            # same shape as action [batch_size, action_dim]
            # a_high = np.tile(self.action_space.high, [batch_size, 1])
            action = (action-d_low)/(d_high-d_low) * (a_high-a_low) + a_low
            # action = np.clip(action, a_low+eps, a_high-eps)
        return action

    def get_init_kwargs(self) -> Dict[str, Any]:
        init_kwargs = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            policy_head_arch=self.policy_head_arch,
            value_head_arch=self.value_head_arch,
            features_extractor_entry_point=self.features_extractor_entry_point,
            features_extractor_kwargs=self.features_extractor_kwargs,
            distribution_entry_point=self.distribution_entry_point,
            distribution_kwargs=self.distribution_kwargs,
        )
        return init_kwargs

    @classmethod
    def load(cls, path):
        if th.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        saved_variables = th.load(path, map_location=device)
        # Create policy object
        model = cls(**saved_variables['policy_init_kwargs'])
        
        # Load weights
        model.load_state_dict(saved_variables['policy_state_dict'])
        model.to(device)
        return model, saved_variables['train_init_kwargs']

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
