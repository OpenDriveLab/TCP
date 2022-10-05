from typing import Optional, Tuple
import torch as th
import torch.nn as nn
from torch.distributions import Beta, Normal
from torch.nn import functional as F
import numpy as np


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class DiagGaussianDistribution():
    def __init__(self, action_dim: int, dist_init=None, action_dependent_std=False):
        self.distribution = None
        self.action_dim = action_dim
        self.dist_init = dist_init
        self.action_dependent_std = action_dependent_std

        self.low = None
        self.high = None
        self.log_std_max = 2
        self.log_std_min = -20

        # [mu, log_std], [0, 1]
        self.acc_exploration_dist = {
            'go': th.FloatTensor([0.66, -3]),
            'stop': th.FloatTensor([-0.66, -3])
        }
        self.steer_exploration_dist = {
            'turn': th.FloatTensor([0.0, -1]),
            'straight': th.FloatTensor([3.0, 3.0])
        }

        if th.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def proba_distribution_net(self, latent_dim: int) -> Tuple[nn.Module, nn.Parameter]:
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        if self.action_dependent_std:
            log_std = nn.Linear(latent_dim, self.action_dim)
        else:
            log_std = nn.Parameter(-2.0*th.ones(self.action_dim), requires_grad=True)

        if self.dist_init is not None:
            # log_std.weight.data.fill_(0.01)
            # mean_actions.weight.data.fill_(0.01)
            # acc/steer
            mean_actions.bias.data[0] = self.dist_init[0][0]
            mean_actions.bias.data[1] = self.dist_init[1][0]
            if self.action_dependent_std:
                log_std.bias.data[0] = self.dist_init[0][1]
                log_std.bias.data[1] = self.dist_init[1][1]
            else:
                init_tensor = th.FloatTensor([self.dist_init[0][1], self.dist_init[1][1]])
                log_std = nn.Parameter(init_tensor, requires_grad=True)

        return mean_actions, log_std

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor) -> "DiagGaussianDistribution":
        if self.action_dependent_std:
            log_std = th.clamp(log_std, self.log_std_min, self.log_std_max)
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy_loss(self) -> th.Tensor:
        entropy_loss = -1.0 * self.distribution.entropy()
        return th.mean(entropy_loss)

    def exploration_loss(self, exploration_suggests) -> th.Tensor:
        # [('stop'/'go'/None, 'turn'/'straight'/None)]
        # (batch_size, action_dim)
        mu = self.distribution.loc.detach().clone()
        sigma = self.distribution.scale.detach().clone()

        for i, (acc_suggest, steer_suggest) in enumerate(exploration_suggests):
            if acc_suggest != '':
                mu[i, 0] = self.acc_exploration_dist[acc_suggest][0]
                sigma[i, 0] = self.acc_exploration_dist[acc_suggest][1]
            if steer_suggest != '':
                mu[i, 1] = self.steer_exploration_dist[steer_suggest][0]
                sigma[i, 1] = self.steer_exploration_dist[steer_suggest][1]

        dist_ent = Normal(mu, sigma)

        exploration_loss = th.distributions.kl_divergence(dist_ent, self.distribution)
        return th.mean(exploration_loss)

    def sample(self) -> th.Tensor:
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()


class SquashedDiagGaussianDistribution():
    def __init__(self, action_dim: int, log_std_init: float = 0.0, action_dependent_std=False):
        self.distribution = None

        self.action_dim = action_dim
        self.log_std_init = log_std_init
        self.eps = 1e-7
        self.action_dependent_std = action_dependent_std

        self.low = -1.0
        self.high = 1.0
        self.log_std_max = 2
        self.log_std_min = -20

        self.gaussian_actions = None

    def proba_distribution_net(self, latent_dim: int):
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        if self.action_dependent_std:
            log_std = nn.Linear(latent_dim, self.action_dim)
        else:
            log_std = nn.Parameter(th.ones(self.action_dim) * self.log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor):
        if self.action_dependent_std:
            log_std = th.clamp(log_std, self.log_std_min, self.log_std_max)
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: th.Tensor, gaussian_actions: Optional[th.Tensor] = None) -> th.Tensor:
        # Inverse tanh
        if gaussian_actions is None:
            gaussian_actions = th.clamp(actions, min=-1.0 + self.eps, max=1.0 - self.eps)
            gaussian_actions = 0.5 * (gaussian_actions.log1p() - (-gaussian_actions).log1p())

        # Log likelihood for a Gaussian distribution
        log_prob = self.distribution.log_prob(gaussian_actions)
        log_prob = sum_independent_dims(log_prob)

        # sb3 correction
        # log_prob -= th.sum(th.log(1 - actions ** 2 + self.eps), dim=1)
        # spinning-up correction
        log_prob -= (2*(np.log(2) - gaussian_actions - F.softplus(-2*gaussian_actions))).sum(axis=1)
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        return None

    def sample(self) -> th.Tensor:
        return th.tanh(self.distribution.rsample())

    def mode(self) -> th.Tensor:
        return th.tanh(self.distribution.mean)

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()


class BetaDistribution():
    def __init__(self, action_dim=2, dist_init=None):
        assert action_dim == 2

        self.distribution = None
        self.action_dim = action_dim
        self.dist_init = dist_init
        self.low = 0.0
        self.high = 1.0

        # [beta, alpha], [0, 1]
        self.acc_exploration_dist = {
            # [1, 2.5]
            # [1.5, 1.0]
            'go': th.FloatTensor([1.0, 2.5]),
            'stop': th.FloatTensor([1.5, 1.0])
        }
        self.steer_exploration_dist = {
            'turn': th.FloatTensor([1.0, 1.0]),
            'straight': th.FloatTensor([3.0, 3.0])
        }

        if th.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def proba_distribution_net(self, latent_dim: int) -> Tuple[nn.Module, nn.Module]:

        linear_alpha = nn.Linear(latent_dim, self.action_dim)
        linear_beta = nn.Linear(latent_dim, self.action_dim)

        if self.dist_init is not None:
            # linear_alpha.weight.data.fill_(0.01)
            # linear_beta.weight.data.fill_(0.01)
            # acc
            linear_alpha.bias.data[0] = self.dist_init[0][1]
            linear_beta.bias.data[0] = self.dist_init[0][0]
            # steer
            linear_alpha.bias.data[1] = self.dist_init[1][1]
            linear_beta.bias.data[1] = self.dist_init[1][0]

        alpha = nn.Sequential(linear_alpha, nn.Softplus())
        beta = nn.Sequential(linear_beta, nn.Softplus())
        return alpha, beta

    def proba_distribution(self, alpha, beta):
        self.distribution = Beta(alpha, beta)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy_loss(self) -> th.Tensor:
        entropy_loss = -1.0 * self.distribution.entropy()
        return th.mean(entropy_loss)

    def exploration_loss(self, exploration_suggests) -> th.Tensor:
        # [('stop'/'go'/None, 'turn'/'straight'/None)]
        # (batch_size, action_dim)
        alpha = self.distribution.concentration1.detach().clone()
        beta = self.distribution.concentration0.detach().clone()

        for i, (acc_suggest, steer_suggest) in enumerate(exploration_suggests):
            if acc_suggest != '':
                beta[i, 0] = self.acc_exploration_dist[acc_suggest][0]
                alpha[i, 0] = self.acc_exploration_dist[acc_suggest][1]
            if steer_suggest != '':
                beta[i, 1] = self.steer_exploration_dist[steer_suggest][0]
                alpha[i, 1] = self.steer_exploration_dist[steer_suggest][1]

        dist_ent = Beta(alpha, beta)

        exploration_loss = th.distributions.kl_divergence(self.distribution, dist_ent)
        return th.mean(exploration_loss)

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        alpha = self.distribution.concentration1
        beta = self.distribution.concentration0
        x = th.zeros_like(alpha)
        x[:, 1] += 0.5
        mask1 = (alpha > 1) & (beta > 1)
        x[mask1] = (alpha[mask1]-1)/(alpha[mask1]+beta[mask1]-2)

        mask2 = (alpha <= 1) & (beta > 1)
        x[mask2] = 0.0

        mask3 = (alpha > 1) & (beta <= 1)
        x[mask3] = 1.0

        # mean
        mask4 = (alpha <= 1) & (beta <= 1)
        x[mask4] = self.distribution.mean[mask4]

        return x

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()
