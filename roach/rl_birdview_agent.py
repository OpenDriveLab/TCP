import logging
import numpy as np
from omegaconf import OmegaConf
import copy

from roach.utils.config_utils import load_entry_point


class RlBirdviewAgent():
    def __init__(self, path_to_conf_file='config_agent.yaml', ckpt = None):

        self._render_dict = None
        self.supervision_dict = None
        self._ckpt = ckpt
        self.setup(path_to_conf_file)

    def setup(self, path_to_conf_file):
        cfg = OmegaConf.load(path_to_conf_file)
        cfg = OmegaConf.to_container(cfg)

        self._obs_configs = cfg['obs_configs']
        self._train_cfg = cfg['training']


        self._policy_class = load_entry_point(cfg['policy']['entry_point'])
        self._policy_kwargs = cfg['policy']['kwargs']
        if self._ckpt is None:
            self._policy = None
        else:
            self._policy, self._train_cfg['kwargs'] = self._policy_class.load(self._ckpt)
            self._policy = self._policy.eval()

        self._wrapper_class = load_entry_point(cfg['env_wrapper']['entry_point'])
        self._wrapper_kwargs = cfg['env_wrapper']['kwargs']

    def run_step(self, input_data, timestamp):
        input_data = copy.deepcopy(input_data)

        policy_input = self._wrapper_class.process_obs(input_data, self._wrapper_kwargs['input_states'], train=False)

        actions, values, log_probs, mu, sigma, features = self._policy.forward(
            policy_input, deterministic=True, clip_action=True)
        control = self._wrapper_class.process_act(actions, self._wrapper_kwargs['acc_as_action'], train=False)
        self.supervision_dict = {
            'action': np.array([control.throttle, control.steer, control.brake], dtype=np.float32),
            'value': values[0],
            'action_mu': mu[0],
            'action_sigma': sigma[0],
            'features': features[0],
            'speed': input_data['speed']['forward_speed']
        }
        self.supervision_dict = copy.deepcopy(self.supervision_dict)

        return control

    @property
    def obs_configs(self):
        return self._obs_configs
