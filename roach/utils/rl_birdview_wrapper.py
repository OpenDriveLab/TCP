import gym
import numpy as np
import cv2
import carla

eval_num_zombie_vehicles = {
    'Town01': 120,
    'Town02': 70,
    'Town03': 70,
    'Town04': 150,
    'Town05': 120,
    'Town06': 120
}
eval_num_zombie_walkers = {
    'Town01': 120,
    'Town02': 70,
    'Town03': 70,
    'Town04': 80,
    'Town05': 120,
    'Town06': 80
}

class RlBirdviewWrapper(gym.Wrapper):
    def __init__(self, env, input_states=[], acc_as_action=False):
        assert len(env._obs_configs) == 1
        self._ev_id = list(env._obs_configs.keys())[0]
        self._input_states = input_states
        self._acc_as_action = acc_as_action
        self._render_dict = {}

        state_spaces = []
        if 'speed' in self._input_states:
            state_spaces.append(env.observation_space[self._ev_id]['speed']['speed_xy'])
        if 'speed_limit' in self._input_states:
            state_spaces.append(env.observation_space[self._ev_id]['control']['speed_limit'])
        if 'control' in self._input_states:
            state_spaces.append(env.observation_space[self._ev_id]['control']['throttle'])
            state_spaces.append(env.observation_space[self._ev_id]['control']['steer'])
            state_spaces.append(env.observation_space[self._ev_id]['control']['brake'])
            state_spaces.append(env.observation_space[self._ev_id]['control']['gear'])
        if 'acc_xy' in self._input_states:
            state_spaces.append(env.observation_space[self._ev_id]['velocity']['acc_xy'])
        if 'vel_xy' in self._input_states:
            state_spaces.append(env.observation_space[self._ev_id]['velocity']['vel_xy'])
        if 'vel_ang_z' in self._input_states:
            state_spaces.append(env.observation_space[self._ev_id]['velocity']['vel_ang_z'])

        state_low = np.concatenate([s.low for s in state_spaces])
        state_high = np.concatenate([s.high for s in state_spaces])

        env.observation_space = gym.spaces.Dict(
            {'state': gym.spaces.Box(low=state_low, high=state_high, dtype=np.float32),
             'birdview': env.observation_space[self._ev_id]['birdview']['masks']})

        if self._acc_as_action:
            # act: acc(throttle/brake), steer
            env.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        else:
            # act: throttle, steer, brake
            env.action_space = gym.spaces.Box(low=np.array([0, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

        super(RlBirdviewWrapper, self).__init__(env)

        self.eval_mode = False

    def step(self, action):
        action_ma = {self._ev_id: self.process_act(action, self._acc_as_action)}

        obs_ma, reward_ma, done_ma, info_ma = self.env.step(action_ma)

        obs = self.process_obs(obs_ma[self._ev_id], self._input_states)
        reward = reward_ma[self._ev_id]
        done = done_ma[self._ev_id]
        info = info_ma[self._ev_id]

        self._render_dict = {
            'timestamp': self.env.timestamp,
            'obs': self._render_dict['prev_obs'],
            'prev_obs': obs,
            'im_render': self._render_dict['prev_im_render'],
            'prev_im_render': obs_ma[self._ev_id]['birdview']['rendered'],
            'action': action,
            'reward_debug': info['reward_debug'],
            'terminal_debug': info['terminal_debug']
        }
        return obs, reward, done, info


    @staticmethod
    def process_obs(obs, input_states, train=True):

        state_list = []
        if 'speed' in input_states:
            state_list.append(obs['speed']['speed_xy'])
        if 'speed_limit' in input_states:
            state_list.append(obs['control']['speed_limit'])
        if 'control' in input_states:
            state_list.append(obs['control']['throttle'])
            state_list.append(obs['control']['steer'])
            state_list.append(obs['control']['brake'])
            state_list.append(obs['control']['gear']/5.0)
        if 'acc_xy' in input_states:
            state_list.append(obs['velocity']['acc_xy'])
        if 'vel_xy' in input_states:
            state_list.append(obs['velocity']['vel_xy'])
        if 'vel_ang_z' in input_states:
            state_list.append(obs['velocity']['vel_ang_z'])

        state = np.concatenate(state_list)

        birdview = obs['birdview']['masks']

        if not train:
            birdview = np.expand_dims(birdview, 0)
            state = np.expand_dims(state, 0)

        obs_dict = {
            'state': state.astype(np.float32),
            'birdview': birdview
        }
        return obs_dict

    @staticmethod
    def process_act(action, acc_as_action, train=True):
        if not train:
            action = action[0]
        if acc_as_action:
            acc, steer = action.astype(np.float64)
            if acc >= 0.0:
                throttle = acc
                brake = 0.0
            else:
                throttle = 0.0
                brake = np.abs(acc)
        else:
            throttle, steer, brake = action.astype(np.float64)

        throttle = np.clip(throttle, 0, 1)
        steer = np.clip(steer, -1, 1)
        brake = np.clip(brake, 0, 1)
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        return control
