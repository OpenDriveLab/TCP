import numpy as np
from gym import spaces

from roach.obs_manager.obs_manager import ObsManagerBase


class ObsManager():

    def __init__(self, obs_configs):
        self._parent_actor = None
    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor

    def get_observation(self):
        control = self._parent_actor.vehicle.get_control()
        speed_limit = self._parent_actor.vehicle.get_speed_limit() / 3.6 * 0.8
        obs = {
            'throttle': np.array([control.throttle], dtype=np.float32),
            'steer': np.array([control.steer], dtype=np.float32),
            'brake': np.array([control.brake], dtype=np.float32),
            'gear': np.array([control.gear], dtype=np.float32),
            'speed_limit': np.array([speed_limit], dtype=np.float32),
        }
        return obs

    def clean(self):
        self._parent_actor = None
