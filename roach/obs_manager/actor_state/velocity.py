import numpy as np
from gym import spaces

from carla_gym.core.obs_manager.obs_manager import ObsManagerBase
import carla_gym.utils.transforms as trans_utils


class ObsManager(ObsManagerBase):

    def __init__(self, obs_configs):
        super(ObsManager, self).__init__()

    def _define_obs_space(self):
        # acc_x, acc_y: m/s2
        # vel_x, vel_y: m/s
        # vel_angular z: rad/s
        self.obs_space = spaces.Dict({
            'acc_xy': spaces.Box(low=-1e3, high=1e3, shape=(2,), dtype=np.float32),
            'vel_xy': spaces.Box(low=-1e2, high=1e2, shape=(2,), dtype=np.float32),
            'vel_ang_z': spaces.Box(low=-1e3, high=1e3, shape=(1,), dtype=np.float32)
        })

    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor

    def get_observation(self):
        ev_transform = self._parent_actor.vehicle.get_transform()
        acc_w = self._parent_actor.vehicle.get_acceleration()
        vel_w = self._parent_actor.vehicle.get_velocity()
        ang_w = self._parent_actor.vehicle.get_angular_velocity()

        acc_ev = trans_utils.vec_global_to_ref(acc_w, ev_transform.rotation)
        vel_ev = trans_utils.vec_global_to_ref(vel_w, ev_transform.rotation)

        obs = {
            'acc_xy': np.array([acc_ev.x, acc_ev.y], dtype=np.float32),
            'vel_xy': np.array([vel_ev.x, vel_ev.y], dtype=np.float32),
            'vel_ang_z': np.array([ang_w.z], dtype=np.float32)
        }
        return obs

    def clean(self):
        self._parent_actor = None
