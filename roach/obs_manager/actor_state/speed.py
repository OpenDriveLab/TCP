import numpy as np
from gym import spaces

from carla_gym.core.obs_manager.obs_manager import ObsManagerBase


class ObsManager(ObsManagerBase):
    """
    in m/s
    """

    def __init__(self, obs_configs):
        self._parent_actor = None
        super(ObsManager, self).__init__()

    def _define_obs_space(self):
        self.obs_space = spaces.Dict({
            'speed': spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32),
            'speed_xy': spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32),
            'forward_speed': spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32)
        })

    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor

    def get_observation(self):
        velocity = self._parent_actor.vehicle.get_velocity()
        transform = self._parent_actor.vehicle.get_transform()
        forward_vec = transform.get_forward_vector()

        np_vel = np.array([velocity.x, velocity.y, velocity.z])
        np_fvec = np.array([forward_vec.x, forward_vec.y, forward_vec.z])

        speed = np.linalg.norm(np_vel)
        speed_xy = np.linalg.norm(np_vel[0:2])
        forward_speed = np.dot(np_vel, np_fvec)

        obs = {
            'speed': np.array([speed], dtype=np.float32),
            'speed_xy': np.array([speed_xy], dtype=np.float32),
            'forward_speed': np.array([forward_speed], dtype=np.float32)
        }
        return obs

    def clean(self):
        self._parent_actor = None
