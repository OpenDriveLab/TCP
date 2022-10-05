import numpy as np
from gym import spaces

from carla_gym.core.obs_manager.obs_manager import ObsManagerBase
import carla_gym.utils.transforms as trans_utils


class ObsManager(ObsManagerBase):

    def __init__(self, obs_configs):
        self._parent_actor = None
        self._route_steps = 5
        super(ObsManager, self).__init__()

    def _define_obs_space(self):
        self.obs_space = spaces.Dict({
            'lateral_dist': spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32),
            'angle_diff': spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32),
            'route_locs': spaces.Box(low=-5.0, high=5.0, shape=(self._route_steps*2,), dtype=np.float32),
            'dist_remaining': spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float32)
        })

    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor

    def get_observation(self):
        ev_transform = self._parent_actor.vehicle.get_transform()
        route_plan = self._parent_actor.route_plan

        # lateral_dist
        waypoint, road_option = route_plan[0]
        wp_transform = waypoint.transform

        d_vec = ev_transform.location - wp_transform.location
        np_d_vec = np.array([d_vec.x, d_vec.y], dtype=np.float32)
        wp_unit_forward = wp_transform.rotation.get_forward_vector()
        np_wp_unit_right = np.array([-wp_unit_forward.y, wp_unit_forward.x], dtype=np.float32)

        lateral_dist = np.abs(np.dot(np_wp_unit_right, np_d_vec))
        lateral_dist = np.clip(lateral_dist, 0, 2)

        # angle_diff
        angle_diff = np.deg2rad(np.abs(trans_utils.cast_angle(ev_transform.rotation.yaw - wp_transform.rotation.yaw)))
        angle_diff = np.clip(angle_diff, -2, 2)
        
        # route_locs
        location_list = []
        route_length = len(route_plan)
        for i in range(self._route_steps):
            if i < route_length:
                waypoint, road_option = route_plan[i]
            else:
                waypoint, road_option = route_plan[-1]

            wp_location_world_coord = waypoint.transform.location
            wp_location_actor_coord = trans_utils.loc_global_to_ref(wp_location_world_coord, ev_transform)
            location_list += [wp_location_actor_coord.x, wp_location_actor_coord.y]
        
        # dist_remaining_in_km
        dist_remaining_in_km = (self._parent_actor.route_length - self._parent_actor.route_completed)  / 1000.0

        obs = {
            'lateral_dist': np.array([lateral_dist], dtype=np.float32),
            'angle_diff': np.array([angle_diff], dtype=np.float32),
            'route_locs': np.array(location_list, dtype=np.float32),
            'dist_remaining': np.array([dist_remaining_in_km], dtype=np.float32)
        }
        return obs

    def clean(self):
        self._parent_actor = None
