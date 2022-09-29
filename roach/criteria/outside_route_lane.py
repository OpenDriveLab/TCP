import carla
from carla_gym.utils.transforms import cast_angle


class OutsideRouteLane():

    def __init__(self, carla_map, vehicle_loc,
                 allowed_out_distance=1.3, max_allowed_vehicle_angle=120.0, max_allowed_waypint_angle=150.0):
        self._map = carla_map
        self._pre_ego_waypoint = self._map.get_waypoint(vehicle_loc)

        self._allowed_out_distance = allowed_out_distance
        self._max_allowed_vehicle_angle = max_allowed_vehicle_angle
        self._max_allowed_waypint_angle = max_allowed_waypint_angle

        self._outside_lane_active = False
        self._wrong_lane_active = False
        self._last_road_id = None
        self._last_lane_id = None

    def tick(self, vehicle, timestamp, distance_traveled):
        ev_loc = vehicle.get_location()
        ev_yaw = vehicle.get_transform().rotation.yaw
        self._is_outside_driving_lanes(ev_loc)
        self._is_at_wrong_lane(ev_loc, ev_yaw)

        info = None
        if self._outside_lane_active or self._wrong_lane_active:
            info = {
                'step': timestamp['step'],
                'simulation_time': timestamp['relative_simulation_time'],
                'ev_loc': [ev_loc.x, ev_loc.y, ev_loc.z],
                'distance_traveled': distance_traveled,
                'outside_lane': self._outside_lane_active,
                'wrong_lane': self._wrong_lane_active
            }
        return info

    def _is_outside_driving_lanes(self, location):
        """
        Detects if the ego_vehicle is outside driving/parking lanes
        """

        current_driving_wp = self._map.get_waypoint(location, lane_type=carla.LaneType.Driving, project_to_road=True)
        current_parking_wp = self._map.get_waypoint(location, lane_type=carla.LaneType.Parking, project_to_road=True)

        driving_distance = location.distance(current_driving_wp.transform.location)
        if current_parking_wp is not None:  # Some towns have no parking
            parking_distance = location.distance(current_parking_wp.transform.location)
        else:
            parking_distance = float('inf')

        if driving_distance >= parking_distance:
            distance = parking_distance
            lane_width = current_parking_wp.lane_width
        else:
            distance = driving_distance
            lane_width = current_driving_wp.lane_width

        self._outside_lane_active = distance > (lane_width / 2 + self._allowed_out_distance)

    def _is_at_wrong_lane(self, location, yaw):
        """
        Detects if the ego_vehicle has invaded a wrong driving lane
        """

        current_waypoint = self._map.get_waypoint(location, lane_type=carla.LaneType.Driving, project_to_road=True)
        current_lane_id = current_waypoint.lane_id
        current_road_id = current_waypoint.road_id

        # Lanes and roads are too chaotic at junctions
        if current_waypoint.is_junction:
            self._wrong_lane_active = False
        elif self._last_road_id != current_road_id or self._last_lane_id != current_lane_id:

            # Route direction can be considered continuous, except after exiting a junction.
            if self._pre_ego_waypoint.is_junction:
                # cast angle to [-180, +180)
                vehicle_lane_angle = cast_angle(
                    current_waypoint.transform.rotation.yaw - yaw)

                self._wrong_lane_active = abs(vehicle_lane_angle) > self._max_allowed_vehicle_angle

            else:
                # Check for a big gap in waypoint directions.
                waypoint_angle = cast_angle(
                    current_waypoint.transform.rotation.yaw - self._pre_ego_waypoint.transform.rotation.yaw)

                if abs(waypoint_angle) >= self._max_allowed_waypint_angle:
                    # Is the ego vehicle going back to the lane, or going out? Take the opposite
                    self._wrong_lane_active = not bool(self._wrong_lane_active)
                else:
                    # Changing to a lane with the same direction
                    self._wrong_lane_active = False

        # Remember the last state
        self._last_lane_id = current_lane_id
        self._last_road_id = current_road_id
        self._pre_ego_waypoint = current_waypoint
