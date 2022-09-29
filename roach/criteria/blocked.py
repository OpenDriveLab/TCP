import numpy as np


class Blocked():

    def __init__(self, speed_threshold=0.1, below_threshold_max_time=90.0):
        self._speed_threshold = speed_threshold
        self._below_threshold_max_time = below_threshold_max_time
        self._time_last_valid_state = None

    def tick(self, vehicle, timestamp):
        info = None
        linear_speed = self._calculate_speed(vehicle.get_velocity())

        if linear_speed < self._speed_threshold and self._time_last_valid_state:
            if (timestamp['relative_simulation_time'] - self._time_last_valid_state) > self._below_threshold_max_time:
                # The actor has been "blocked" for too long
                ev_loc = vehicle.get_location()
                info = {
                    'step': timestamp['step'],
                    'simulation_time': timestamp['relative_simulation_time'],
                    'ev_loc': [ev_loc.x, ev_loc.y, ev_loc.z]
                }
        else:
            self._time_last_valid_state = timestamp['relative_simulation_time']
        return info

    @staticmethod
    def _calculate_speed(carla_velocity):
        return np.linalg.norm([carla_velocity.x, carla_velocity.y])
