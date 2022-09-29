from carla_gym.utils.traffic_light import TrafficLightHandler


class EncounterLight():

    def __init__(self, dist_threshold=7.5):
        self._last_light_id = None
        self._dist_threshold = dist_threshold

    def tick(self, vehicle, timestamp):
        info = None

        light_state, light_loc, light_id = TrafficLightHandler.get_light_state(
            vehicle, dist_threshold=self._dist_threshold)

        if light_id is not None:
            if light_id != self._last_light_id:
                self._last_light_id = light_id
                info = {
                    'step': timestamp['step'],
                    'simulation_time': timestamp['relative_simulation_time'],
                    'id': light_id,
                    'tl_loc': light_loc.tolist()
                }

        return info
