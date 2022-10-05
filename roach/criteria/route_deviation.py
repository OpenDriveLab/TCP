class RouteDeviation():

    def __init__(self, offroad_min=15, offroad_max=30, max_route_percentage=0.3):
        self._offroad_min = offroad_min
        self._offroad_max = offroad_max
        self._max_route_percentage = max_route_percentage
        self._out_route_distance = 0.0

    def tick(self, vehicle, timestamp, ref_waypoint, distance_traveled, route_length):
        ev_loc = vehicle.get_location()

        distance = ev_loc.distance(ref_waypoint.transform.location)

        # fail if off_route is True
        off_route_max = distance > self._offroad_max

        # fail if off_safe_route more than 30% of total route length
        off_route_min = False
        if distance > self._offroad_min:
            self._out_route_distance += distance_traveled
            out_route_percentage = self._out_route_distance / route_length
            if out_route_percentage > self._max_route_percentage:
                off_route_min = True

        info = None
        if off_route_max or off_route_min:
            info = {
                'step': timestamp['step'],
                'simulation_time': timestamp['relative_simulation_time'],
                'ev_loc': [ev_loc.x, ev_loc.y, ev_loc.z],
                'off_route_max': off_route_max,
                'off_route_min': off_route_min
            }
        return info
