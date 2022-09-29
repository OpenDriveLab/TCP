import carla
import shapely.geometry
from carla_gym.utils.traffic_light import TrafficLightHandler


class RunRedLight():

    def __init__(self, carla_map, distance_light=30):
        self._map = carla_map
        self._last_red_light_id = None
        self._distance_light = distance_light

    def tick(self, vehicle, timestamp):
        ev_tra = vehicle.get_transform()
        ev_loc = ev_tra.location
        ev_dir = ev_tra.get_forward_vector()
        ev_extent = vehicle.bounding_box.extent.x

        tail_close_pt = ev_tra.transform(carla.Location(x=-0.8 * ev_extent))
        tail_far_pt = ev_tra.transform(carla.Location(x=-ev_extent - 1.0))
        tail_wp = self._map.get_waypoint(tail_far_pt)

        info = None
        for idx_tl in range(TrafficLightHandler.num_tl):
            traffic_light = TrafficLightHandler.list_tl_actor[idx_tl]
            tl_tv_loc = TrafficLightHandler.list_tv_loc[idx_tl]
            if tl_tv_loc.distance(ev_loc) > self._distance_light:
                continue
            if traffic_light.state != carla.TrafficLightState.Red:
                continue
            if self._last_red_light_id and self._last_red_light_id == traffic_light.id:
                continue

            for idx_wp in range(len(TrafficLightHandler.list_stopline_wps[idx_tl])):
                wp = TrafficLightHandler.list_stopline_wps[idx_tl][idx_wp]
                wp_dir = wp.transform.get_forward_vector()
                dot_ve_wp = ev_dir.x * wp_dir.x + ev_dir.y * wp_dir.y + ev_dir.z * wp_dir.z

                if tail_wp.road_id == wp.road_id and tail_wp.lane_id == wp.lane_id and dot_ve_wp > 0:
                    # This light is red and is affecting our lane
                    stop_left_loc, stop_right_loc = TrafficLightHandler.list_stopline_vtx[idx_tl][idx_wp]
                    # Is the vehicle traversing the stop line?
                    if self._is_vehicle_crossing_line((tail_close_pt, tail_far_pt), (stop_left_loc, stop_right_loc)):
                        tl_loc = traffic_light.get_location()
                        # loc_in_ev = trans_utils.loc_global_to_ref(tl_loc, ev_tra)
                        self._last_red_light_id = traffic_light.id
                        info = {
                            'step': timestamp['step'],
                            'simulation_time': timestamp['relative_simulation_time'],
                            'id': traffic_light.id,
                            'tl_loc': [tl_loc.x, tl_loc.y, tl_loc.z],
                            'ev_loc': [ev_loc.x, ev_loc.y, ev_loc.z]
                        }
        return info

    @staticmethod
    def _is_vehicle_crossing_line(seg1, seg2):
        """
        check if vehicle crosses a line segment
        """
        line1 = shapely.geometry.LineString([(seg1[0].x, seg1[0].y), (seg1[1].x, seg1[1].y)])
        line2 = shapely.geometry.LineString([(seg2[0].x, seg2[0].y), (seg2[1].x, seg2[1].y)])
        inter = line1.intersection(line2)
        return not inter.is_empty
