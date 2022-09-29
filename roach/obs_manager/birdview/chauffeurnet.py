import numpy as np
import carla
from gym import spaces
import cv2 as cv
from collections import deque
from pathlib import Path
import h5py

from roach.utils.traffic_light import TrafficLightHandler


COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_YELLOW_2 = (160, 160, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_5 = (46, 52, 54)


def tint(color, factor):
	r, g, b = color
	r = int(r + (255-r) * factor)
	g = int(g + (255-g) * factor)
	b = int(b + (255-b) * factor)
	r = min(r, 255)
	g = min(g, 255)
	b = min(b, 255)
	return (r, g, b)


class ObsManager():
	def __init__(self, obs_configs, criteria_stop=None):
		self._width = int(obs_configs['width_in_pixels'])
		self._pixels_ev_to_bottom = obs_configs['pixels_ev_to_bottom']
		self._pixels_per_meter = obs_configs['pixels_per_meter']
		self._history_idx = obs_configs['history_idx']
		self._scale_bbox = obs_configs.get('scale_bbox', True)
		self._scale_mask_col = obs_configs.get('scale_mask_col', 1.1)

		self._history_queue = deque(maxlen=20)

		self._image_channels = 3
		self._masks_channels = 3 + 3*len(self._history_idx)
		self._parent_actor = None
		self._world = None

		self._map_dir = Path(__file__).resolve().parent / 'maps'

		self._criteria_stop =criteria_stop

		super(ObsManager, self).__init__()

	def attach_ego_vehicle(self, ego_vehicle):
		self._parent_actor = ego_vehicle
		self._world = self._parent_actor.get_world()

		maps_h5_path = self._map_dir / (self._world.get_map().name + '.h5')
		with h5py.File(maps_h5_path, 'r', libver='latest', swmr=True) as hf:
			self._road = np.array(hf['road'], dtype=np.uint8)
			self._lane_marking_all = np.array(hf['lane_marking_all'], dtype=np.uint8)
			self._lane_marking_white_broken = np.array(hf['lane_marking_white_broken'], dtype=np.uint8)

			self._world_offset = np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)
			assert np.isclose(self._pixels_per_meter, float(hf.attrs['pixels_per_meter']))

		self._distance_threshold = np.ceil(self._width / self._pixels_per_meter)

	@staticmethod
	def _get_stops(criteria_stop):
		stop_sign = criteria_stop._target_stop_sign
		stops = []
		if (stop_sign is not None) and (not criteria_stop._stop_completed):
			bb_loc = carla.Location(stop_sign.trigger_volume.location)
			bb_ext = carla.Vector3D(stop_sign.trigger_volume.extent)
			bb_ext.x = max(bb_ext.x, bb_ext.y)
			bb_ext.y = max(bb_ext.x, bb_ext.y)
			trans = stop_sign.get_transform()
			stops = [(carla.Transform(trans.location, trans.rotation), bb_loc, bb_ext)]
		return stops

	def get_observation(self, route_plan):
		ev_transform = self._parent_actor.get_transform()
		ev_loc = ev_transform.location
		ev_rot = ev_transform.rotation
		ev_bbox = self._parent_actor.bounding_box

		def is_within_distance(w):
			c_distance = abs(ev_loc.x - w.location.x) < self._distance_threshold \
				and abs(ev_loc.y - w.location.y) < self._distance_threshold \
				and abs(ev_loc.z - w.location.z) < 8.0
			c_ev = abs(ev_loc.x - w.location.x) < 1.0 and abs(ev_loc.y - w.location.y) < 1.0
			return c_distance and (not c_ev)
			

		vehicle_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Vehicles)
		walker_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Pedestrians)
		if self._scale_bbox:
			vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance, 1.0)
			walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance, 2.0)
		else:
			vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance)
			walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance)

		tl_green = TrafficLightHandler.get_stopline_vtx(ev_loc, 0)
		tl_yellow = TrafficLightHandler.get_stopline_vtx(ev_loc, 1)
		tl_red = TrafficLightHandler.get_stopline_vtx(ev_loc, 2)
		stops = self._get_stops(self._criteria_stop)

		self._history_queue.append((vehicles, walkers, tl_green, tl_yellow, tl_red, stops))

		M_warp = self._get_warp_transform(ev_loc, ev_rot)

		# objects with history
		vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks \
			= self._get_history_masks(M_warp)

		# road_mask, lane_mask
		road_mask = cv.warpAffine(self._road, M_warp, (self._width, self._width)).astype(np.bool)
		lane_mask_all = cv.warpAffine(self._lane_marking_all, M_warp, (self._width, self._width)).astype(np.bool)
		lane_mask_broken = cv.warpAffine(self._lane_marking_white_broken, M_warp,
										 (self._width, self._width)).astype(np.bool)

		# route_mask
		route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
		route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
								   for wp, _ in route_plan[0:80]])
		route_warped = cv.transform(route_in_pixel, M_warp)
		cv.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=16)
		route_mask = route_mask.astype(np.bool)

		# ev_mask
		ev_mask = self._get_mask_from_actor_list([(ev_transform, ev_bbox.location, ev_bbox.extent)], M_warp)
		ev_mask_col = self._get_mask_from_actor_list([(ev_transform, ev_bbox.location,
													   ev_bbox.extent*self._scale_mask_col)], M_warp)
		# render
		image = np.zeros([self._width, self._width, 3], dtype=np.uint8)
		image[road_mask] = COLOR_ALUMINIUM_5
		image[route_mask] = COLOR_ALUMINIUM_3
		image[lane_mask_all] = COLOR_MAGENTA
		image[lane_mask_broken] = COLOR_MAGENTA_2

		h_len = len(self._history_idx)-1
		for i, mask in enumerate(stop_masks):
			image[mask] = tint(COLOR_YELLOW_2, (h_len-i)*0.2)
		for i, mask in enumerate(tl_green_masks):
			image[mask] = tint(COLOR_GREEN, (h_len-i)*0.2)
		for i, mask in enumerate(tl_yellow_masks):
			image[mask] = tint(COLOR_YELLOW, (h_len-i)*0.2)
		for i, mask in enumerate(tl_red_masks):
			image[mask] = tint(COLOR_RED, (h_len-i)*0.2)

		for i, mask in enumerate(vehicle_masks):
			image[mask] = tint(COLOR_BLUE, (h_len-i)*0.2)
		for i, mask in enumerate(walker_masks):
			image[mask] = tint(COLOR_CYAN, (h_len-i)*0.2)

		image[ev_mask] = COLOR_WHITE
		# image[obstacle_mask] = COLOR_BLUE

		# masks
		c_road = road_mask * 255
		c_route = route_mask * 255
		c_lane = lane_mask_all * 255
		c_lane[lane_mask_broken] = 120

		# masks with history
		c_tl_history = []
		for i in range(len(self._history_idx)):
			c_tl = np.zeros([self._width, self._width], dtype=np.uint8)
			c_tl[tl_green_masks[i]] = 80
			c_tl[tl_yellow_masks[i]] = 170
			c_tl[tl_red_masks[i]] = 255
			c_tl[stop_masks[i]] = 255
			c_tl_history.append(c_tl)

		c_vehicle_history = [m*255 for m in vehicle_masks]
		c_walker_history = [m*255 for m in walker_masks]

		masks = np.stack((c_road, c_route, c_lane, *c_vehicle_history, *c_walker_history, *c_tl_history), axis=2)
		masks = np.transpose(masks, [2, 0, 1])

		obs_dict = {'rendered': image, 'masks': masks}

		# self._parent_actor.collision_px = np.any(ev_mask_col & walker_masks[-1])

		return obs_dict

	def _get_history_masks(self, M_warp):
		qsize = len(self._history_queue)
		vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks = [], [], [], [], [], []
		for idx in self._history_idx:
			idx = max(idx, -1 * qsize)

			vehicles, walkers, tl_green, tl_yellow, tl_red, stops = self._history_queue[idx]

			vehicle_masks.append(self._get_mask_from_actor_list(vehicles, M_warp))
			walker_masks.append(self._get_mask_from_actor_list(walkers, M_warp))
			tl_green_masks.append(self._get_mask_from_stopline_vtx(tl_green, M_warp))
			tl_yellow_masks.append(self._get_mask_from_stopline_vtx(tl_yellow, M_warp))
			tl_red_masks.append(self._get_mask_from_stopline_vtx(tl_red, M_warp))
			stop_masks.append(self._get_mask_from_actor_list(stops, M_warp))

		return vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks

	def _get_mask_from_stopline_vtx(self, stopline_vtx, M_warp):
		mask = np.zeros([self._width, self._width], dtype=np.uint8)
		for sp_locs in stopline_vtx:
			stopline_in_pixel = np.array([[self._world_to_pixel(x)] for x in sp_locs])
			stopline_warped = cv.transform(stopline_in_pixel, M_warp)
			stopline_warped = np.round(stopline_warped).astype(np.int32)
			cv.line(mask, tuple(stopline_warped[0, 0]), tuple(stopline_warped[1, 0]),
					color=1, thickness=6)
		return mask.astype(np.bool)

	def _get_mask_from_actor_list(self, actor_list, M_warp):
		mask = np.zeros([self._width, self._width], dtype=np.uint8)
		for actor_transform, bb_loc, bb_ext in actor_list:

			corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),
					   carla.Location(x=bb_ext.x, y=-bb_ext.y),
					   carla.Location(x=bb_ext.x, y=0),
					   carla.Location(x=bb_ext.x, y=bb_ext.y),
					   carla.Location(x=-bb_ext.x, y=bb_ext.y)]
			corners = [bb_loc + corner for corner in corners]

			corners = [actor_transform.transform(corner) for corner in corners]
			corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])
			corners_warped = cv.transform(corners_in_pixel, M_warp)

			cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
		return mask.astype(np.bool)

	@staticmethod
	def _get_surrounding_actors(bbox_list, criterium, scale=None):
		actors = []
		for bbox in bbox_list:
			is_within_distance = criterium(bbox)
			if is_within_distance:
				bb_loc = carla.Location()
				bb_ext = carla.Vector3D(bbox.extent)
				if scale is not None:
					bb_ext = bb_ext * scale
					bb_ext.x = max(bb_ext.x, 0.8)
					bb_ext.y = max(bb_ext.y, 0.8)

				actors.append((carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext))
		return actors

	def _get_warp_transform(self, ev_loc, ev_rot):
		ev_loc_in_px = self._world_to_pixel(ev_loc)
		yaw = np.deg2rad(ev_rot.yaw)

		forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
		right_vec = np.array([np.cos(yaw + 0.5*np.pi), np.sin(yaw + 0.5*np.pi)])

		bottom_left = ev_loc_in_px - self._pixels_ev_to_bottom * forward_vec - (0.5*self._width) * right_vec
		top_left = ev_loc_in_px + (self._width-self._pixels_ev_to_bottom) * forward_vec - (0.5*self._width) * right_vec
		top_right = ev_loc_in_px + (self._width-self._pixels_ev_to_bottom) * forward_vec + (0.5*self._width) * right_vec

		src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
		dst_pts = np.array([[0, self._width-1],
							[0, 0],
							[self._width-1, 0]], dtype=np.float32)
		return cv.getAffineTransform(src_pts, dst_pts)

	def _world_to_pixel(self, location, projective=False):
		"""Converts the world coordinates to pixel coordinates"""
		x = self._pixels_per_meter * (location.x - self._world_offset[0])
		y = self._pixels_per_meter * (location.y - self._world_offset[1])

		if projective:
			p = np.array([x, y, 1], dtype=np.float32)
		else:
			p = np.array([x, y], dtype=np.float32)
		return p

	def _world_to_pixel_width(self, width):
		"""Converts the world units to pixel units"""
		return self._pixels_per_meter * width

	def clean(self):
		self._parent_actor = None
		self._world = None
		self._history_queue.clear()
