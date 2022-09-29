import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict

import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T

from leaderboard.autoagents import autonomous_agent

from TCP.model import TCP
from TCP.config import GlobalConfig
from team_code.planner import RoutePlanner


SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
	return 'TCPAgent'


class TCPAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file):
		self.track = autonomous_agent.Track.SENSORS
		self.alpha = 0.3
		self.status = 0
		self.steer_step = 0
		self.last_moving_status = 0
		self.last_moving_step = -1
		self.last_steers = deque()

		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		self.config = GlobalConfig()
		self.net = TCP(self.config)


		ckpt = torch.load(path_to_conf_file)
		ckpt = ckpt["state_dict"]
		new_state_dict = OrderedDict()
		for key, value in ckpt.items():
			new_key = key.replace("model.","")
			new_state_dict[new_key] = value
		self.net.load_state_dict(new_state_dict, strict = False)
		self.net.cuda()
		self.net.eval()

		self.takeover = False
		self.stop_time = 0
		self.takeover_time = 0

		self.save_path = None
		self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

		self.last_steers = deque()
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

			print (string)

			self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
			self.save_path.mkdir(parents=True, exist_ok=False)

			(self.save_path / 'rgb').mkdir()
			(self.save_path / 'meta').mkdir()
			(self.save_path / 'bev').mkdir()

	def _init(self):
		self._route_planner = RoutePlanner(4.0, 50.0)
		self._route_planner.set_route(self._global_plan, True)

		self.initialized = True

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._route_planner.mean) * self._route_planner.scale

		return gps

	def sensors(self):
				return [
				{
					'type': 'sensor.camera.rgb',
					'x': -1.5, 'y': 0.0, 'z':2.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 900, 'height': 256, 'fov': 100,
					'id': 'rgb'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': 0.0, 'y': 0.0, 'z': 50.0,
					'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
					'width': 512, 'height': 512, 'fov': 5 * 10.0,
					'id': 'bev'
					},	
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'imu'
					},
				{
					'type': 'sensor.other.gnss',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'gps'
					},
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'speed'
					}
				]

	def tick(self, input_data):
		self.step += 1

		rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		gps = input_data['gps'][1][:2]
		speed = input_data['speed'][1]['speed']
		compass = input_data['imu'][1][-1]

		if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
			compass = 0.0

		result = {
				'rgb': rgb,
				'gps': gps,
				'speed': speed,
				'compass': compass,
				'bev': bev
				}
		
		pos = self._get_position(result)
		result['gps'] = pos
		next_wp, next_cmd = self._route_planner.run_step(pos)
		result['next_command'] = next_cmd.value


		theta = compass + np.pi/2
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta), np.cos(theta)]
			])

		local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
		local_command_point = R.T.dot(local_command_point)
		result['target_point'] = tuple(local_command_point)

		return result
	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()
		tick_data = self.tick(input_data)
		if self.step < self.config.seq_len:
			rgb = self._im_transform(tick_data['rgb']).unsqueeze(0)

			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			
			return control

		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
		command = tick_data['next_command']
		if command < 0:
			command = 4
		command -= 1
		assert command in [0, 1, 2, 3, 4, 5]
		cmd_one_hot = [0] * 6
		cmd_one_hot[command] = 1
		cmd_one_hot = torch.tensor(cmd_one_hot).view(1, 6).to('cuda', dtype=torch.float32)
		speed = torch.FloatTensor([float(tick_data['speed'])]).view(1,1).to('cuda', dtype=torch.float32)
		speed = speed / 12
		rgb = self._im_transform(tick_data['rgb']).unsqueeze(0).to('cuda', dtype=torch.float32)

		tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
										torch.FloatTensor([tick_data['target_point'][1]])]
		target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)
		state = torch.cat([speed, target_point, cmd_one_hot], 1)

		pred= self.net(rgb, state, target_point)

		steer_ctrl, throttle_ctrl, brake_ctrl, metadata = self.net.process_action(pred, tick_data['next_command'], gt_velocity, target_point)

		steer_traj, throttle_traj, brake_traj, metadata_traj = self.net.control_pid(pred['pred_wp'], gt_velocity, target_point)
		if brake_traj < 0.05: brake_traj = 0.0
		if throttle_traj > brake_traj: brake_traj = 0.0

		self.pid_metadata = metadata_traj
		control = carla.VehicleControl()

		if self.status == 0:
			self.alpha = 0.3
			self.pid_metadata['agent'] = 'traj'
			control.steer = np.clip(self.alpha*steer_ctrl + (1-self.alpha)*steer_traj, -1, 1)
			control.throttle = np.clip(self.alpha*throttle_ctrl + (1-self.alpha)*throttle_traj, 0, 0.75)
			control.brake = np.clip(self.alpha*brake_ctrl + (1-self.alpha)*brake_traj, 0, 1)
		else:
			self.alpha = 0.3
			self.pid_metadata['agent'] = 'ctrl'
			control.steer = np.clip(self.alpha*steer_traj + (1-self.alpha)*steer_ctrl, -1, 1)
			control.throttle = np.clip(self.alpha*throttle_traj + (1-self.alpha)*throttle_ctrl, 0, 0.75)
			control.brake = np.clip(self.alpha*brake_traj + (1-self.alpha)*brake_ctrl, 0, 1)


		self.pid_metadata['steer_ctrl'] = float(steer_ctrl)
		self.pid_metadata['steer_traj'] = float(steer_traj)
		self.pid_metadata['throttle_ctrl'] = float(throttle_ctrl)
		self.pid_metadata['throttle_traj'] = float(throttle_traj)
		self.pid_metadata['brake_ctrl'] = float(brake_ctrl)
		self.pid_metadata['brake_traj'] = float(brake_traj)

		if control.brake > 0.5:
			control.throttle = float(0)

		if len(self.last_steers) >= 20:
			self.last_steers.popleft()
		self.last_steers.append(abs(float(control.steer)))
		#chech whether ego is turning
		# num of steers larger than 0.1
		num = 0
		for s in self.last_steers:
			if s > 0.10:
				num += 1
		if num > 10:
			self.status = 1
			self.steer_step += 1

		else:
			self.status = 0

		self.pid_metadata['status'] = self.status

		if SAVE_PATH is not None and self.step % 10 == 0:
			self.save(tick_data)
		return control

	def save(self, tick_data):
		frame = self.step // 10

		Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))

		Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))

		outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()

	def destroy(self):
		del self.net
		torch.cuda.empty_cache()