import os
from PIL import Image
import numpy as np
import torch 
from torch.utils.data import Dataset
from torchvision import transforms as T

from TCP.augment import hard as augmenter

class CARLA_Data(Dataset):

	def __init__(self, root, data_folders, img_aug = False):
		self.root = root
		self.img_aug = img_aug
		self._batch_read_number = 0

		self.front_img = []
		self.x = []
		self.y = []
		self.command = []
		self.target_command = []
		self.target_gps = []
		self.theta = []
		self.speed = []


		self.value = []
		self.feature = []
		self.action = []
		self.action_mu = []
		self.action_sigma = []

		self.future_x = []
		self.future_y = []
		self.future_theta = []

		self.future_feature = []
		self.future_action = []
		self.future_action_mu = []
		self.future_action_sigma = []
		self.future_only_ap_brake = []

		self.x_command = []
		self.y_command = []
		self.command = []
		self.only_ap_brake = []

		for sub_root in data_folders:
			data = np.load(os.path.join(sub_root, "packed_data.npy"), allow_pickle=True).item()

			self.x_command += data['x_target']
			self.y_command += data['y_target']
			self.command += data['target_command']

			self.front_img += data['front_img']
			self.x += data['input_x']
			self.y += data['input_y']
			self.theta += data['input_theta']
			self.speed += data['speed']

			self.future_x += data['future_x']
			self.future_y += data['future_y']
			self.future_theta += data['future_theta']

			self.future_feature += data['future_feature']
			self.future_action += data['future_action']
			self.future_action_mu += data['future_action_mu']
			self.future_action_sigma += data['future_action_sigma']
			self.future_only_ap_brake += data['future_only_ap_brake']

			self.value += data['value']
			self.feature += data['feature']
			self.action += data['action']
			self.action_mu += data['action_mu']
			self.action_sigma += data['action_sigma']
			self.only_ap_brake += data['only_ap_brake']
		self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

	def __len__(self):
		"""Returns the length of the dataset. """
		return len(self.front_img)

	def __getitem__(self, index):
		"""Returns the item at index idx. """
		data = dict()
		data['front_img'] = self.front_img[index]

		if self.img_aug:
			data['front_img'] = self._im_transform(augmenter(self._batch_read_number).augment_image(np.array(
					Image.open(self.root+self.front_img[index][0]))))
		else:
			data['front_img'] = self._im_transform(np.array(
					Image.open(self.root+self.front_img[index][0])))

		# fix for theta=nan in some measurements
		if np.isnan(self.theta[index][0]):
			self.theta[index][0] = 0.

		ego_x = self.x[index][0]
		ego_y = self.y[index][0]
		ego_theta = self.theta[index][0]

		waypoints = []
		for i in range(4):
			R = np.array([
			[np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
			[np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
			])
			local_command_point = np.array([self.future_y[index][i]-ego_y, self.future_x[index][i]-ego_x] )
			local_command_point = R.T.dot(local_command_point)
			waypoints.append(local_command_point)

		data['waypoints'] = np.array(waypoints)

		data['action'] = self.action[index]
		data['action_mu'] = self.action_mu[index]
		data['action_sigma'] = self.action_sigma[index]


		future_only_ap_brake = self.future_only_ap_brake[index]
		future_action_mu = self.future_action_mu[index]
		future_action_sigma = self.future_action_sigma[index]

		# use the average value of roach braking action when the brake is only performed by the rule-based detector
		for i in range(len(future_only_ap_brake)):
			if future_only_ap_brake[i]:
				future_action_mu[i][0] = 0.8
				future_action_sigma[i][0] = 5.5
		data['future_action_mu'] = future_action_mu
		data['future_action_sigma'] = future_action_sigma
		data['future_feature'] = self.future_feature[index]

		only_ap_brake = self.only_ap_brake[index]
		if only_ap_brake:
			data['action_mu'][0] = 0.8
			data['action_sigma'][0] = 5.5

		R = np.array([
			[np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
			[np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
			])
		local_command_point = np.array([-1*(self.x_command[index]-ego_x), self.y_command[index]-ego_y] )
		local_command_point = R.T.dot(local_command_point)
		data['target_point'] = local_command_point[:2]


		local_command_point_aim = np.array([(self.y_command[index]-ego_y), self.x_command[index]-ego_x] )
		local_command_point_aim = R.T.dot(local_command_point_aim)
		data['target_point_aim'] = local_command_point_aim[:2]

		data['target_point'] = local_command_point_aim[:2]

		data['speed'] = self.speed[index]
		data['feature'] = self.feature[index]
		data['value'] = self.value[index]
		command = self.command[index]

		# VOID = -1
		# LEFT = 1
		# RIGHT = 2
		# STRAIGHT = 3
		# LANEFOLLOW = 4
		# CHANGELANELEFT = 5
		# CHANGELANERIGHT = 6
		if command < 0:
			command = 4
		command -= 1
		assert command in [0, 1, 2, 3, 4, 5]
		cmd_one_hot = [0] * 6
		cmd_one_hot[command] = 1
		data['target_command'] = torch.tensor(cmd_one_hot)		

		self._batch_read_number += 1
		return data


def scale_and_crop_image(image, scale=1, crop_w=256, crop_h=256):
	"""
	Scale and crop a PIL image
	"""
	(width, height) = (int(image.width // scale), int(image.height // scale))
	im_resized = image.resize((width, height))
	start_x = height//2 - crop_h//2
	start_y = width//2 - crop_w//2
	cropped_image = im_resized.crop((start_y, start_x, start_y+crop_w, start_x+crop_h))

	# cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
	# cropped_image = np.transpose(cropped_image, (2,0,1))
	return cropped_image


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
	"""
	Build a rotation matrix and take the dot product.
	"""
	# z value to 1 for rotation
	xy1 = xyz.copy()
	xy1[:,2] = 1

	c, s = np.cos(r1), np.sin(r1)
	r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

	# np.dot converts to a matrix, so we explicitly change it back to an array
	world = np.asarray(r1_to_world @ xy1.T)

	c, s = np.cos(r2), np.sin(r2)
	r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
	world_to_r2 = np.linalg.inv(r2_to_world)

	out = np.asarray(world_to_r2 @ world).T
	
	# reset z-coordinate
	out[:,2] = xyz[:,2]

	return out

def rot_to_mat(roll, pitch, yaw):
	roll = np.deg2rad(roll)
	pitch = np.deg2rad(pitch)
	yaw = np.deg2rad(yaw)

	yaw_matrix = np.array([
		[np.cos(yaw), -np.sin(yaw), 0],
		[np.sin(yaw), np.cos(yaw), 0],
		[0, 0, 1]
	])
	pitch_matrix = np.array([
		[np.cos(pitch), 0, -np.sin(pitch)],
		[0, 1, 0],
		[np.sin(pitch), 0, np.cos(pitch)]
	])
	roll_matrix = np.array([
		[1, 0, 0],
		[0, np.cos(roll), np.sin(roll)],
		[0, -np.sin(roll), np.cos(roll)]
	])

	rotation_matrix = yaw_matrix.dot(pitch_matrix).dot(roll_matrix)
	return rotation_matrix


def vec_global_to_ref(target_vec_in_global, ref_rot_in_global):
	R = rot_to_mat(ref_rot_in_global['roll'], ref_rot_in_global['pitch'], ref_rot_in_global['yaw'])
	np_vec_in_global = np.array([[target_vec_in_global[0]],
								 [target_vec_in_global[1]],
								 [target_vec_in_global[2]]])
	np_vec_in_ref = R.T.dot(np_vec_in_global)
	return np_vec_in_ref[:,0]

def get_action_beta(alpha, beta):
		x = torch.zeros_like(alpha)
		x[:, 1] += 0.5
		mask1 = (alpha > 1) & (beta > 1)
		x[mask1] = (alpha[mask1]-1)/(alpha[mask1]+beta[mask1]-2)

		mask2 = (alpha <= 1) & (beta > 1)
		x[mask2] = 0.0

		mask3 = (alpha > 1) & (beta <= 1)
		x[mask3] = 1.0

		# mean
		mask4 = (alpha <= 1) & (beta <= 1)
		x[mask4] = alpha[mask4]/(alpha[mask4]+beta[mask4])

		x = x * 2 - 1

		return x


	