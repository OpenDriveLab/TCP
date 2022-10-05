from distutils.log import error
import os
import json
from typing import DefaultDict
import numpy as np
import tqdm

from multiprocessing import Pool


INPUT_FRAMES = 1
FUTURE_FRAMES = 4


def gen_single_route(route_folder):

	length = len(os.listdir(os.path.join(route_folder, 'measurements')))
	if length < INPUT_FRAMES + FUTURE_FRAMES:
		return

	seq_future_x = []
	seq_future_y = []
	seq_future_theta = []
	seq_future_feature = []
	seq_future_action = []
	seq_future_action_mu = []
	seq_future_action_sigma = []
	seq_future_only_ap_brake = []


	seq_input_x = []
	seq_input_y = []
	seq_input_theta = []

	seq_front_img = []
	seq_feature = []
	seq_value = []
	seq_speed = []

	seq_action = []
	seq_action_mu = []
	seq_action_sigma = []

	seq_x_target = []
	seq_y_target = []
	seq_target_command = []

	seq_only_ap_brake = []

	full_seq_x = []
	full_seq_y = []
	full_seq_theta = []

	full_seq_feature = []
	full_seq_action = []
	full_seq_action_mu = []
	full_seq_action_sigma = []
	full_seq_only_ap_brake = []

	for i in range(length):
		with open(os.path.join(route_folder, "measurements", f"{str(i).zfill(4)}.json"), "r") as read_file:
			measurement = json.load(read_file)
			full_seq_x.append(measurement['y'])
			full_seq_y.append(measurement['x'])
			full_seq_theta.append(measurement['theta'])


		roach_supervision_data = np.load(os.path.join(route_folder, "supervision", f"{str(i).zfill(4)}.npy"), allow_pickle=True).item()
		full_seq_feature.append(roach_supervision_data['features'])
		full_seq_action.append(roach_supervision_data['action'])
		full_seq_action_mu.append(roach_supervision_data['action_mu'])
		full_seq_action_sigma.append(roach_supervision_data['action_sigma'])
		full_seq_only_ap_brake.append(roach_supervision_data['only_ap_brake'])

	for i in range(INPUT_FRAMES-1, length-FUTURE_FRAMES):

		with open(os.path.join(route_folder, "measurements", f"{str(i).zfill(4)}.json"), "r") as read_file:
			measurement = json.load(read_file)

		seq_input_x.append(full_seq_x[i-(INPUT_FRAMES-1):i+1])
		seq_input_y.append(full_seq_y[i-(INPUT_FRAMES-1):i+1])
		seq_input_theta.append(full_seq_theta[i-(INPUT_FRAMES-1):i+1])

		seq_future_x.append(full_seq_x[i+1:i+FUTURE_FRAMES+1])
		seq_future_y.append(full_seq_y[i+1:i+FUTURE_FRAMES+1])
		seq_future_theta.append(full_seq_theta[i+1:i+FUTURE_FRAMES+1])

		seq_future_feature.append(full_seq_feature[i+1:i+FUTURE_FRAMES+1])
		seq_future_action.append(full_seq_action[i+1:i+FUTURE_FRAMES+1])
		seq_future_action_mu.append(full_seq_action_mu[i+1:i+FUTURE_FRAMES+1])
		seq_future_action_sigma.append(full_seq_action_sigma[i+1:i+FUTURE_FRAMES+1])
		seq_future_only_ap_brake.append(full_seq_only_ap_brake[i+1:i+FUTURE_FRAMES+1])

		roach_supervision_data = np.load(os.path.join(route_folder, "supervision", f"{str(i).zfill(4)}.npy"), allow_pickle=True).item()
		seq_feature.append(roach_supervision_data["features"])
		seq_value.append(roach_supervision_data["value"])
		
		front_img_list = [route_folder.replace(data_path,'')+"/rgb/"f"{str(i-_).zfill(4)}.png" for _ in range(INPUT_FRAMES-1, -1, -1)]
		seq_front_img.append(front_img_list)

		seq_speed.append(measurement["speed"])

		seq_action.append(roach_supervision_data["action"])
		seq_action_mu.append(roach_supervision_data["action_mu"])
		seq_action_sigma.append(roach_supervision_data["action_sigma"])

		seq_x_target.append(measurement["y_target"])
		seq_y_target.append(measurement["x_target"])
		seq_target_command.append(measurement["target_command"])

		seq_only_ap_brake.append(roach_supervision_data["only_ap_brake"])

	return seq_future_x, seq_future_y, seq_future_theta, seq_future_feature, seq_future_action, seq_future_action_mu, seq_future_action_sigma, seq_future_only_ap_brake, seq_input_x, seq_input_y, seq_input_theta, seq_front_img, seq_feature, seq_value, seq_speed, seq_action, seq_action_mu, seq_action_sigma, seq_x_target, seq_y_target, seq_target_command, seq_only_ap_brake

def gen_sub_folder(folder_path):
	route_list = [folder for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]
	route_list = sorted(route_list)

	total_future_x = []
	total_future_y = []
	total_future_theta = []

	total_future_feature = []
	total_future_action = []
	total_future_action_mu = []
	total_future_action_sigma = []
	total_future_only_ap_brake = []

	total_input_x = []
	total_input_y = []
	total_input_theta = []

	total_front_img = []
	total_feature = []
	total_value = []
	total_speed = []

	total_action = []
	total_action_mu = []
	total_action_sigma = []

	total_x_target = []
	total_y_target = []
	total_target_command = []

	total_only_ap_brake = []

	for route in route_list:
		seq_data = gen_single_route(os.path.join(folder_path, route))
		if not seq_data:
			continue
		seq_future_x, seq_future_y, seq_future_theta, seq_future_feature, seq_future_action, seq_future_action_mu, seq_future_action_sigma, seq_future_only_ap_brake, seq_input_x, seq_input_y, seq_input_theta, seq_front_img, seq_feature, seq_value, seq_speed, seq_action, seq_action_mu, seq_action_sigma, seq_x_target, seq_y_target, seq_target_command, seq_only_ap_brake = seq_data
		total_future_x.extend(seq_future_x)
		total_future_y.extend(seq_future_y)
		total_future_theta.extend(seq_future_theta)
		total_future_feature.extend(seq_future_feature)
		total_future_action.extend(seq_future_action)
		total_future_action_mu.extend(seq_future_action_mu)
		total_future_action_sigma.extend(seq_future_action_sigma)
		total_future_only_ap_brake.extend(seq_future_only_ap_brake)
		total_input_x.extend(seq_input_x)
		total_input_y.extend(seq_input_y)
		total_input_theta.extend(seq_input_theta)
		total_front_img.extend(seq_front_img)
		total_feature.extend(seq_feature)
		total_value.extend(seq_value)
		total_speed.extend(seq_speed)
		total_action.extend(seq_action)
		total_action_mu.extend(seq_action_mu)
		total_action_sigma.extend(seq_action_sigma)
		total_x_target.extend(seq_x_target)
		total_y_target.extend(seq_y_target)
		total_target_command.extend(seq_target_command)
		total_only_ap_brake.extend(seq_only_ap_brake)

	data_dict = {}
	data_dict['future_x'] = total_future_x
	data_dict['future_y'] = total_future_y
	data_dict['future_theta'] = total_future_theta
	data_dict['future_feature'] = total_future_feature
	data_dict['future_action'] = total_future_action
	data_dict['future_action_mu'] = total_future_action_mu
	data_dict['future_action_sigma'] = total_future_action_sigma
	data_dict['future_only_ap_brake'] = total_future_only_ap_brake
	data_dict['input_x'] = total_input_x
	data_dict['input_y'] = total_input_y
	data_dict['input_theta'] = total_input_theta
	data_dict['front_img'] = total_front_img
	data_dict['feature'] = total_feature
	data_dict['value'] = total_value
	data_dict['speed'] = total_speed
	data_dict['action'] = total_action
	data_dict['action_mu'] = total_action_mu
	data_dict['action_sigma'] = total_action_sigma
	data_dict['x_target'] = total_x_target
	data_dict['y_target'] = total_y_target
	data_dict['target_command'] = total_target_command
	data_dict['only_ap_brake'] = total_only_ap_brake

	file_path = os.path.join(folder_path, "packed_data")
	np.save(file_path, data_dict)
	return len(total_future_x)


if __name__ == '__main__':
	global data_path
	data_path = "tcp_carla_data"
	towns = ["town01","town01_val","town01_addition","town02","town02_val","town03","town03_val","town03_addition", "town04","town04_val", "town04_addition", "town05", "town05_val", "town05_addition" ,"town06","town06_val", "town06_addition","town07", "town07_val", "town10", "town10_addition","town10_val"]
	pattern = "{}" # town type
	import tqdm
	total = 0
	for town in tqdm.tqdm(towns):
		number = gen_sub_folder(os.path.join(data_path, pattern.format(town)))
		total += number

	print(total)

	