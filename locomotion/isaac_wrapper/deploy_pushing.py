from locomotion.isaac_wrapper.deploy_policy_base import *
from absl import app
from absl import logging
import numpy as np
import time
from tqdm import tqdm
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client
import torch
from locomotion.robots import a1_robot, a1
from locomotion.robots import robot_config
from locomotion.isaac_wrapper.load_policy import *
from locomotion.isaac_wrapper.mocap import Mocap
import threading


def main(_):
	logging.info(
		"WARNING: this code executes low-level controller on the robot.")
	input("Press enter to continue...")
	# Setup
	p = setup_pybullet()
	obs_list = ["scaled_base_lin_vel",
					"scaled_base_ang_vel",
					"projected_gravity",
					"object_location",
					"target_location",
					"relative_dof",
					"scaled_dof_vel",
					"actions"]
	reach_obs_list = ["scaled_base_lin_vel",
					"scaled_base_ang_vel",
					"projected_gravity",
					"target_location",
					"relative_dof",
					"scaled_dof_vel",
					"actions"]
	mocap = Mocap(dummy=False)
	update_thread = threading.Thread(target=mocap.update, args=(),daemon=True)
		# update_thread.setDaemon(True)
	update_thread.start()
	time.sleep(1)
	# Load and init policy
	policy, actor_critic = get_pushing_policy()
	reach_policy, actor_critic = get_target_reach_policy()
	action = DEFAULT_ACTION*0
	robot = a1_robot.A1Robot(pybullet_client=p, action_repeat=4)
	observation = torch.from_numpy(get_observation(robot,p,action,obs_list,mocap)).to(torch.device('cuda'))
	action = policy(observation.float().unsqueeze(0)).cpu().detach().numpy()[0]

	# Get robot to default pose
	getup(robot,robot_config)
	stand(robot, robot_config)

	for t in range(700):
		robot.ReceiveObservation()
		observation = torch.from_numpy(get_observation(robot,p,action,obs_list,mocap)).to(torch.device('cuda'))
		action = policy(observation.float().unsqueeze(0)).cpu().detach().numpy()[0]
		action_rearranged = rearrange_joints(action)*0.25
		robot.Step(np.array(action_rearranged)+DEFAULT_ACTION, robot_config.MotorControlMode.POSITION)
		time.sleep(0.005)
		if torch.norm(observation[9:10]-observation[11:12])<0.2:
			break
  
	for t in range(500):
		robot.ReceiveObservation()
		observation = torch.from_numpy(get_observation(robot,p,action,reach_obs_list,mocap)).to(torch.device('cuda'))
		observation[9]=0
		observation[10]=0
			# print(t)
			# break
		if t%1==0:
			action = reach_policy(observation.float().unsqueeze(0)).cpu().detach().numpy()[0]
		action_rearranged = rearrange_joints(action)*0.25
		# print(action)
		robot.Step(np.array(action_rearranged)+DEFAULT_ACTION, robot_config.MotorControlMode.POSITION)
		time.sleep(0.005)
	robot.Terminate()

if __name__ == '__main__':
	app.run(main)