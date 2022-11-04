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
# from locomotion.isaac_wrapper.mocap import Mocap

global count 
count = -250
# DEFAULT_ACTION =  np.array([ -0.1000,  0.8000, -1.5000, 
# 							0.1000,  0.8000, -1.5000, 
# 							-0.1000,  1.0000, -1.5000,   
# 							0.1000,  1.0000, -1.5000])
DEFAULT_ACTION =  np.array([ -0.1000,  0.7000, -1.4000, 
							0.1000,  0.7000, -1.4000, 
							-0.1000,  0.7000, -1.4000,   
							0.1000,  0.7000, -1.4000])
def get_projected_gravity(orientation, p):
	return quat_rotate_inverse(orientation,[0,0,-1],p)

def quat_rotate_inverse(q,v,p):
	orientation = np.array(q)
	orientation[-1] = -orientation[-1]
	return p.multiplyTransforms([0,0,0],orientation,v,[0,0,0,1])[0]

def rearrange_joints(joint_obs):
	joint_obs = np.array(joint_obs)
	return np.concatenate([joint_obs[3:6],joint_obs[0:3],joint_obs[9:12],joint_obs[6:9]])

def get_observation(robot,p,actions,obs_list,mocap):
	# mocap.update()
	global count
	rpy = list(robot.GetBaseRollPitchYaw())
	base_quat = p.getQuaternionFromEuler(rpy)
	target_pos = mocap.get_target_pos().copy()
	puck_pos = mocap.get_puck_pos().copy()
	# target_pos = np.array([1,0])
	# door_state,door_angle = mocap.get_door_config()
	door_state = np.array([0,0])
	door_angle = 0
	# print(base_quat,robot.GetBaseVelocity())
	# duck_ht = -1.4+abs(count/250)*0.8#+abs(count/1000)
	count +=1
	# duck_ht = -1.0 if count>0 else 1.0
	# duck_ht = max(-count/500,-1)
	duck_ht = -1.0
	obs_dict = {"scaled_base_lin_vel": np.array(quat_rotate_inverse(base_quat, robot.GetBaseVelocity(), p))*2,
					"scaled_base_ang_vel": np.array(quat_rotate_inverse(base_quat, robot.GetBaseRollPitchYawRate(), p))*0.25,
					"projected_gravity": np.array(get_projected_gravity(base_quat,p)),
					# "crouch_target": np.array([-2.9,  0.1953]),
                 	"door_state": np.array(door_state),
            		"door_angle": np.array([door_angle,]),
					"crouch_target":np.array([duck_ht,]),
					"object_location": puck_pos,
					"target_location": target_pos,
					"relative_dof": rearrange_joints(robot.GetMotorAngles())- DEFAULT_ACTION,
					"scaled_dof_vel": rearrange_joints(robot.GetMotorVelocities())*0.05,
					"actions": np.array(actions)}
	obs = np.concatenate([obs_dict[obs_] for obs_ in obs_list]) 
	print(obs_dict["object_location"], obs_dict["target_location"])
	if np.isnan(obs).any():
		assert False
	return obs
	
def getup(robot, robot_config):
	for t in range(100):
		robot.ReceiveObservation()
		current_motor_angle = np.array(robot.GetMotorAngles())
		blend_ratio = np.minimum(t / 100., 1)
		blend_action = (1 - blend_ratio) * current_motor_angle + blend_ratio * DEFAULT_ACTION
		robot.Step(blend_action, robot_config.MotorControlMode.POSITION)

		time.sleep(0.005)

def stand(robot, robot_config):
	for t in range(1000):
		start = time.time()
		robot.ReceiveObservation()
		# print(rearrange_joints(robot.GetMotorAngles())- DEFAULT_ACTION)
		robot.Step(DEFAULT_ACTION, robot_config.MotorControlMode.POSITION)
		# print(time.time()-start)
		time.sleep(0.005)

def setup_pybullet():
	p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
	p.setPhysicsEngineParameter(numSolverIterations=30)
	p.setTimeStep(0.001)
	p.setGravity(0, 0, -10)
	p.setPhysicsEngineParameter(enableConeFriction=0)
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	p.loadURDF("plane.urdf")
	return p
	

