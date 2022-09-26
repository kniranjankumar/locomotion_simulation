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
FREQ = 0.5

def get_projected_gravity(orientation, p):
    orientation = np.array(orientation)
    orientation[-1] = -orientation[-1]
    return p.multiplyTransforms([0,0,0],orientation,[0,0,-1],[0,0,0,1])[0]

def rearrange_joints(joint_obs):
  joint_obs = np.array(joint_obs)
  return np.concatenate([joint_obs[3:6],joint_obs[0:3],joint_obs[9:12],joint_obs[6:9]])
  
def get_observation(robot,p,actions):
    base_motor_angle = np.array([ -0.1000,  0.8000, -1.5000, 0.1000,  0.8000, -1.5000, -0.1000,  1.0000, -1.5000,   0.1000,  1.0000,
         -1.5000])  
    obs = [np.array(robot.GetBaseVelocity())*2,
     np.array(robot.GetBaseRollPitchYawRate())*0.25,
     np.array(get_projected_gravity(robot.GetBaseOrientation(),p)),
     np.array([-2.9,  0.1953]),
     rearrange_joints(robot.GetMotorAngles())- base_motor_angle,
     rearrange_joints(robot.GetMotorVelocities())*0.05,
     np.array(actions)]
    # obs = [np.array([0.5,0,0])*2,
    #  np.array(robot.GetBaseRollPitchYawRate())*0.25,
    #  np.array(get_projected_gravity(robot.GetBaseOrientation(),p)),
    #  rearrange_joints(robot.GetMotorAngles())- base_motor_angle,
    #  rearrange_joints(robot.GetMotorVelocities())*0.05,
    #  np.array(actions)]
    return np.concatenate(obs)
    
    
def main(_):
  logging.info(
      "WARNING: this code executes low-level controller on the robot.")
  logging.info("Make sure the robot is hang on rack before proceeding.")
  input("Press enter to continue...")

  # Construct sim env and real robot
  p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
  p.setPhysicsEngineParameter(numSolverIterations=30)
  p.setTimeStep(0.001)
  p.setGravity(0, 0, -10)
  p.setPhysicsEngineParameter(enableConeFriction=0)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  p.loadURDF("plane.urdf")
#   robot = a1_robot.A1Robot(pybullet_client=p, action_repeat=1)
  robot = a1_robot.A1Robot(pybullet_client=p, action_repeat=4)
  # Move the motors slowly to initial position
  robot.ReceiveObservation()
#   robot.ResetPose() # only in sim
  current_motor_angle = np.array(robot.GetMotorAngles())
  # print(current_motor_angle)
  default_action =  np.array([ -0.1000,  0.8000, -1.5000, 0.1000,  0.8000, -1.5000, -0.1000,  1.0000, -1.5000,   0.1000,  1.0000,
         -1.5000])  
  # base_motor_angle = np.array([0., 0.9, -1.5] * 4)
#  desired_motor_angle[-1] = -1.0
  # input("press to get to state")
  policy, actor_critic = get_crouching_policy()
  action = default_action*0
  for t in range(10):
    robot.Step(default_action, robot_config.MotorControlMode.POSITION)
  # time.sleep(3.0)
  
  for t in tqdm(range(600)):
    observation = torch.from_numpy(get_observation(robot,p,action)).to(torch.device('cuda'))
    print(observation)
    # break
    # action = policy(observation.float()).cpu().detach().numpy()
    action = policy(observation.float().unsqueeze(0)).cpu().detach().numpy()[0]
    
    action_rearranged = rearrange_joints(action)*0.25
    robot.Step(np.array(action_rearranged)+default_action, robot_config.MotorControlMode.POSITION)
    robot.ReceiveObservation()
    # current_motor_angle = np.array(robot.GetMotorAngles())
    # print(get_projected_gravity(robot.GetBaseOrientation(),p), robot.GetBaseOrientation())
    time.sleep(0.005)
  print("here")

  robot.Terminate()

if __name__ == '__main__':
  app.run(main)
