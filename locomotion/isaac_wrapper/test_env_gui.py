"""Simple script for executing random actions on A1 robot."""

from absl import app
from absl import flags
import numpy as np
from tqdm import tqdm
import pybullet as p  # pytype: disable=import-error

from locomotion.envs import env_builder
from locomotion.robots import a1
from locomotion.robots import laikago
from locomotion.robots import robot_config
from locomotion.isaac_wrapper.load_policy import *
import torch 

FLAGS = flags.FLAGS
flags.DEFINE_enum('robot_type', 'A1', ['A1', 'Laikago'], 'Robot Type.')
flags.DEFINE_enum('motor_control_mode', 'Position',
                  ['Torque', 'Position', 'Hybrid'], 'Motor Control Mode.')
flags.DEFINE_bool('on_rack', False, 'Whether to put the robot on rack.')
flags.DEFINE_string('video_dir', None,
                    'Where to save video (or None for not saving).')

ROBOT_CLASS_MAP = {'A1': a1.A1, 'Laikago': laikago.Laikago}

MOTOR_CONTROL_MODE_MAP = {
    'Torque': robot_config.MotorControlMode.TORQUE,
    'Position': robot_config.MotorControlMode.POSITION,
    'Hybrid': robot_config.MotorControlMode.HYBRID
}

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
  robot = ROBOT_CLASS_MAP[FLAGS.robot_type]
  motor_control_mode = MOTOR_CONTROL_MODE_MAP[FLAGS.motor_control_mode]
  env = env_builder.build_regular_env(robot,
                                      motor_control_mode=motor_control_mode,
                                      enable_rendering=True,
                                      on_rack=FLAGS.on_rack,
                                      action_limit=(2,2,2),
                                      wrap_trajectory_generator=False)
  env.robot.SetFootFriction(1.0)
  # p.resetBasePositionAndOrientation(env.robot.quadruped, [0,-2,0.32], [0,0,0,1])
  # reference = a1.A1(pybullet_client=p, on_rack=False,action_repeat=4)
  # print(p.getNumJoints(2))
  # for i in range(-1,22):
  #   for j in range(-1,22):
    
  #   # p.setCollisionFilterGroupMask(reference.quadruped, i, collisionFilterGroup, collisionFilterMask)
  #     p.setCollisionFilterPair(env.robot.quadruped, reference.quadruped, i, j, True)
  action_low, action_high = env.action_space.low, env.action_space.high
  action_median = (action_low + action_high) / 2.
  dim_action = action_low.shape[0]
  action_selector_ids = []
  default_action =  np.array([ -0.1000,  0.8000, -1.5000, 0.1000,  0.8000, -1.5000, -0.1000,  1.0000, -1.5000,   0.1000,  1.0000,
         -1.5000])  
  policy, actor_critic = get_crouching_policy()

  action = default_action*0
  
  
  # print(get_projected_gravity([ 0.0348995, 0, 0, 0.9993908 ],p))
  
  for i in tqdm(range(1000)):
    observation = torch.from_numpy(get_observation(env.robot,p,action)).to(torch.device('cuda'))
    # print(observation[:3]) # linear velocity
    # print(observation[3:6]) # angular velocity
    # print(observation[21:33]) # motor velocities
    # break
    action = policy(observation.float().unsqueeze(0)).cpu().detach().numpy()[0]
    # print(observation, action)
    
    action_rearranged = rearrange_joints(action)*0.25
    obs, rew, done, info = env.step(np.array(action_rearranged)+default_action)



if __name__ == "__main__":
  app.run(main)
