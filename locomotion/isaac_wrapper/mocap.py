from __future__ import print_function
# from vicon_dssdk import ViconDataStream
import argparse
import pickle
import zlib
from typing import Any, Dict, cast
import numpy as np
import zmq
import pybullet as p
import time
from multiprocessing import Process as Process
import threading

# PyZMQ class to send information
class SerializingSocket(zmq.Socket):
    """A class with some extra serialization methods
    send_zipped_pickle is just like send_pyobj, but uses
    zlib to compress the stream before sending.
    send_array sends numpy arrays with metadata necessary
    for reconstructing the array on the other side (dtype,shape).
    """
 
    def send_zipped_pickle(
        self, obj: Any, flags: int = 0, protocol: int = pickle.HIGHEST_PROTOCOL
    ) -> None:
        """pack and compress an object with pickle and zlib."""
        pobj = pickle.dumps(obj, protocol)
        zobj = zlib.compress(pobj)
        print('zipped pickle is %i bytes' % len(zobj))
        return self.send(zobj, flags=flags)
 
    def recv_zipped_pickle(self, flags: int = 0) -> Any:
        """reconstruct a Python object sent with zipped_pickle"""
        zobj = self.recv(flags)
        pobj = zlib.decompress(zobj)
        return pickle.loads(pobj)
 
    def send_array(
        self, A: np.ndarray, flags: int = 0, copy: bool = True, track: bool = False
    ) -> Any:
        """send a numpy array with metadata"""
        md = dict(
            dtype=str(A.dtype),
            shape=A.shape,
        )
        self.send_json(md, flags | zmq.SNDMORE)
        return self.send(A, flags, copy=copy, track=track)
 
    def recv_array(
        self, flags: int = 0, copy: bool = True, track: bool = False
    ) -> np.ndarray:
        """recv a numpy array"""
        md = cast(Dict[str, Any], self.recv_json(flags=flags))
        msg = self.recv(flags=flags, copy=copy, track=track)
        A = np.frombuffer(msg, dtype=md['dtype'])
        # print(md['dtype'])
        return A.reshape(md['shape'])
 
class SerializingContext(zmq.Context[SerializingSocket]):
    _socket_class = SerializingSocket
 
# Transform position from body to world frame
def pos_in_world_frame(
    global_rotation_matrix: np.ndarray,
    global_translation: np.ndarray,
    pos_b: np.ndarray = np.array([0., 0., 315.])
    )-> np.ndarray:
    pos_w = global_translation + global_rotation_matrix.T @ pos_b
    return pos_w
 
# Compute angular velocity given orientation in quaternion
def compute_ang_vel(
    q_curr: np.ndarray,
    q_prev: np.ndarray,
    dt: float
    ) -> np.ndarray:
    # Quaternions in Vicon are of the form (x, y, z, w) where w is the
    # real component. This convention is the same as that of PyBullet.
    # Need to transform this to (w, x, y, z) only for these calculations.
    q_curr = q_curr[[3, 0, 1, 2]]
    q_prev = q_prev[[3, 0, 1, 2]]
    q_dot = (q_curr - q_prev) / dt
    E = np.array([[-q_curr[1], q_curr[0], -q_curr[3], q_curr[2]],
                  [-q_curr[2], q_curr[3], q_curr[0], -q_curr[1]],
                  [-q_curr[3], -q_curr[2], q_curr[1], q_curr[0]]])
    return 2 * E @ q_dot
 
# parser = argparse.ArgumentParser(description=__doc__)
# parser.add_argument('host', nargs='?', help="Host name, in the format of server:port", default = "localhost:801")
# args = parser.parse_args()
 
# client = ViconDataStream.Client()
# i = 0
class Mocap:
    def __init__(self,dummy=True):
        self.dummy = dummy
        if not dummy:
            ctx = SerializingContext()
            self.rep = ctx.socket(zmq.SUB)  # rep is short for "reply" (server side)
            self.rep.connect('tcp://192.168.1.103:9999')
            self.rep.subscribe(b"")
            print("waiting")
        self.obs = np.zeros(12)
        self.abs_target_pos = None
        self.door_config = None
        self.count = 0
        self.xy = np.array([1,0])
        # robot xyzrpy + target xyzrpy
    
    def update(self):
        if not self.dummy:
            while True:
                    obs_ = self.rep.recv_array(copy=False).copy()
                    # print(obs_)
                    if not np.isnan(obs_).any():
                        self.obs = obs_.copy()
                    else:obs(0.001)
                # print("here")
                # time.sleep(1)    
            # while np.isnan(self.obs).any():
                # self.obs = self.rep.recv_array(copy=False)
    def get_puck_pos(self):
        # return np.array([0,0])
        # self.update()
        
        # print("target_pos",self.obs[6:9]/10)
        obs = self.obs
        # print(obs)
        abs_target_pos =  obs[7:10]/1000
        # print(abs_target_pos)
        robot_pos = obs[:3]/1000
        robot_q = obs[3:7]
        pos,q = p.invertTransform(robot_pos,robot_q)
        relative_target_pos,_ = p.multiplyTransforms(pos,q,abs_target_pos,[0,0,0,1])
        xy =  np.array(relative_target_pos)[:-1]
        # if np.isnan(xy).any():
        #     xy = np.array([0,1])
        return xy
    
    def get_target_pos(self):
        # return np.array([0,0])
        # self.update()
        
        # print("target_pos",self.obs[6:9]/10)
        obs = self.obs
        
        # if not isinstance(self.abs_target_pos,np.ndarray):
        # self.abs_target_pos =  obs[10:13]/1000
        self.abs_target_pos =  obs[14:17]/1000
        

        # if not self.abs_target_pos:
            
        robot_pos = obs[:3]/1000
        # print(abs_target_pos,robot_pos)
        robot_q = obs[3:7]
        pos,q = p.invertTransform(robot_pos,robot_q)
        relative_target_pos,_ = p.multiplyTransforms(pos,q,self.abs_target_pos,[0,0,0,1])
        # print(np.array(relative	# for t in range(600):
	# 	robot.ReceiveObservation()
	# 	observation = torch.from_numpy(get_observation(robot,p,action,obs_list_crouch,mocap)).to(torch.device('cuda'))
	# 	if t%4==0:
	# 		action = crouch_policy(observation.float().unsqueeze(0)).cpu().detach().numpy()[0]
	# 	action_rearranged = rearrange_joints(action)*0.25
	# 	robot.Step(np.array(action_rearranged)+DEFAULT_ACTION, robot_config.MotorControlMode.POSITION)
	# 	time.sleep(0.005)_target_pos))
        xy =  np.array(relative_target_pos)[:-1]
        if not np.isnan(xy).any():
            self.xy = xy
        # print(xy)
        # if np.linalg.norm(xy)<0.1:
        #     return xy*0
        # print(xy)
        # if np.linalg.norm(xy) < 0.1:
        #     xy *=0
        return self.xy
    
    def get_door_config(self):
        if self.dummy:
            self.count +=1
            
            # return np.array(self.door_config[self.count][:2]), self.door_config[self.count][-1]*0
            
            return np.array([1.5,0]),0
        obs = self.obs
        door_pos = obs[7:10]/1000
        door_q = obs[10:14]
        door_angle = p.getEulerFromQuaternion(door_q)
        door_frame_pos = obs[14:17]/1000
        robot_pos = obs[:3]/1000
        robot_q = obs[3:7]
        pos,q = p.invertTransform(robot_pos,robot_q)
        relative_door_frame_pos,_ = p.multiplyTransforms(pos,q,door_frame_pos,[0,0,0,1])
        
        # print(relative_door_frame_pos[:-1], door_angle[2])#, door_frame_pos)
        # return np.array([1.5,0]), 0
        # return np.array([0.4277747869491577, 0.029472736120224]), 0.006
        return np.array(relative_door_frame_pos[:-1])-np.array([0.0,0]), door_angle[2]
    
    def get_robot_pos(self):
        return self.obs[:3]/1000
    
    def get_robot_q(self):
        return self.obs[3:7]

    def print_target(self):
        while True:
            # mocap.update()
            print(self.get_target_pos())
            
if __name__ == '__main__':
    mocap = Mocap(False)
    # update_thread.start()
    # time.sleep(.2)

    # mocap.update()
    while True:
        mocap.update()
        # print(mocap.get_door_config())
        print(mocap.get_target_pos(), mocap.get_puck_pos())
        # mocap.update()
        # time.sleep(1)
        