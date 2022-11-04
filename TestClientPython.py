from __future__ import print_function
from vicon_dssdk import ViconDataStream
import argparse
import pickle
import zlib
from typing import Any, Dict, cast
import numpy as np
import zmq


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



parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('host', nargs='?', help="Host name, in the format of server:port", default = "localhost:801")
args = parser.parse_args()

client = ViconDataStream.Client()
i = 0
ctx = SerializingContext()
rep = ctx.socket(zmq.PUB)  # rep is short for "reply" (server side)
rep.bind('tcp://*:9999')
# rep.connect('tcp://*:9999')

try:
    # Connect to the MoCap system
    client.Connect( args.host )

    # Check the version
    print( 'Version', client.GetVersion() )

    # Check setting the buffer size works
    client.SetBufferSize( 1 )

    #Enable all the data types
    client.EnableSegmentData()
    client.EnableMarkerData()
    client.EnableUnlabeledMarkerData()
    client.EnableMarkerRayData()
    client.EnableDeviceData()
    client.EnableCentroidData()

    # Report whether the data types have been enabled
    print( 'Segments', client.IsSegmentDataEnabled() )
    print( 'Markers', client.IsMarkerDataEnabled() )
    print( 'Unlabeled Markers', client.IsUnlabeledMarkerDataEnabled() )
    print( 'Marker Rays', client.IsMarkerRayDataEnabled() )
    print( 'Devices', client.IsDeviceDataEnabled() )
    print( 'Centroids', client.IsCentroidDataEnabled() )

    # Initial setup
    HasFrame = False
    while not HasFrame:
        try:
            client.GetFrame()
            HasFrame = True
        except ViconDataStream.DataStreamException as e:
            client.GetFrame()
    
    # Set streaming mode to Frame push - page 56 in Vicon DataStream manual
    client.SetStreamMode( ViconDataStream.Client.StreamMode.EServerPush )
    print( 'Get Frame Push', client.GetFrame(), client.GetFrameNumber() )

    print( 'Frame Rate', client.GetFrameRate() )
    dt = 1 / client.GetFrameRate()

    hours, minutes, seconds, frames, subframe, fieldFlag, standard, subFramesPerFrame, userBits = client.GetTimecode()
    print( ('Timecode:', hours, 'hours', minutes, 'minutes', seconds, 'seconds', frames, 
        'frames', subframe, 'sub frame', fieldFlag, 'field flag', 
        standard, 'standard', subFramesPerFrame, 'sub frames per frame', userBits, 'user bits') )

    print( 'Total Latency', client.GetLatencyTotal() )
    print( 'Latency Samples' )
    for sampleName, sampleValue in client.GetLatencySamples().items():
        print( sampleName, sampleValue )

    print( 'Frame Rates' )
    for frameRateName, frameRateValue in client.GetFrameRates().items():
        print( frameRateName, frameRateValue )

    try:
        client.SetApexDeviceFeedback( 'BogusDevice', True )
    except ViconDataStream.DataStreamException as e:
        print( 'No Apex Devices connected' )

    client.SetAxisMapping( ViconDataStream.Client.AxisMapping.EForward, ViconDataStream.Client.AxisMapping.ELeft, ViconDataStream.Client.AxisMapping.EUp )
    xAxis, yAxis, zAxis = client.GetAxisMapping()
    print( 'X Axis', xAxis, 'Y Axis', yAxis, 'Z Axis', zAxis )

    print( 'Server Orientation', client.GetServerOrientation() )

    try:
        client.SetTimingLog( '', '' )
    except ViconDataStream.DataStreamException as e:
        print( 'Failed to set timing log' )

    try:
        client.ConfigureWireless()
    except ViconDataStream.DataStreamException as e:
        print( 'Failed to configure wireless', e )

    # Initialize arrays for storing observation data components
    # Current base position
    # Left thigh position needed for transformation to world coordinates
    l_thigh_p = np.array(client.GetSegmentGlobalTranslation("Left_thigh", "Left_thigh")[0])
    # Left thigh orientation needed for transformation to world coordinates
    l_thigh_glob_rot = np.array(client.GetSegmentGlobalRotationMatrix("Left_thigh", "Left_thigh")[0])
    # Convert to world coordinates
    l_base_p = pos_in_world_frame(global_rotation_matrix=l_thigh_glob_rot, global_translation=l_thigh_p)
    # Right thigh position needed for transformation to world coordinates
    r_thigh_p = np.array(client.GetSegmentGlobalTranslation("Right_thigh", "Right_thigh")[0])
    # Right thigh orientation needed for transformation to world coordinates
    r_thigh_glob_rot = np.array(client.GetSegmentGlobalRotationMatrix("Right_thigh", "Right_thigh")[0])
    # Convert to world coordinates
    r_base_p = pos_in_world_frame(global_rotation_matrix=r_thigh_glob_rot, global_translation=r_thigh_p)
    # Average to get base position
    base_p = 0.5 * (l_base_p + r_base_p)
    # Convert from mm to m
    base_p /= 1000.0

    # Origin
    origin = base_p.copy()
    origin[-1] = 0.0

    # Previous base position
    base_p_prev = base_p.copy()

    # Differentiate to get base velocity
    # base_vel
    base_v = (base_p - base_p_prev) / dt
    
    # Current feet position
    # Left foot
    l_foot_p = np.array(client.GetSegmentGlobalTranslation("Left_foot", "Left_foot")[0])
    # Convert from mm to m
    l_foot_p /= 1000.0
    # Adjust w.r.t. origin
    l_foot_p -= origin
    # Right foot
    r_foot_p = np.array(client.GetSegmentGlobalTranslation("Right_foot", "Right_foot")[0])
    # Convert from mm to m
    r_foot_p /= 1000.0
    # Adjust w.r.t. origin
    r_foot_p -= origin
    
    # Previous feet position
    # Left foot
    l_foot_p_prev = l_foot_p.copy()
    # Right foot
    r_foot_p_prev = r_foot_p.copy()

    # Current feet velocity
    # Left foot
    l_foot_v = (l_foot_p - l_foot_p_prev) / dt
    # Right foot
    r_foot_v = (r_foot_p - r_foot_p_prev) / dt

    # Current feet orientation
    # Left foot
    l_foot_orn = np.array(client.GetSegmentGlobalRotationQuaternion("Left_foot", "Left_foot")[0])
    # Right foot
    r_foot_orn = np.array(client.GetSegmentGlobalRotationQuaternion("Right_foot", "Right_foot")[0])

    # Previous feet orientation needed for angular velocity calculation
    # Left foot
    l_foot_orn_prev = l_foot_orn.copy()
    # Right foot
    r_foot_orn_prev = r_foot_orn.copy()

    # Current feet angular velocity
    # Left foot
    l_foot_ang_vel = compute_ang_vel(l_foot_orn, l_foot_orn_prev, dt)
    # Right foot
    r_foot_ang_vel = compute_ang_vel(r_foot_orn, r_foot_orn_prev, dt)
    
    # Current balloon position
    balloon_p = np.array(client.GetSegmentGlobalTranslation("Balloon", "Balloon")[0])
    # Convert from mm to m
    balloon_p /= 1000.0
    # Adjust w.r.t. origin
    balloon_p -= origin

    # Previous balloon position
    balloon_p_prev = balloon_p.copy()
    
    # Current balloon linear velocity
    balloon_v = (balloon_p - balloon_p_prev) / dt

    # Current balloon orientation
    balloon_orn = np.array(client.GetSegmentGlobalRotationQuaternion("Balloon", "Balloon")[0])

    # Previous balloon orientation needed for angular velocity calculation
    balloon_orn_prev = balloon_orn.copy()

    # Current balloon angular velocity
    balloon_ang_vel = compute_ang_vel(balloon_orn, balloon_orn_prev, dt)

    while i < 50000:
    # while i < 1:

        HasFrame = False
        while not HasFrame:
            try:
                client.GetFrame()
                HasFrame = True
            except ViconDataStream.DataStreamException as e:
                client.GetFrame()

        # Current base position
        # Left thigh position needed for transformation to world coordinates
        l_thigh_p = np.array(client.GetSegmentGlobalTranslation("Left_thigh", "Left_thigh")[0])
        # Left thigh orientation needed for transformation to world coordinates
        l_thigh_glob_rot = np.array(client.GetSegmentGlobalRotationMatrix("Left_thigh", "Left_thigh")[0])
        # Convert to world coordinates
        l_base_p = pos_in_world_frame(global_rotation_matrix=l_thigh_glob_rot, global_translation=l_thigh_p)
        # Right thigh position needed for transformation to world coordinates
        r_thigh_p = np.array(client.GetSegmentGlobalTranslation("Right_thigh", "Right_thigh")[0])
        # Right thigh orientation needed for transformation to world coordinates
        r_thigh_glob_rot = np.array(client.GetSegmentGlobalRotationMatrix("Right_thigh", "Right_thigh")[0])
        # Convert to world coordinates
        r_base_p = pos_in_world_frame(global_rotation_matrix=r_thigh_glob_rot, global_translation=r_thigh_p)
        # Average to get base position
        base_p = 0.5 * (l_base_p + r_base_p)
        # Convert from mm to m
        base_p /= 1000.0
        # Adjust w.r.t. origin
        base_p -= origin

        # Differentiate to get base velocity
        # base_vel
        base_v = (base_p - base_p_prev) / dt
        
        # # Current feet position
        # # Left foot
        # l_foot_p = np.array(client.GetSegmentGlobalTranslation("Left_foot", "Left_foot")[0])
        # # Convert from mm to m
        # l_foot_p /= 1000.0
        # # Adjust w.r.t. origin
        # l_foot_p -= origin
        # # Right foot
        # r_foot_p = np.array(client.GetSegmentGlobalTranslation("Right_foot", "Right_foot")[0])
        # # Convert from mm to m
        # r_foot_p /= 1000.0
        # # Adjust w.r.t. origin
        # r_foot_p -= origin

        # # Current feet velocity
        # # Left foot
        # l_foot_v = (l_foot_p - l_foot_p_prev) / dt
        # # Right foot
        # r_foot_v = (r_foot_p - r_foot_p_prev) / dt

        # # Current feet orientation
        # # Left foot
        # l_foot_orn = np.array(client.GetSegmentGlobalRotationQuaternion("Left_foot", "Left_foot")[0])
        # # Right foot
        # r_foot_orn = np.array(client.GetSegmentGlobalRotationQuaternion("Right_foot", "Right_foot")[0])

        # # Current feet angular velocity
        # # Left foot
        # l_foot_ang_vel = compute_ang_vel(l_foot_orn, l_foot_orn_prev, dt)
        # # Right foot
        # r_foot_ang_vel = compute_ang_vel(r_foot_orn, r_foot_orn_prev, dt)
        
        # # Current balloon position
        # balloon_p = np.array(client.GetSegmentGlobalTranslation("Balloon", "Balloon")[0])
        # # Convert from mm to m
        # balloon_p /= 1000.0
        # # Adjust w.r.t. origin
        # balloon_p -= origin
        
        # # Current balloon linear velocity
        # balloon_v = (balloon_p - balloon_p_prev) / dt

        # # Current balloon orientation
        # balloon_orn = np.array(client.GetSegmentGlobalRotationQuaternion("Balloon", "Balloon")[0])

        # # Current balloon angular velocity
        # balloon_ang_vel = compute_ang_vel(balloon_orn, balloon_orn_prev, dt)

        # Update previous base position
        base_p_prev = base_p.copy()
        
        # # Update previous feet position
        # # Left foot
        # l_foot_p_prev = l_foot_p.copy()
        # # Right foot
        # r_foot_p_prev = r_foot_p.copy()

        # # Update previous feet orientation needed for angular velocity calculation
        # # Left foot
        # l_foot_orn_prev = l_foot_orn.copy()
        # # Right foot
        # r_foot_orn_prev = r_foot_orn.copy()

        # # Update previous balloon position
        # balloon_p_prev = balloon_p.copy()

        # # Update previous balloon orientation needed for angular velocity calculation
        # balloon_orn_prev = balloon_orn.copy()

        # obs = np.concatenate((base_p, base_v,
        #                       l_foot_p, r_foot_p, l_foot_v, r_foot_v,
        #                       l_foot_orn, r_foot_orn, l_foot_ang_vel, r_foot_ang_vel,
        #                       balloon_p, balloon_v, balloon_orn, balloon_ang_vel))
        
        obs = np.concatenate((base_p, base_v))

        rep.send_array(obs, copy=False)

        i += 1

except ViconDataStream.DataStreamException as e:
    print( 'Handled data stream error', e )
    i += 1
