'''p3_shooting.py

   This is a demo for moving/placing an ungrounded robot

   In particular, imagine a humanoid robot.  This moves/rotates the
   pelvis frame relative to the world.

   Node:        /p3_shooting
   Broadcast:   'rightPalm' w.r.t. 'world'     geometry_msgs/TransformStamped

'''

import rclpy
import numpy as np

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

from rclpy.node                 import Node
from rclpy.time                 import Duration
from tf2_ros                    import TransformBroadcaster
from geometry_msgs.msg          import TransformStamped
from sensor_msgs.msg            import JointState
from asyncio                    import Future

# Grab the utilities
from demos.TransformHelpers     import *
from hw5code.TrajectoryUtils    import *

# Grab the general fkin from HW5 P5.
from hw5code.KinematicChain     import KinematicChain

# For Building Visualizations
from rclpy.qos                  import QoSProfile, DurabilityPolicy
from geometry_msgs.msg          import Point, Vector3, Quaternion
from std_msgs.msg               import ColorRGBA
from visualization_msgs.msg     import Marker
from visualization_msgs.msg     import MarkerArray

joint_names = {
    'left_leg': ['leftHipYaw', 'leftHipRoll', 'leftHipPitch', 'leftKneePitch', 
                    'leftAnklePitch', 'leftAnkleRoll'],

    'right_leg': ['rightHipYaw', 'rightHipRoll', 'rightHipPitch', 'rightKneePitch', 'rightAnklePitch',
                'rightAnkleRoll'],

    'left_arm': ['torsoYaw', 'torsoPitch', 'torsoRoll', 'leftShoulderPitch', 'leftShoulderRoll',
                'leftShoulderYaw', 'leftElbowPitch', 'leftForearmYaw', 'leftWristRoll', 'leftWristPitch'],

    'left_hand': ['leftThumbRoll', 'leftThumbPitch1', 'leftIndexFingerPitch1', 'leftMiddleFingerPitch1',
                'leftPinkyPitch1'],

    'right_arm': ['torsoYaw', 'torsoPitch', 'torsoRoll', 'rightShoulderPitch', 'rightShoulderRoll',
                'rightShoulderYaw', 'rightElbowPitch', 'rightForearmYaw', 'rightWristRoll', 'rightWristPitch'],

    'right_hand': ['rightThumbRoll', 'rightThumbPitch1', 'rightIndexFingerPitch1', 'rightMiddleFingerPitch1',
                'rightPinkyPitch1'],
                
    'neck': ['torsoYaw', 'torsoPitch', 'torsoRoll', 'lowerNeckPitch', 'neckYaw', 'upperNeckPitch']
}

#
#   Create the marker array to visualize.
#
def post(x, y, diameter, height):
    # Create the cyclinder marker.
    marker = Marker()
    marker.type             = Marker.CYLINDER
    marker.pose.orientation = Quaternion()
    marker.pose.position    = Point(x = x-4, y = y-4, z = height/2)
    marker.scale            = Vector3(x = diameter, y = diameter, z = height)
    marker.color            = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
    return marker

def line(x, y, z, lx, ly, lz, cr, cg, cb, ca):
    # Create the cube marker.
    marker = Marker()
    marker.type             = Marker.CUBE
    marker.pose.orientation = Quaternion()
    marker.pose.position    = Point(x = x-4, y = y-4, z = z)
    marker.scale            = Vector3(x = lx, y = ly, z = lz)
    marker.color            = ColorRGBA(r=cr, g=cg, b=cb, a=ca)
    return marker

def building_hoop():
    # Start with an empty marker list.
    markers = []

    # Building the pole of the hoop
    markers.append(post(1.0,  1.0, 0.1, 3.0))

    # Building the backboard of the hoop
    markers.append(line(1.0, 1.0, 3.0, 1.4, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0))

    # Build the Shooting box on the backboard
    markers.append(line(1.35, 1.05, 2.85, 0.1, 0.1, 0.5, 1.0, 0.0, 0.0, 1.0))
    markers.append(line(0.65, 1.05, 2.85, 0.1, 0.1, 0.5, 1.0, 0.0, 0.0, 1.0))
    markers.append(line(1.0, 1.05, 3.1-0.05, 0.7, 0.1, 0.1, 1.0, 0.0, 0.0, 1.0))

    # Build the Edge of the Backboard
    markers.append(line(1.0, 1.05, 2.55, 1.4, 0.1, 0.1, 1.0, 0.0, 0.0, 1.0))
    markers.append(line(1.0, 1.05, 3.45, 1.4, 0.1, 0.1, 1.0, 0.0, 0.0, 1.0))
    
    markers.append(line(0.35, 1.05, 3.00, 0.1, 0.1, 1.0, 1.0, 0.0, 0.0, 1.0))
    markers.append(line(1.65, 1.05, 3.00, 0.1, 0.1, 1.0, 1.0, 0.0, 0.0, 1.0))

    # Building the rim of the hoop
    markers.append(line(1.35, 1.4, 2.6, 0.1, 0.7, 0.1, 1.0, 0.0, 0.0, 1.0))
    markers.append(line(0.65, 1.4, 2.6, 0.1, 0.7, 0.1, 1.0, 0.0, 0.0, 1.0))
    markers.append(line(1.0, 1.05, 2.6, 0.7, 0.1, 0.1, 1.0, 0.0, 0.0, 1.0))
    markers.append(line(1.0, 1.75, 2.6, 0.7, 0.1, 0.1, 1.0, 0.0, 0.0, 1.0))

    # Return the list of markers
    return markers

#
#   Trajectory Generator Node Class
#
#   This inherits all the standard ROS node stuff, but adds
#     1) an update() method to be called regularly by an internal timer,
#     2) a spin() method, aware when a trajectory ends,
#     3) a shutdown() method to stop the timer.
#
#   Take the node name, the update frequency, and the trajectory class
#   as arguments.
#
class GeneratorNode(Node):
    # Initialization.
    def __init__(self, name, rate, Trajectory):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Initialize the transform broadcaster
        self.broadcaster = TransformBroadcaster(self)

        # Set up a trajectory.
        self.trajectory = Trajectory(self)
        self.jointnames = self.trajectory.jointnames()

        # Add a publisher to send the joint commands.
        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_states subscriber...")
        while(not self.count_subscribers('/joint_states')):
            pass

        # Create a future object to signal when the trajectory ends,
        # i.e. no longer returns useful data.
        self.future = Future()

        # Set up the timing so (t=0) will occur in the first update
        # cycle (dt) from now.
        self.dt    = 1.0 / float(rate)
        self.t     = -self.dt
        self.start = self.get_clock().now()+rclpy.time.Duration(seconds=self.dt)

        # Create a timer to keep calculating/sending commands.
        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, rate))





        # Prepare the publisher (latching for new subscribers).
        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             depth=1)
        self.pub2 = self.create_publisher(
            MarkerArray, '/visualization_marker_array', quality)

        # Wait for a connection to happen, so only have to send once.
        self.get_logger().info("Waiting for RVIZ...")
        #while(not self.count_subscribers('/visualization_marker_array')):
        #    pass
        self.get_logger().info("Got Passed . . . ")

        # Create the markers visualize.
        self.markers = building_hoop()

        # Add the timestamp, frame, namespace, action, and id to each marker.
        timestamp = self.get_clock().now().to_msg()
        for (i,marker) in enumerate(self.markers):
            marker.header.stamp       = timestamp
            marker.header.frame_id    = 'world'
            marker.ns                 = 'hoop'
            marker.action             = Marker.ADD
            marker.id                 = i
        
        # Create the marker array message and publish.
        arraymsg = MarkerArray()
        arraymsg.markers = self.markers
        self.pub2.publish(arraymsg)


    # Shutdown
    def shutdown(self):
        # Destroy the timer, then shut down the node.
        self.timer.destroy()
        self.destroy_node()

        # Return the current time (in ROS format).
    def now(self):
        return self.start + Duration(seconds=self.t)

    # Spin
    def spin(self):
        # Keep running (taking care of the timer callbacks and message
        # passing), until interrupted or the trajectory is complete
        # (as signaled by the future object).
        rclpy.spin_until_future_complete(self, self.future)

        # Report the reason for shutting down.
        if self.future.done():
            self.get_logger().info("Stopping: " + self.future.result())
        else:
            self.get_logger().info("Stopping: Interrupted")


    # Update - send a new joint command every time step.
    def update(self):
        # To avoid any time jitter enforce a constant time step and
        # integrate to get the current time.
        self.t += self.dt

        # Determine the corresponding ROS time (seconds since 1970).
        now = self.start + rclpy.time.Duration(seconds=self.t)

        # Compute the desired joint positions and velocities for this time.
        desired = self.trajectory.evaluate(self.t, self.dt)
        if desired is None:
            self.future.set_result("Trajectory has ended")
            return
        (q, qdot) = desired

        # Check the results.
        if not (isinstance(q, list) and isinstance(qdot, list)):
            self.get_logger().warn("(q) and (qdot) must be python lists!")
            return
        if not (len(q) == len(self.jointnames)):
            self.get_logger().warn("(q) must be same length as jointnames!")
            return
        if not (len(q) == len(self.jointnames)):
            self.get_logger().warn("(qdot) must be same length as (q)!")
            return
        if not (isinstance(q[0], float) and isinstance(qdot[0], float)):
            self.get_logger().warn("Flatten NumPy arrays before making lists!")
            return

        # Build up a command message and publish.
        cmdmsg = JointState()
        cmdmsg.header.stamp = now.to_msg()      # Current time for ROS
        cmdmsg.name         = self.jointnames   # List of joint names
        cmdmsg.position     = q                 # List of joint positions
        cmdmsg.velocity     = qdot              # List of joint velocities
        self.pub.publish(cmdmsg)

        now = self.start + Duration(seconds=self.t)
        for (i,marker) in enumerate(self.markers):
            marker.header.stamp       = now.to_msg()
            marker.header.frame_id    = 'world'
            marker.ns                 = 'hoop'
            marker.action             = Marker.ADD
            marker.id                 = i
        
        # Create the marker array message and publish.
        arraymsg = MarkerArray()
        arraymsg.markers = self.markers
        self.pub2.publish(arraymsg)


class Trajectory():
    # Initialization.
    def __init__(self, node):
        #   Set up the kinematic chain object.
        joints = self.jointnames()
        self.node = node
        self.chain_left_leg = KinematicChain(node, 'pelvis', 'leftFoot', joint_names['left_leg'])
        self.chain_right_leg = KinematicChain(node, 'pelvis', 'rightFoot', joint_names['right_leg'])
        self.chain_right_arm = KinematicChain(node, 'pelvis', 'rightPalm', joint_names['right_arm'])
        self.chain_left_arm = KinematicChain(node, 'pelvis', 'leftPalm', joint_names['left_arm'])

        # Initial q
        self.q = np.zeros((len(self.jointnames()), 1))
        self.qdot = np.zeros((len(self.jointnames()), 1))
        self.q[joints.index('torsoYaw')], self.q[joints.index('torsoPitch')], self.q[joints.index('torsoRoll')] = -0.1, 0.1, 0
        self.q[joints.index('lowerNeckPitch')], self.q[joints.index('neckYaw')], self.q[joints.index('upperNeckPitch')] = 0, 0, 0
        self.q[joints.index('rightShoulderPitch')], self.q[joints.index('rightShoulderRoll')], self.q[joints.index('rightShoulderYaw')] = -0.543, 1.519, 0.2
        self.q[joints.index('rightElbowPitch')] = 0.810
        self.q[joints.index('rightForearmYaw')] = 0.965
        self.q[joints.index('rightWristRoll')],  self.q[joints.index('rightWristPitch')],  self.q[joints.index('rightThumbRoll')] = -0.389, 0.231, 1.350
        self.q[joints.index('leftShoulderPitch')], self.q[joints.index('leftShoulderRoll')], self.q[joints.index('leftShoulderYaw')] = -0.543, -1.549, 0.710
        self.q[joints.index('leftElbowPitch')] = -0.847
        self.q[joints.index('leftForearmYaw')] = 1.216
        self.q[joints.index('leftWristRoll')],  self.q[joints.index('leftWristPitch')],  self.q[joints.index('leftThumbRoll')] = 0.235, -0.309, 0.675
        self.q[joints.index('rightHipYaw')], self.q[joints.index('rightHipRoll')], self.q[joints.index('rightHipPitch')] = -0.40, 0, -0.935
        self.q[joints.index('rightKneePitch')], self.q[joints.index('rightAnklePitch')], self.q[joints.index('rightAnkleRoll')] = 1.467, -0.452, 0
        self.q[joints.index('leftHipYaw')], self.q[joints.index('leftHipRoll')], self.q[joints.index('leftHipPitch')] = -0.169, 0, -0.935
        self.q[joints.index('leftKneePitch')], self.q[joints.index('leftAnklePitch')], self.q[joints.index('leftAnkleRoll')] = 1.467, -0.452, 0
        
        # Set up initial positions for the chain tips
        self.p_ll_world, self.R_ll_world = (np.array([0.1361, 0.115, 1.0968e-06]).reshape((-1, 1)), R_from_quat(np.array([0.99563, 0.0033751, 0.039847, -0.084332])))
        self.p_rl_world, self.R_rl_world = (np.array([0.10744, -0.18622, 1.0968e-06]).reshape((-1, 1)), R_from_quat(np.array([0.97928, 0.0079447, 0.039192, -0.19851])))
        self.p_pelvis_world, self.R_pelvis_world = (np.array([0, 0, 0.84695]).reshape((-1, 1)), R_from_quat(np.array([1, 0, 0, 0])))

        # Weighted matrix
        weights = np.ones(42)
        # weights[joints.index('leftHipPitch')], weights[joints.index('rightHipPitch')] = 10, 10
        # weights[joints.index('torsoRoll')], weights[joints.index('torsoPitch')], weights[joints.index('torsoYaw')]  = 10, 10, 10
        W = np.diag(weights)
        self.M = np.linalg.inv(W @ W)

        # Goal vector for secondary task: Keep joints near beginning position
        # self.qgoal = np.zeros((len(joints), 1))
        q0 = self.q.copy()
        self.qgoal = q0

        # Other constants
        self.lam = 1
        self.lam_s = 20

    def get_some_q(self, q, chain):
        curr_joints = joint_names[chain]
        all_joints = self.jointnames()
        indices = [all_joints.index(x) for x in curr_joints]
        return q[indices]

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!
        return joint_names['left_leg'] + joint_names['right_leg'] + joint_names['neck'] + joint_names['left_arm'][3:] + joint_names['left_hand'] + joint_names['right_arm'][3:] + joint_names['right_hand']

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        # Compute the joints.
        if t <= 3 or t >= 6:
            Tpelvis = T_from_Rp(self.R_pelvis_world, self.p_pelvis_world)

            broadcast = self.node.broadcaster
            now = self.node.now()
            
            trans = TransformStamped()
            trans.header.stamp    = now.to_msg()
            trans.header.frame_id = 'world'
            trans.child_frame_id  = 'pelvis'
            trans.transform       = Transform_from_T(Tpelvis)
            broadcast.sendTransform(trans)

            return (self.q.flatten().tolist(), self.qdot.flatten().tolist())
        
        # Desired trajectory of right palm with respect to both legs:
        elif 3 < t < 6:
            # Broadcasting pelvis and left foot
            Tpelvis = T_from_Rp(self.R_pelvis_world, self.p_pelvis_world)
            broadcast = self.node.broadcaster
            now = self.node.now()
            
            trans = TransformStamped()
            trans.header.stamp    = now.to_msg()
            trans.header.frame_id = 'world'
            trans.child_frame_id  = 'pelvis'
            trans.transform       = Transform_from_T(Tpelvis)
            broadcast.sendTransform(trans)

            p_lh_world = pxyz(0.4374, 0.072564, 0.75006 + 0.2)
            v_lh_world = pxyz(0, 0, 0)
            R_lh_world = R_from_quat(np.array([-0.31281, 0.053035, -0.046646, 0.94718]))

            p_rh_world = pxyz(0.44239, -0.28093, 0.76293 + 0.2)
            v_rh_world = pxyz(0, 0, 0)
            R_rh_world = R_from_quat(np.array([0.66249, 0.11349, -0.18655, 0.71654]))
            wd = pxyz(0, 0, 0)
            
            # Fkin
            qlast = self.q
            (p_rh_pelvis, R_rh_pelvis, Jv_rh_pelvis, Jw_rh_pelvis) = self.chain_right_arm.fkin(self.get_some_q(qlast, 'right_arm')) 
            (p_lh_pelvis, R_lh_pelvis, Jv_lh_pelvis, Jw_lh_pelvis) = self.chain_left_arm.fkin(self.get_some_q(qlast, 'left_arm')) 
            (p_ll_pelvis, R_ll_pelvis, Jv_ll_pelvis, Jw_ll_pelvis) = self.chain_left_leg.fkin(self.get_some_q(qlast, 'left_leg')) 
            (p_rl_pelvis, R_rl_pelvis, Jv_rl_pelvis, Jw_rl_pelvis) = self.chain_right_leg.fkin(self.get_some_q(qlast, 'right_leg'))

            # print("qlast: ")
            # print(qlast)
            # print("get_some_method:")
            # print(self.get_some_q(qlast, 'right_arm'))
            # print("----")

            #Jv_ll_pelvis *= 0
            #Jw_ll_pelvis *= 0

            # T matrices based on desired positions
            T_rh_world = T_from_Rp(R_rh_world, p_rh_world)
            T_lh_world = T_from_Rp(R_lh_world, p_lh_world)
            T_ll_world = T_from_Rp(self.R_ll_world, self.p_ll_world)
            T_rl_world = T_from_Rp(self.R_rl_world, self.p_rl_world)

            # Get desired positions of right hand and left hand w.r.t leg
            Td_rh_ll = np.linalg.inv(T_ll_world) @ T_rh_world
            Td_lh_ll = np.linalg.inv(T_ll_world) @ T_lh_world
            Td_rl_ll = np.linalg.inv(T_ll_world) @ T_rl_world
            Td_ll_rh = np.linalg.inv(T_rh_world) @ T_ll_world

            pd_rh_ll, Rd_rh_ll = p_from_T(Td_rh_ll), R_from_T(Td_rh_ll)
            pd_lh_ll, Rd_lh_ll = p_from_T(Td_lh_ll), R_from_T(Td_lh_ll)
            pd_rl_ll, Rd_rl_ll = p_from_T(Td_rl_ll), R_from_T(Td_rl_ll)
            pd_ll_rh, Rd_ll_rh = p_from_T(Td_ll_rh), R_from_T(Td_ll_rh)

            # T matrices based on positions from fkin
            T_ll_pelvis = T_from_Rp(R_ll_pelvis, p_ll_pelvis)
            T_rl_pelvis = T_from_Rp(R_rl_pelvis, p_rl_pelvis)
            T_rh_pelvis = T_from_Rp(R_rh_pelvis, p_rh_pelvis)
            T_lh_pelvis = T_from_Rp(R_lh_pelvis, p_lh_pelvis)

            # Get current positions of right and left hand w.r.t left leg
            T_rh_ll = np.linalg.inv(T_ll_pelvis) @ T_rh_pelvis
            T_lh_ll = np.linalg.inv(T_ll_pelvis) @ T_lh_pelvis
            T_rl_ll = np.linalg.inv(T_ll_pelvis) @ T_rl_pelvis
            T_ll_rh = np.linalg.inv(T_rh_pelvis) @ T_ll_pelvis

            p_rh_ll, R_rh_ll = p_from_T(T_rh_ll), R_from_T(T_rh_ll)
            p_lh_ll, R_lh_ll = p_from_T(T_lh_ll), R_from_T(T_lh_ll)
            p_rl_ll, R_rl_ll = p_from_T(T_rl_ll), R_from_T(T_rl_ll)
            p_ll_rh, R_ll_rh = p_from_T(T_ll_rh), R_from_T(T_ll_rh)
            
            # Get new position of pelvis with respect to world
            T_pelvis_world = T_ll_world @ np.linalg.inv(T_ll_pelvis)
            p_pelvis_world, R_pelvis_world = p_from_T(T_pelvis_world), R_from_T(T_pelvis_world)

            # Stacking Jacobians
            J_rh_ll = np.vstack((np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jv_ll_pelvis), Jv_rh_pelvis]]) - np.block([[Jv_ll_pelvis, np.zeros_like(Jv_rh_pelvis)]])),
                                np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jw_ll_pelvis), Jw_rh_pelvis]]) - np.block([[Jw_ll_pelvis, np.zeros_like(Jw_rh_pelvis)]]))))
        
            # e_rh_ll = np.vstack((ep(pd_rh_ll, p_rh_ll), eR(Rd_rh_ll, R_rh_ll)))
            e_rh_ll = np.array([0, 0, 1, 0, 0, 0]).reshape((-1, 1))

            J_lh_ll = np.vstack((np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jv_ll_pelvis), Jv_lh_pelvis]]) - np.block([[Jv_ll_pelvis, np.zeros_like(Jv_lh_pelvis)]])),
                                np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jw_ll_pelvis), Jw_lh_pelvis]]) - np.block([[Jw_ll_pelvis, np.zeros_like(Jw_lh_pelvis)]]))))
            
            e_lh_ll = np.vstack((ep(pd_lh_ll, p_lh_ll), eR(Rd_lh_ll, R_lh_ll)))

            J_rl_ll = np.vstack((np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jv_ll_pelvis), Jv_rl_pelvis]]) - np.block([[Jv_ll_pelvis, np.zeros_like(Jv_rl_pelvis)]])),
                                np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jw_ll_pelvis), Jw_rl_pelvis]]) - np.block([[Jw_ll_pelvis, np.zeros_like(Jw_rl_pelvis)]]))))

            e_rl_ll = np.vstack((ep(pd_rl_ll, p_rl_ll), eR(Rd_rl_ll, R_rl_ll)))

            J_ll_rh = np.vstack((np.transpose(R_rh_pelvis) @ (np.block([[Jv_ll_pelvis, np.zeros_like(Jv_rh_pelvis)]]) - np.block([[np.zeros_like(Jv_ll_pelvis), Jv_rh_pelvis]])),
                                np.transpose(R_rh_pelvis) @ (np.block([[Jw_ll_pelvis, np.zeros_like(Jw_rh_pelvis)]]) - np.block([[np.zeros_like(Jw_ll_pelvis), Jw_rh_pelvis]]))))
                                
            e_ll_rh = np.vstack((ep(pd_ll_rh, p_ll_rh), eR(Rd_ll_rh, R_ll_rh)))


            v = np.zeros((6, 1))
            # v[0:3] = np.transpose(self.R_ll_world) @ v_lh_world
            # v[0:3] = (np.transpose(self.R_ll_world) @ v_rh_world)
            # v[12:15] = np.transpose(self.R_ll_world) @ v_lh_world
            # v[18:21] = R_ll_rh @ (-v[6:9])
            e = np.vstack((e_rh_ll))

            J = np.block([
                # [J_rl_ll, np.zeros((6,30))],
                [J_rh_ll[:,:6], np.zeros((6,6)), J_rh_ll[:,6:9], np.zeros((6,15)), J_rh_ll[:,9:], np.zeros((6,5))],
                # [J_lh_ll[:,:6], np.zeros((6,6)), J_lh_ll[:,6:9], np.zeros((6,3)), J_lh_ll[:,9:], np.zeros((6,17))],
                # [J_ll_rh[:,:6], np.zeros((6,6)), J_ll_rh[:,6:9], np.zeros((6,15)), J_ll_rh[:,9:], np.zeros((6,5))]
            ])

            # print(f'v[6:9]: {-R_ll_rh @ v[6:9]} \n v[18:21]: {v[18:21]}')
            # print(f'v[9:12]: {-R_ll_rh @ v[9:12]} \n v[21:24]: {v[21:24]}')
            # print(f'diff v: {(-R_ll_rh @ np.transpose(R_ll_pelvis) @ Jv_rh_pelvis) - (np.transpose(R_rh_pelvis) @ -Jv_rh_pelvis)} \n')
            # print(f'diff w: {(-R_ll_rh @ np.transpose(R_ll_pelvis) @ Jw_rh_pelvis) - (np.transpose(R_rh_pelvis) @ -Jw_rh_pelvis)} \n')

            gamma = 0.2
            qdot_s = self.lam_s * (self.qgoal - qlast)
            #print(f'{qdot_s} \n')
            Jinv_W = np.linalg.inv(self.M @ np.transpose(J) @ J + gamma ** 2 * np.eye(42)) @ self.M @ np.transpose(J)
            # qdot = Jinv_W @ (v + self.lam * e) + (np.eye(42) - Jinv_W @ J) @ qdot_s
            qdot = Jinv_W @ (v + self.lam * e)
            # qdot = qdot_s
            q = qlast + dt * qdot
            self.q = q
            self.qdot = qdot
            self.p_pelvis_world, self.R_pelvis_world = p_pelvis_world, R_pelvis_world
            # check = np.array([(self.jointnames()[i], q[i]) for i in range(42)])
            # print(check)

            return (q.flatten().tolist(), qdot.flatten().tolist())

#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Initialize the generator node for 100Hz udpates, using the above
    # Trajectory class.
    generator = GeneratorNode('generator', 100, Trajectory)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()