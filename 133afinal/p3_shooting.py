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
#   Create the marker array to visualize the basketball hoop.
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
    markers.append(post(1.0 + 3,  1.0 - 1, 0.1, 3.0))

    # Building the backboard of the hoop
    markers.append(line(1.0 + 3, 1.0 - 1, 3.0, 1.4, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0))

    # Build the Shooting box on the backboard
    markers.append(line(1.35 + 3, 1.05 - 1, 2.85, 0.1, 0.1, 0.5, 1.0, 0.0, 0.0, 1.0))
    markers.append(line(0.65 + 3, 1.05 - 1, 2.85, 0.1, 0.1, 0.5, 1.0, 0.0, 0.0, 1.0))
    markers.append(line(1.0 + 3, 1.05 - 1, 3.1-0.05, 0.7, 0.1, 0.1, 1.0, 0.0, 0.0, 1.0))

    # Build the Edge of the Backboard
    markers.append(line(1.0 + 3, 1.05 - 1, 2.55, 1.4, 0.1, 0.1, 1.0, 0.0, 0.0, 1.0))
    markers.append(line(1.0 + 3, 1.05 - 1, 3.45, 1.4, 0.1, 0.1, 1.0, 0.0, 0.0, 1.0))
    
    markers.append(line(0.35 + 3, 1.05 - 1, 3.00, 0.1, 0.1, 1.0, 1.0, 0.0, 0.0, 1.0))
    markers.append(line(1.65 + 3, 1.05 - 1, 3.00, 0.1, 0.1, 1.0, 1.0, 0.0, 0.0, 1.0))

    # Building the rim of the hoop
    markers.append(line(1.35 + 3, 1.4 - 1, 2.6, 0.1, 0.7, 0.1, 1.0, 0.0, 0.0, 1.0))
    markers.append(line(0.65 + 3, 1.4 - 1, 2.6, 0.1, 0.7, 0.1, 1.0, 0.0, 0.0, 1.0))
    markers.append(line(1.0 + 3, 1.05 - 1, 2.6, 0.7, 0.1, 0.1, 1.0, 0.0, 0.0, 1.0))
    markers.append(line(1.0 + 3, 1.75 - 1, 2.6, 0.7, 0.1, 0.1, 1.0, 0.0, 0.0, 1.0))

    # Return the list of markers
    return markers

def ball(radius, p, msg):
     # Create the sphere marker.
    diam        = 2 * radius
    marker = Marker()
    marker.header.frame_id  = "world"
    marker.header.stamp     = msg
    marker.action           = Marker.ADD
    marker.ns               = "point"
    marker.id               = 1
    marker.type             = Marker.SPHERE
    marker.pose.orientation = Quaternion()
    marker.pose.position    = Point_from_p(p)
    marker.scale            = Vector3(x = diam, y = diam, z = diam)
    marker.color            = ColorRGBA(r=1.0, g=95.0/255.0, b=21.0/255.0, a=0.8)
    return marker


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

        # Initialize the ball position, velocity, set the acceleration.
        self.radius = 0.2

        self.p = np.array([0.0-4.0, 0.0-4.0, 1.0+self.radius]).reshape((3,1))
        self.v = np.array([1.0, 0.1,  5.0       ]).reshape((3,1))
        self.a = np.array([0.0, 0.0, -9.81      ]).reshape((3,1))

        # Add the timestamp, frame, namespace, action, and id to each marker.
        timestamp = self.get_clock().now().to_msg()
        for (i,marker) in enumerate(self.markers):
            marker.header.stamp       = timestamp
            marker.header.frame_id    = 'world'
            marker.ns                 = 'hoop'
            marker.action             = Marker.ADD
            marker.id                 = i+1
        
        # Create the marker array message and publish.
        self.arraymsg = MarkerArray()

        self.marker = ball(self.radius, self.p, self.get_clock().now().to_msg())

        
        self.arraymsg.markers = self.markers
        self.arraymsg.markers.append(self.marker)
        self.pub2.publish(self.arraymsg)


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

        # Integrate the velocity, then the position.
        self.v += self.dt * self.a
        self.p += self.dt * self.v

        # Check for a bounce - not the change in x velocity is non-physical.
        if self.p[2,0] < self.radius:
            self.p[2,0] = self.radius + (self.radius - self.p[2,0])
            self.v[2,0] *= -1.0
            self.v[0,0] *= -1.0   # Change x just for the fun of it!

        # Update the ID number to create a new ball and leave the
        # previous balls where they are.
        #####################
        # self.marker.id += 1
        #####################

        # Update the message and publish.
        now = self.start + Duration(seconds=self.t)
        self.marker.header.stamp  = now.to_msg()
        self.marker.pose.position = Point_from_p(self.p)
        self.pub2.publish(self.arraymsg)


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
        self.q[joints.index('torsoPitch')] = 0.343
        self.q[joints.index('leftShoulderPitch')], self.q[joints.index('rightShoulderPitch')] = -0.438, -0.438
        self.q[joints.index('leftShoulderRoll')], self.q[joints.index('rightShoulderRoll')] = -1.5, 1.5
        self.q[joints.index('rightHipYaw')], self.q[joints.index('leftHipYaw')] = -0.2, 0.2
        self.q[joints.index('leftKneePitch')], self.q[joints.index('rightKneePitch')] = 1.664, 1.664
        self.q[joints.index('leftAnklePitch')], self.q[joints.index('rightAnklePitch')] = -0.913, -0.913
        self.q[joints.index('leftHipPitch')], self.q[joints.index('rightHipPitch')] = -0.739, -0.739
        self.q[joints.index('leftForearmYaw')], self.q[joints.index('rightForearmYaw')] = 1.132, 1.132
        self.q[joints.index('leftElbowPitch')], self.q[joints.index('rightElbowPitch')] = -1.579, 1.579
        self.q[joints.index('leftShoulderYaw')], self.q[joints.index('rightShoulderYaw')] = 0.4, 0.4

        
        # Set up initial positions for the chain tips
        self.y_offset = 1
        self.p_ll_world, self.R_ll_world = (np.array([0.12899, 0.046633 + self.y_offset, 0.00011451]).reshape((-1, 1)), R_from_quat(np.array([0.77415, 0.0037979, 0.004645, -0.63297])))
        self.p_rl_world, self.R_rl_world = (np.array([-0.12899, 0.046633 + self.y_offset, 0.00011451]).reshape((-1, 1)), R_from_quat(np.array([0.63297, 0.004645, 0.0037979, -0.77415])))
        self.p_pelvis_world, self.R_pelvis_world = (np.array([0, + self.y_offset, 0.80111]).reshape((-1, 1)), Rotz(-pi/2))

        # Weighted matrix
        weights = np.ones(42)
        weights[joints.index('torsoPitch')] = 10
        weights[joints.index('torsoRoll')] = 10
        weights[joints.index('torsoYaw')] = 10
        weights[joints.index('leftKneePitch')] = 0.5
        weights[joints.index('rightKneePitch')] = 0.5
        weights[joints.index('leftElbowPitch')] = 0.5
        weights[joints.index('rightElbowPitch')] = 0.5
        W = np.diag(weights)
        self.M = np.linalg.inv(W @ W)

        # Goal vector for secondary task: Keep joints near beginning position
        self.qgoal = np.zeros((len(joints), 1))
        self.qgoal[joints.index('torsoPitch')] = 0
        self.qgoal[joints.index('torsoRoll')] = 0
        self.qgoal[joints.index('torsoYaw')] = 0

        # Other constants
        self.lam = 20
        self.lam_s = 20
        self.shot_time = 1

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
        if t <= 3 or t >= 3 + self.shot_time:
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
        else:
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

            # Trajectory of right hand and left hand
            p_lh_world = pxyz(0.15341 - 0.05 * (t-3), -0.44027 + self.y_offset, 0.78005 + 1.2 * (t-3))
            R_lh_world = R_from_quat(np.array([0.2016, 0.17074, -0.00038883, 0.96447]))
            v_lh_world = pxyz(-0.05, 0, 1.2)
            w_lh_world = pxyz(0, 0, 0)
            
            p_rh_world = pxyz(-0.15341 - 0.05 * (t-3), -0.44027 + self.y_offset, 0.78005 + 1.3 * (t-3))
            R_rh_world = R_from_quat(np.array([0.96447, 0.00038883, -0.17074, 0.2016]))
            v_rh_world = pxyz(-0.05, 0, 1.3)
            alpha, alphadot = 1.0 * (t-3), 1.0
            R_rh_world = Roty(alpha)
            w_rh_world = ey() * alphadot
            
            # Fkin
            qlast = self.q
            (p_rh_pelvis, R_rh_pelvis, Jv_rh_pelvis, Jw_rh_pelvis) = self.chain_right_arm.fkin(self.get_some_q(qlast, 'right_arm')) 
            (p_lh_pelvis, R_lh_pelvis, Jv_lh_pelvis, Jw_lh_pelvis) = self.chain_left_arm.fkin(self.get_some_q(qlast, 'left_arm')) 
            (p_ll_pelvis, R_ll_pelvis, Jv_ll_pelvis, Jw_ll_pelvis) = self.chain_left_leg.fkin(self.get_some_q(qlast, 'left_leg')) 
            (p_rl_pelvis, R_rl_pelvis, Jv_rl_pelvis, Jw_rl_pelvis) = self.chain_right_leg.fkin(self.get_some_q(qlast, 'right_leg'))

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

            # T matrices based on positions from fkin
            T_ll_pelvis = T_from_Rp(R_ll_pelvis, p_ll_pelvis)
            T_rl_pelvis = T_from_Rp(R_rl_pelvis, p_rl_pelvis)
            T_rh_pelvis = T_from_Rp(R_rh_pelvis, p_rh_pelvis)
            T_lh_pelvis = T_from_Rp(R_lh_pelvis, p_lh_pelvis)

            # Get current positions of right and left hand w.r.t left leg
            T_rh_ll = np.linalg.inv(T_ll_pelvis) @ T_rh_pelvis
            T_lh_ll = np.linalg.inv(T_ll_pelvis) @ T_lh_pelvis
            T_rl_ll = np.linalg.inv(T_ll_pelvis) @ T_rl_pelvis

            p_rh_ll, R_rh_ll = p_from_T(T_rh_ll), R_from_T(T_rh_ll)
            p_lh_ll, R_lh_ll = p_from_T(T_lh_ll), R_from_T(T_lh_ll)
            p_rl_ll, R_rl_ll = p_from_T(T_rl_ll), R_from_T(T_rl_ll)
            
            # Get new position of pelvis with respect to world
            T_pelvis_world = T_ll_world @ np.linalg.inv(T_ll_pelvis)
            p_pelvis_world, R_pelvis_world = p_from_T(T_pelvis_world), R_from_T(T_pelvis_world)

            # Stacking Jacobians
            J_rh_ll = np.vstack((np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jv_ll_pelvis), Jv_rh_pelvis]]) + np.block([[-Jv_ll_pelvis + crossmat(p_rh_ll) @ Jw_ll_pelvis, np.zeros_like(Jv_rh_pelvis)]])),
                                np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jw_ll_pelvis), Jw_rh_pelvis]]) + np.block([[-Jw_ll_pelvis, np.zeros_like(Jw_rh_pelvis)]]))))
        
            e_rh_ll = np.vstack((ep(pd_rh_ll, p_rh_ll), eR(Rd_rh_ll, R_rh_ll)))

            J_lh_ll = np.vstack((np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jv_ll_pelvis), Jv_lh_pelvis]]) + np.block([[-Jv_ll_pelvis + crossmat(p_lh_ll) @ Jw_ll_pelvis, np.zeros_like(Jv_lh_pelvis)]])),
                                np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jw_ll_pelvis), Jw_lh_pelvis]]) + np.block([[-Jw_ll_pelvis, np.zeros_like(Jw_lh_pelvis)]]))))
            
            e_lh_ll = np.vstack((ep(pd_lh_ll, p_lh_ll), eR(Rd_lh_ll, R_lh_ll)))

            J_rl_ll = np.vstack((np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jv_ll_pelvis), Jv_rl_pelvis]]) + np.block([[-Jv_ll_pelvis + crossmat(p_rl_ll) @ Jw_ll_pelvis, np.zeros_like(Jv_rl_pelvis)]])),
                                np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jw_ll_pelvis), Jw_rl_pelvis]]) + np.block([[-Jw_ll_pelvis, np.zeros_like(Jw_rl_pelvis)]]))))

            e_rl_ll = np.vstack((ep(pd_rl_ll, p_rl_ll), eR(Rd_rl_ll, R_rl_ll)))


            v = np.zeros((18, 1))
            v[6:9] = np.transpose(self.R_ll_world) @ v_rh_world
            v[9:12] = np.transpose(self.R_ll_world) @ w_rh_world
            v[12:15] = np.transpose(self.R_ll_world) @ v_lh_world
            v[15:18] = np.transpose(self.R_ll_world) @ w_lh_world
            e = np.vstack((e_rl_ll, e_rh_ll, e_lh_ll))

            J = np.block([
                [J_rl_ll, np.zeros((6,30))],
                [J_rh_ll[:,:6], np.zeros((6,6)), J_rh_ll[:,6:9], np.zeros((6,15)), J_rh_ll[:,9:], np.zeros((6,5))],
                [J_lh_ll[:,:6], np.zeros((6,6)), J_lh_ll[:,6:9], np.zeros((6,3)), J_lh_ll[:,9:], np.zeros((6,17))],
            ])

            gamma = 0.15
            qlast_modified = np.zeros((42, 1))
            qlast_modified[12:15] = qlast[12:15]
            qdot_s = self.lam_s * (self.qgoal - qlast_modified)
            Jinv_W = np.linalg.inv(self.M @ np.transpose(J) @ J + gamma ** 2 * np.eye(42)) @ self.M @ np.transpose(J)
            qdot = Jinv_W @ (v + self.lam * e) + (np.eye(42) - Jinv_W @ J) @ qdot_s
            # qdot = Jinv_W @ (v + self.lam * e)
            q = qlast + dt * qdot
            # print(q[12:15])
            # print(q[self.jointnames().index('rightForearmYaw')])
            self.q = q
            self.qdot = qdot
            self.p_pelvis_world, self.R_pelvis_world = p_pelvis_world, R_pelvis_world

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