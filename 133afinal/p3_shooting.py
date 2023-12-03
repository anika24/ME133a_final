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

        # p_ll = pxyz(-0.010126, 0.1377, -1.0834)
        # R_ll = Reye()
        # Tll = T_from_Rp(R_ll, p_ll)
        # trans = TransformStamped()
        # trans.header.stamp    = self.now().to_msg()
        # trans.header.frame_id = 'world'
        # trans.child_frame_id  = 'leftFoot'
        # trans.transform       = Transform_from_T(Tll)
        # self.broadcaster.sendTransform(trans)

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
        self.q[joints.index('leftKneePitch')], self.q[joints.index('rightKneePitch')] = 1.664, 1.664
        self.q[joints.index('leftAnklePitch')], self.q[joints.index('rightAnklePitch')] = -0.913, -0.913
        self.q[joints.index('leftHipPitch')], self.q[joints.index('rightHipPitch')] = -0.739, -0.739
        self.q[joints.index('leftForearmYaw')], self.q[joints.index('rightForearmYaw')] = 1.132, 1.132
        self.q[joints.index('leftElbowPitch')], self.q[joints.index('rightElbowPitch')] = -1.579, 1.579
        self.q[joints.index('leftShoulderYaw')], self.q[joints.index('rightShoulderYaw')] = -1.559, -1.559
        
        # Set up initial positions for the chain tips
        self.p_ll_world, self.R_ll_world = (np.array([-0.040262, 0.1377, 0.00011451]).reshape((-1, 1)), R_from_quat(np.array([0.99998, 0, 0.006, 0])))
        self.p_rl_world, self.R_rl_world = (np.array([-0.040262, -0.1377, 0.00011451]).reshape((-1, 1)), R_from_quat(np.array([0.99998, 0, 0.006, 0])))
        self.p_pelvis_world, self.R_pelvis_world = (np.array([0, 0, 0.80111]).reshape((-1, 1)), R_from_quat(np.array([1, 0, 0, 0])))

        # Weighted matrix
        weights = np.ones(42)
        W = np.diag(weights)
        self.M = np.linalg.inv(W @ W)
        
        # Other constants
        self.lam = 20

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
        if t <= 3 or t >= 5:
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
        elif 3 < t < 5:
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

            # Define right hand trajectory
            # p_rh_world = pxyz(0.3, -0.1 * (t-3), -0.1 + 0.5 * (t-3))
            # v_rh_world = pxyz(0, -0.1, 0.5)
            # alpha, alphadot = 0.9 * (t-3), 0.9
            # R_rh_world = Rote(pxyz(0, np.sqrt(2)/2, np.sqrt(2)/2), alpha)
            # wd = ez() * alphadot

            p_rh_world = pxyz(0.013999, -0.60294, 1.4325)
            v_rh_world = pxyz(0, 0, 0)
            R_rh_world = R_from_quat(np.array([0.68821, -0.69206, -0.14921, 0.15861]))
            wd = pxyz(0, 0, 0)

            # Define left hand trajectory
            # p_lh_world = pxyz(0.4, 0.15 - 0.1 * (t-3), -0.1  + 0.4 * (t-3))
            # v_lh_world = pxyz(0, -0.1, 0.4)
            # R_lh_world = Reye()

            p_lh_world = pxyz(0.013999, 0.60294, 1.4325)
            v_lh_world = pxyz(0, 0, 0)
            R_lh_world = R_from_quat(np.array([0.68821, 0.69206, -0.14921, -0.15861]))

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

            # print(p_rl_ll, R_rl_ll)
            # print("---------")
            
            # Get new position of pelvis with respect to world
            T_pelvis_world = T_ll_world @ np.linalg.inv(T_ll_pelvis)
            p_pelvis_world, R_pelvis_world = p_from_T(T_pelvis_world), R_from_T(T_pelvis_world)

            # Stacking Jacobians
            J_rh_ll = np.vstack((np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jv_ll_pelvis), Jv_rh_pelvis]]) - np.block([[Jv_ll_pelvis, np.zeros_like(Jv_rh_pelvis)]])),
                                np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jw_ll_pelvis), Jw_rh_pelvis]]) - np.block([[Jw_ll_pelvis, np.zeros_like(Jw_rh_pelvis)]]))))
        
            e_rh_ll = np.vstack((ep(pd_rh_ll, p_rh_ll), eR(Rd_rh_ll, R_rh_ll)))

            J_lh_ll = np.vstack((np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jv_ll_pelvis), Jv_lh_pelvis]]) - np.block([[Jv_ll_pelvis, np.zeros_like(Jv_lh_pelvis)]])),
                                np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jw_ll_pelvis), Jw_lh_pelvis]]) - np.block([[Jw_ll_pelvis, np.zeros_like(Jw_lh_pelvis)]]))))
            
            e_lh_ll = np.vstack((ep(pd_lh_ll, p_lh_ll), eR(Rd_lh_ll, R_lh_ll)))

            J_rl_ll = np.vstack((np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jv_ll_pelvis), Jv_rl_pelvis]]) - np.block([[Jv_ll_pelvis, np.zeros_like(Jv_rl_pelvis)]])),
                                np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jw_ll_pelvis), Jw_rl_pelvis]]) - np.block([[Jw_ll_pelvis, np.zeros_like(Jw_rl_pelvis)]]))))

            e_rl_ll = np.vstack((ep(pd_rl_ll, p_rl_ll), eR(Rd_rl_ll, R_rl_ll)))

            J_ll_rh = np.vstack((np.transpose(R_rh_pelvis) @ (np.block([[Jv_ll_pelvis, np.zeros_like(Jv_rh_pelvis)]]) - np.block([[np.zeros_like(Jv_ll_pelvis), Jv_rh_pelvis]])),
                                np.transpose(R_rh_pelvis) @ (np.block([[Jw_ll_pelvis, np.zeros_like(Jv_rh_pelvis)]]) - np.block([[np.zeros_like(Jw_ll_pelvis), Jv_rh_pelvis]]))))
                                
            e_ll_rh = np.vstack((ep(pd_ll_rh, p_ll_rh), eR(Rd_ll_rh, R_ll_rh)))


            v = np.zeros((18, 1))
            # v[6:9] = v_rh_world
            # v[12:15] = v_lh_world

            e = np.vstack((e_rl_ll, e_rh_ll, e_lh_ll))
            
            J = np.block([
                [J_rl_ll, np.zeros((6,30))],
                [J_rh_ll[:,:6], np.zeros((6,6)), J_rh_ll[:,6:9], np.zeros((6,15)), J_rh_ll[:,9:], np.zeros((6,5))],
                [J_lh_ll[:,:6], np.zeros((6,6)), J_lh_ll[:,6:9], np.zeros((6,3)), J_lh_ll[:,9:], np.zeros((6,17))],
            ])

            gamma = 0.1
            Jinv_W = np.linalg.inv(self.M @ np.transpose(J) @ J + gamma ** 2 * np.eye(42)) @ self.M @ np.transpose(J)
            qdot = np.linalg.pinv(J) @ (v + self.lam * e)
            q = qlast + dt * qdot
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
