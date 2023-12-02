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

        # ppelvis = pxyz(0.0, 0.0, 0.0)
        # Rpelvis = Reye()
        # Tpelvis = T_from_Rp(Rpelvis, ppelvis)
        
        # trans = TransformStamped()
        # trans.header.stamp    = self.now().to_msg()
        # trans.header.frame_id = 'world'
        # trans.child_frame_id  = 'pelvis'
        # trans.transform       = Transform_from_T(Tpelvis)
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
        # self.q[joints.index('leftKneePitch')], self.q[joints.index('rightKneePitch')] = 1, 1
        # self.q[joints.index('torsoPitch')] = 0.2
        self.q[joints.index('rightShoulderRoll')] = 1.5
        self.q[joints.index('rightForearmYaw')] = 0.5
        self.q[joints.index('rightElbowPitch')] = 1.5
        self.q[joints.index('rightShoulderPitch')] = -0.35
        
        # Set up initial positions for the chain tips
        self.p_ll_world, self.R_ll_world = (np.array([-0.010126, 0.1377, -1.0834]).reshape((-1, 1)), Reye())
        self.p_rl_world, self.R_rl_world = (np.array([-0.010126, -0.1377, -1.0834]).reshape((-1, 1)), Reye())
        self.p_pelvis_world, self.R_pelvis_world = (np.array([0, 0, -0.5]).reshape((-1, 1)), Reye())

        # Weighted matrix
        weights = np.ones(42)
        weights[joints.index('torsoPitch')] = 10
        weights[joints.index('torsoYaw')] = 10
        weights[joints.index('torsoRoll')] = 10
        weights[joints.index('leftKneePitch')] = 2
        weights[joints.index('rightKneePitch')] = 2
        # weights[joints.index('leftHipYaw')] = 3
        # weights[joints.index('leftHipRoll')] = 3
        # weights[joints.index('leftHipPitch')] = 3
        # weights[joints.index('rightHipYaw')] = 3
        # weights[joints.index('rightHipRoll')] = 3
        # weights[joints.index('rightHipPitch')] = 3
        weights[joints.index('rightShoulderPitch')] = 10
        # weights[joints.index('rightShoulderYaw')] = 0.5
        # weights[joints.index('rightShoulderRoll')] = 0.5
        W = np.diag(weights)
        self.M = np.linalg.inv(W @ W)
        
        # Other constants
        self.lam = 30

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
        if t >= 5 or t <= 3:
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
            # Broadcasting pelvis
            Tpelvis = T_from_Rp(self.R_pelvis_world, self.p_pelvis_world)

            broadcast = self.node.broadcaster
            now = self.node.now()
            
            trans = TransformStamped()
            trans.header.stamp    = now.to_msg()

            trans.header.frame_id = 'world'
            trans.child_frame_id  = 'pelvis'
            trans.transform       = Transform_from_T(Tpelvis)
            broadcast.sendTransform(trans)

            # trans2 = TransformStamped()
            # trans2.header.frame_id = 'world'
            # trans2.header.stamp    = now.to_msg()
            # trans2.child_frame_id  = 'leftFoot'
            # broadcast.sendTransform(trans2)

            # Define hand trajectory
            p_rh_world = pxyz(0.3, -0.2 * (t-3), -0.1 + 0.5 * (t-3))
            v_rh_world = pxyz(0, -0.2, 0.5)
            alpha, alphadot = 0.9 * (t-3), 0.9
            R_rh_world = Rote(pxyz(0, np.sqrt(2)/2, np.sqrt(2)/2), alpha)
            wd = ez() * alphadot

            # Fkin
            qlast = self.q
            (p_rh_pelvis, R_rh_pelvis, Jv_rh_pelvis, Jw_rh_pelvis) = self.chain_right_arm.fkin(self.get_some_q(qlast, 'right_arm')) 
            (p_ll_pelvis, R_ll_pelvis, Jv_ll_pelvis, Jw_ll_pelvis) = self.chain_left_leg.fkin(self.get_some_q(qlast, 'left_leg')) 
            (p_rl_pelvis, R_rl_pelvis, Jv_rl_pelvis, Jw_rl_pelvis) = self.chain_right_leg.fkin(self.get_some_q(qlast, 'right_leg'))

            J_ll_pelvis = np.vstack((Jv_ll_pelvis, Jw_ll_pelvis))
            J_rl_pelvis = np.vstack((Jv_rl_pelvis, Jw_rl_pelvis))

            T_ll_pelvis = T_from_Rp(R_ll_pelvis, p_ll_pelvis)
            T_rl_pelvis = T_from_Rp(R_rl_pelvis, p_rl_pelvis)
            T_rh_pelvis = T_from_Rp(R_rh_pelvis, p_rh_pelvis)

            T_rh_ll = np.linalg.inv(T_ll_pelvis) @ T_rh_pelvis
            T_rh_rl = np.linalg.inv(T_rl_pelvis) @ T_rh_pelvis

            p_rh_ll, R_rh_ll = p_from_T(T_rh_ll), R_from_T(T_rh_ll)
            p_rh_rl, R_rh_rl = p_from_T(T_rh_rl), R_from_T(T_rh_rl)

            
            p_pelvis_world, R_pelvis_world = p_rh_world - p_rh_pelvis, Reye()

            # Stacking Jacobians
            J_rh_ll = np.vstack((np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jv_ll_pelvis), Jv_rh_pelvis]]) - np.block([[Jv_ll_pelvis, np.zeros_like(Jv_rh_pelvis)]])),
                                np.transpose(R_ll_pelvis) @ (np.block([[np.zeros_like(Jw_ll_pelvis), Jw_rh_pelvis]]) - np.block([[Jw_ll_pelvis, np.zeros_like(Jw_rh_pelvis)]]))))
        
            e_rh_ll = np.vstack((ep(-self.p_ll_world + p_rh_world, p_rh_ll), eR(R_rh_world, R_rh_ll)))

            J_rh_rl = np.vstack((np.transpose(R_rl_pelvis) @ (np.block([[np.zeros_like(Jv_rl_pelvis), Jv_rh_pelvis]]) - np.block([[Jv_rl_pelvis, np.zeros_like(Jv_rh_pelvis)]])),
                                np.transpose(R_rl_pelvis) @ (np.block([[np.zeros_like(Jw_rl_pelvis), Jw_rh_pelvis]]) - np.block([[Jw_rl_pelvis, np.zeros_like(Jw_rh_pelvis)]]))))
            
            e_rh_rl = np.vstack((ep(-self.p_rl_world + p_rh_world, p_rh_rl), eR(R_rh_world, R_rh_rl)))

            e_ll_pelvis = np.vstack((ep(-self.p_pelvis_world + self.p_ll_world, p_ll_pelvis), eR(Reye(), R_ll_pelvis)))
            e_rl_pelvis = np.vstack((ep(-self.p_pelvis_world + self.p_rl_world, p_rl_pelvis), eR(Reye(), R_rl_pelvis)))

            v = np.zeros((24, 1))
            v[:3] = -(p_pelvis_world - self.p_pelvis_world) / dt
            v[6:9] = -(p_pelvis_world - self.p_pelvis_world) / dt
            v[12:15] = v_rh_world
            v[15:18] = wd
            v[18:21] = v_rh_world
            v[21:24] = wd
            # v[0:3] = -(p_pelvis_world - self.p_pelvis_world) / dt
            # v[6:9] = v_rh_world
            # v[9:12] = wd
            # v[12:15] = v_rh_world
            # v[15:18] = wd

            e = np.vstack((e_ll_pelvis, e_rl_pelvis, e_rh_ll, e_rh_rl))

            J = np.block([
                [J_ll_pelvis, np.zeros((6,36))],
                [np.zeros((6,6)), J_rl_pelvis, np.zeros((6,30))],
                [J_rh_ll[:,:6], np.zeros((6, 6)), J_rh_ll[:,6:9], np.zeros((6, 3)), np.zeros((6, 12)), J_rh_ll[:,9:], np.zeros((6, 5))],
                [np.zeros((6, 6)), J_rh_rl[:,:9], np.zeros((6, 3)), np.zeros((6, 12)), J_rh_rl[:,9:], np.zeros((6, 5))]
            ])

            self.p_pelvis_world, self.R_pelvis_world = p_pelvis_world, R_pelvis_world

            gamma = 0.1

            # Calculating the weighted Jacobian
            # U, S, Vh = np.linalg.svd(J, full_matrices=True)

            # wgamma = 2.0
            # s = np.ones(42)

            # Sdiag = np.diag(S)
            # print("J -> " + str(J.shape))
            # print("U -> " + str(U.shape))
            # print("Sdiag -> " + str(Sdiag.shape))
            # print("Vh -> " + str(Vh.shape))

            # dem1 = U @ Sdiag
            # print("HITTTTT")
            # print(dem1.shape)

            # res = dem1 @ Vh
            # print("-------")
            # print(res.shape)

            # Jwin = Vh @ Sdiag @ np.transpose(U)

            #J = Jwin

            
            # JInv = MSI @ JMerged.T @ np.linalg.inv(JMerged @ MSI @ JMerged.T + self.gamma**2 * np.eye(28))

            Jinv_W = np.linalg.inv(self.M @ np.transpose(J) @ J + gamma ** 2 * np.eye(42)) @ self.M @ np.transpose(J)
            qdot = Jinv_W @ (v + self.lam * e)
            q = qlast + dt * qdot
            self.q = q
            self.qdot = qdot

            # q = np.zeros((42, 1))
            # qdot = np.zeros((42, 1))
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