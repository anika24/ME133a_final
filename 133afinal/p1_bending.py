'''p1_bending.py

   This is a demo for moving/placing an ungrounded robot

   In particular, imagine a humanoid robot.  This moves/rotates the
   pelvis frame relative to the world.

   Node:        /pirouette
   Broadcast:   'pelvis' w.r.t. 'world'     geometry_msgs/TransformStamped

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
                'leftShoulderYaw', 'leftElbowPitch', 'leftForearmYaw', 'leftWristRoll', 'leftWristPitch', 
                'leftThumbRoll', 'leftThumbPitch1', 'leftIndexFingerPitch1', 'leftMiddleFingerPitch1',
                'leftPinkyPitch1'],

    'right_arm': ['torsoYaw', 'torsoPitch', 'torsoRoll', 'rightShoulderPitch', 'rightShoulderRoll',
                'rightShoulderYaw', 'rightElbowPitch', 'rightForearmYaw', 'rightWristRoll', 'rightWristPitch', 
                'rightThumbRoll', 'rightThumbPitch1', 'rightIndexFingerPitch1', 'rightMiddleFingerPitch1',
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

        ppelvis = pxyz(0.0, 0, -0.4 * abs(np.sin(self.t/2)))
        # Rpelvis = Rotz(np.sin(self.t))
        Rpelvis = Reye()
        Tpelvis = T_from_Rp(Rpelvis, ppelvis)
        
        trans = TransformStamped()
        trans.header.stamp    = self.now().to_msg()
        trans.header.frame_id = 'world'
        trans.child_frame_id  = 'pelvis'
        trans.transform       = Transform_from_T(Tpelvis)
        self.broadcaster.sendTransform(trans)

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
        self.chain_left_leg = KinematicChain(node, 'pelvis', 'leftFoot', joint_names['left_leg'])
        self.chain_right_leg = KinematicChain(node, 'pelvis', 'rightFoot', joint_names['right_leg'])

        # Initial q0 is 0
        self.q_left_leg = np.zeros((6, 1))
        self.q_right_leg = np.zeros((6, 1))
        self.q = np.zeros((len(self.jointnames()), 1))
        self.qdot = np.zeros((len(self.jointnames()), 1))
        self.q[0], self.q[6] = 0.1, -0.1
        self.q[3], self.q[9] = 0.05, 0.05

        # Set up initial positions for the chain tips, with respect to the pelvis
        self.pos_left_leg = (np.array([-0.010126, 0.1377, -1.0834]).reshape((-1, 1)), R_from_quat(np.array([0, 0, 0, 1])))
        self.pos_right_leg = (np.array([-0.010126, -0.1377, -1.0834]).reshape((-1, 1)), R_from_quat(np.array([0, 0, 0, 1])))

        self.qgoal = np.zeros((12, 1))
        self.qgoal[0], self.qgoal[6] = 0.1, -0.1
        self.qgoal[3], self.qgoal[9] = 0.05, 0.05

        self.amp = 0.5
        self.period = 0.5
        self.lam = 20

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!
        return joint_names['left_leg'] + joint_names['right_leg'] + joint_names['neck'] + joint_names['left_arm'][3:] + joint_names['right_arm'][3:]

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        if t > 2*pi:

            # Compute the joints.
            left_leg_qlast = self.q_left_leg
            right_leg_qlast = self.q_right_leg
            qlast = self.q[:12]

            # pelvis w/ respect to world
            ppelvis = pxyz(0.0, 0.0, -0.4 * abs(np.sin(t/2)))
            Rpelvis = Reye()

            # left and right leg w/ respect to pelvis
            (pleftleg, Rleftleg, Jllpelvis_v, Jllpelvis_w) = self.chain_left_leg.fkin(left_leg_qlast)
            (prightleg, Rrightleg, Jrlpelvis_v, Jrlpelvis_w) = self.chain_left_leg.fkin(right_leg_qlast)

            Tpelvis_world = T_from_Rp(Rpelvis, ppelvis)
            Tleftleg_pelvis = T_from_Rp(Rleftleg, pleftleg)
            Trightleg_pelvis = T_from_Rp(Rrightleg, prightleg)

            # left and right leg w/ respect to world
            Tleftleg_world = Tpelvis_world @ Tleftleg_pelvis
            Trightleg_world = Tpelvis_world @ Trightleg_pelvis
            pleftleg_world, Rleftleg_world = p_from_T(Tleftleg_world), R_from_T(Tleftleg_world)
            prightleg_world, Rrightleg_world = p_from_T(Trightleg_world), R_from_T(Trightleg_world)

            ep_ll, er_ll = ep(self.pos_left_leg[0], pleftleg_world), eR(self.pos_left_leg[1], Rleftleg_world)
            ep_rl, er_rl = ep(self.pos_right_leg[0], prightleg_world), eR(self.pos_right_leg[1], Rrightleg_world)

            J_ll = np.vstack((Jllpelvis_v, Jllpelvis_w))
            J_rl = np.vstack((Jrlpelvis_v, Jrlpelvis_w))
            J = np.block([[J_ll, np.zeros_like(J_ll)],
                        [np.zeros_like(J_rl), J_rl]])
            J[:,0] = 0
            J[:,6] = 0

            v = np.zeros((12, 1))
            e = np.vstack((ep_ll, er_ll, ep_rl, er_rl))
            
            gamma = 0.05
            Jinv_W = np.linalg.inv(np.transpose(J) @ J + gamma ** 2 * np.eye(12)) @ np.transpose(J)
            qlast_modified = np.array([qlast[0,0], 0, 0, 0, 0, 0, qlast[6, 0], 0, 0, 0, 0, 0]).reshape(-1, 1)
            qdot_s = 10*(self.qgoal - qlast_modified)
            qdot = Jinv_W @ (v + self.lam * e) + (np.eye(12) - Jinv_W @ J) @ qdot_s
            q = qlast + dt * qdot
            self.q_left_leg = q[:6]
            self.q_right_leg = q[6:]
            self.q[:12] = q
            self.qdot[:12] = qdot

            return (self.q.flatten().tolist(), self.qdot.flatten().tolist())
        
        return np.zeros((42, 1)).flatten().tolist(), np.zeros((42, 1)).flatten().tolist()


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
