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

        # ppelvis = pxyz(0.0, 0, -0.3 * abs(np.sin(self.t/2)))
        # Rpelvis = Reye()
        # Tpelvis = T_from_Rp(Rpelvis, ppelvis)
        
        # trans = TransformStamped()
        # trans.header.stamp    = self.now().to_msg()
        # trans.header.frame_id = 'pelvis'
        # trans.child_frame_id  = 'pelvis'
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
        self.chain_right_arm = KinematicChain(node, 'pelvis', 'rightPalm', joint_names['right_arm'])

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

        self.p_ll_pelvis, self.R_ll_pelvis = (np.array([0.1361, 0.115, -0.84695]).reshape((-1, 1)), R_from_quat(np.array([0.99563, 0.0033751, 0.039847, -0.084332])))
        self.p0_rh_ll = np.array([0.30637078, -0.33878031, 0.78993693]).reshape(-1, 1)

        self.lam = 1

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
        if t <= 3 or t >= 6:
            return (self.q.flatten().tolist(), self.qdot.flatten().tolist())
        
        else:
            qlast = self.q
            # dq = np.zeros((42, 1))
            # dq[42] += 1e-3
            # qlast += dq
            (p_rh_pelvis, R_rh_pelvis, Jv_rh_pelvis, Jw_rh_pelvis) = self.chain_right_arm.fkin(self.get_some_q(qlast, 'right_arm')) 
            (p_ll_pelvis, R_ll_pelvis, Jv_ll_pelvis, Jw_ll_pelvis) = self.chain_left_leg.fkin(self.get_some_q(qlast, 'left_leg')) 
            
            pd_rh_pelvis = pxyz(0.44239, -0.28093 - 0.1 * (t-3), -0.084022 + 0.4 * (t-3))
            v_rh_pelvis = pxyz(0, -0.1, 0.4)
            Rd_rh_pelvis = R_from_quat(np.array([0.66249, 0.11349, -0.18655, 0.71654]))

            pd_ll_pelvis = pxyz(0.1361, 0.115, -0.84695)
            v_ll_pelvis = pxyz(0, 0, 0)
            Rd_ll_pelvis = R_from_quat(np.array([0.99563, 0.0033751, 0.039847, -0.084332]))

            # Jw_ll_pelvis *= 0
            # Jv_ll_pelvis *= 0

            # Right hand to pelvis task
            J_rh_pelvis = np.vstack((Jv_rh_pelvis, Jw_rh_pelvis))
            # v = np.zeros((6, 1))
            # v[0:3] = v_rh_pelvis
            # e_rh_pelvis = np.vstack((ep(pd_rh_pelvis, p_rh_pelvis), eR(Rd_rh_pelvis, R_rh_pelvis)))
            # e = e_rh_pelvis

            # left leg to pelvis task
            J_ll_pelvis = np.vstack((Jv_ll_pelvis, Jw_ll_pelvis))
            # v = np.zeros((6, 1))
            # v[0:3] = v_ll_pelvis
            # e_ll_pelvis = np.vstack((ep(pd_ll_pelvis, p_ll_pelvis), eR(Rd_ll_pelvis, R_ll_pelvis)))
            # e = e_ll_pelvis

            T_ll_pelvis = T_from_Rp(R_ll_pelvis, p_ll_pelvis)
            T_rh_pelvis = T_from_Rp(R_rh_pelvis, p_rh_pelvis)
            T_rh_ll = np.linalg.inv(T_ll_pelvis) @ T_rh_pelvis
            p_rh_ll, R_rh_ll = p_from_T(T_rh_ll), R_from_T(T_rh_ll)
            # print((p_rh_ll - self.p0_rh_ll)/1e-3)

            Td_ll_pelvis = T_from_Rp(Rd_ll_pelvis, pd_ll_pelvis)
            Td_rh_pelvis = T_from_Rp(Rd_rh_pelvis, pd_rh_pelvis)
            Td_rh_ll = np.linalg.inv(Td_ll_pelvis) @ Td_rh_pelvis
            pd_rh_ll, Rd_rh_ll = p_from_T(Td_rh_ll), R_from_T(Td_rh_ll)
            # print('position desired')
            # print(p_rh_ll)
            # print('rotation desired')
            # print(Rd_rh_ll)

            Jv_combined = np.hstack((-Jv_ll_pelvis, Jv_rh_pelvis)) + np.hstack(((crossmat(p_rh_ll) @ Jw_ll_pelvis), np.zeros_like(Jv_rh_pelvis)))
            Jw_combined = np.hstack((-Jw_ll_pelvis, Jw_rh_pelvis))
            J_combined = np.vstack((Jv_combined, Jw_combined))
            J_combined[0:3] = np.transpose(R_ll_pelvis) @ J_combined[0:3]
            J_combined[3:6] = np.transpose(R_ll_pelvis) @ J_combined[3:6]
            v = np.zeros((6, 1))
            v[0:3] = np.linalg.inv(Rd_ll_pelvis) @ v_rh_pelvis
            # v[0:3] = v_rh_pelvis

            e = np.vstack((ep(pd_rh_ll, p_rh_ll), eR(Rd_rh_ll, R_rh_ll)))

            J = np.block([
                [J_combined[:,:6], np.zeros((6,6)), J_combined[:,6:9], np.zeros((6,15)), J_combined[:,9:], np.zeros((6,5))],
                #  [J_ll_pelvis, np.zeros((6, 36))],
                #  [np.zeros((6,12)), J_rh_pelvis[:,0:3], np.zeros((6,15)), J_rh_pelvis[:,3:], np.zeros((6,5))]
            ])

            gamma = 0
            Jinv_W = np.transpose(J) @ np.linalg.inv(J @ np.transpose(J) + gamma ** 2 * np.eye(6))
            qdot = Jinv_W @ (0 * v + self.lam * e)
            # print('e:')
            # print(e)
            # print('J x qdot - e')
            # print(J @ qdot - e)
            q = qlast + dt * qdot
            self.q = q
            self.qdot = qdot

            return (self.q.flatten().tolist(), self.qdot.flatten().tolist())

        return None
        
        # return np.zeros((42, 1)).flatten().tolist(), np.zeros((42, 1)).flatten().tolist()


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



#     [[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    2.01875427e-01 -4.94850468e-02 -8.86694462e-03  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00 -3.67993622e-01 -4.48952155e-02
#   -4.61137666e-02  8.59270179e-02  2.73650224e-18  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    4.83338733e-01 -5.79291525e-03  5.95756535e-02  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00 -2.78735014e-02  5.46105963e-01
#    2.08393753e-01  6.03117504e-02  2.18482719e-17  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    1.61845760e-02 -4.31650718e-01 -2.35651592e-01  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00 -4.56809921e-01  9.57462780e-03
#   -3.20620487e-02  2.68422448e-01  2.56556248e-17  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#   -7.99146940e-02 -6.87247543e-02  9.97439911e-01  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00 -6.87247543e-02  8.64181656e-01
#   -5.01354210e-01  2.13658225e-01 -9.52959673e-01 -5.32221220e-02
#   -8.53277797e-02  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  9.97620444e-01  6.86008224e-02  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  9.97620444e-01  6.22899599e-02
#    2.21360971e-02 -9.65547661e-01 -1.72532595e-01  9.04654462e-01
#   -4.25980811e-01  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    9.96801706e-01 -5.50973948e-03 -2.01878781e-02  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00 -5.50973948e-03  4.99309550e-01
#    8.64958930e-01  1.48552611e-01  2.49199448e-01  4.22809308e-01
#    9.00699461e-01  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00]]
# [[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    2.01875345e-01 -4.94649476e-02 -8.85841729e-03  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00 -3.67978320e-01 -4.48920130e-02
#   -4.61118512e-02  8.59180341e-02 -4.50443335e-18  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    4.83338748e-01 -5.79028263e-03  5.95692313e-02  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00 -2.78784241e-02  5.46105674e-01
#    2.08394427e-01  6.03139804e-02 -9.96585852e-18  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    1.61845694e-02 -4.31652656e-01 -2.35644543e-01  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00 -4.56821208e-01  9.56847212e-03
#   -3.20658041e-02  2.68424822e-01 -2.43682195e-17  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#   -7.99146940e-02 -6.87098889e-02  9.97441522e-01  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00 -6.87103967e-02  8.64166604e-01
#   -5.01380871e-01  2.13648560e-01 -9.52969087e-01 -5.32290439e-02
#   -8.53277533e-02  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  9.97621475e-01  6.85861888e-02  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  9.97621297e-01  6.22888572e-02
#    2.21478964e-02 -9.65547207e-01 -1.72525673e-01  9.04654387e-01
#   -4.25980794e-01  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    9.96801706e-01 -5.50854770e-03 -2.01580169e-02  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00 -5.53427317e-03  4.99335738e-01
#    8.64943173e-01  1.48569461e-01  2.49168240e-01  4.22808596e-01
#    9.00699471e-01  0.00000000e+00  0.00000000e+00  0.00000000e+00
#    0.00000000e+00  0.00000000e+00]]
