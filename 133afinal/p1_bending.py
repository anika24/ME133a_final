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

# Grab the utilities
from hw5code.GeneratorNode      import GeneratorNode
from demos.TransformHelpers     import *
from hw5code.TrajectoryUtils    import *

# Grab the general fkin from HW5 P5.
from hw5code.KinematicChain     import KinematicChain


jointnames = ['leftHipYaw', 'leftHipRoll', 'leftHipPitch', 
              'leftKneePitch', 'leftAnklePitch', 'leftAnkleRoll',

              'rightHipYaw', 'rightHipRoll', 'rightHipPitch', 
              'rightKneePitch', 'rightAnklePitch', 'rightAnkleRoll',

              'torsoYaw', 'torsoPitch', 'torsoRoll',
              'lowerNeckPitch', 'neckYaw', 'upperNeckPitch',

              'rightShoulderPitch', 'rightShoulderRoll', 'rightShoulderYaw', 
              'rightElbowPitch', 'rightForearmYaw', 'rightWristRoll', 'rightWristPitch',
              'rightThumbRoll', 'rightThumbPitch1', 'rightIndexFingerPitch1', 'rightMiddleFingerPitch1',
              'rightPinkyPitch1',

              'leftShoulderPitch', 'leftShoulderRoll', 'leftShoulderYaw', 
              'leftElbowPitch', 'leftForearmYaw', 'leftWristRoll', 'leftWristPitch',
              'leftThumbRoll', 'leftThumbPitch1', 'leftIndexFingerPitch1', 'leftMiddleFingerPitch1',
              'leftPinkyPitch1'
            ]

#
#   Demo Node Class
#
class DemoNode(Node):
    # Initialization.
    def __init__(self, name, rate):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Initialize the transform broadcaster
        self.broadcaster = TransformBroadcaster(self)

        # Add a publisher to send the joint commands.
        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_states subscriber...")
        while(not self.count_subscribers('/joint_states')):
            pass

        # Set up the timing so (t=0) will occur in the first update
        # cycle (dt) from now.
        self.dt    = 1.0 / float(rate)
        self.t     = -self.dt
        self.start = self.get_clock().now() + Duration(seconds=self.dt)

        # Create a timer to keep calling update().
        self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, rate))

    # Shutdown.
    def shutdown(self):
        # Destroy the node, including cleaning up the timer.
        self.destroy_node()

    # Return the current time (in ROS format).
    def now(self):
        return self.start + Duration(seconds=self.t)

    # Update - send a new joint command every time step.
    def update(self):
        # To avoid any time jitter enforce a constant time step and
        # integrate to get the current time.
        self.t += self.dt

        # Compute position/orientation of the pelvis (w.r.t. world).
        ppelvis = pxyz(0.0, 0.5, 0.5 * np.cos(self.t/2))
        Rpelvis = Rotz(np.sin(self.t))
        Tpelvis = T_from_Rp(Rpelvis, ppelvis)
        
        # Build up and send the Pelvis w.r.t. World Transform!
        trans = TransformStamped()
        trans.header.stamp    = self.now().to_msg()
        trans.header.frame_id = 'world'
        trans.child_frame_id  = 'pelvis'
        trans.transform       = Transform_from_T(Tpelvis)
        self.broadcaster.sendTransform(trans)

        # Compute the joints.
        q    = np.zeros((len(jointnames), 1))
        qdot = np.zeros((len(jointnames), 1))

        i_relbow = jointnames.index('rightElbowPitch')
        i_rshoulder_yaw = jointnames.index('rightShoulderYaw')
        i_rforearm_yaw = jointnames.index('rightForearmYaw')

        q[i_relbow,0]           = - pi/2 + pi/8 * sin(2*self.t)
        q[i_rshoulder_yaw,0]    = pi/2
        q[i_rforearm_yaw,0]     = -pi/2
        qdot[i_relbow, 0]       = pi/4 * cos(2*self.t)

        # Build up a command message and publish.
        cmdmsg = JointState()
        cmdmsg.header.stamp = self.now().to_msg()       # Current time for ROS
        cmdmsg.name         = jointnames                # List of names
        cmdmsg.position     = q.flatten().tolist()      # List of positions
        cmdmsg.velocity     = qdot.flatten().tolist()   # List of velocities
        self.pub.publish(cmdmsg)

#
#  Main Code
#
def main(args=None):
    # Initialize ROS and the demo node (100Hz).
    rclpy.init(args=args)
    node = DemoNode('p1_bending', 100)

    # Spin, until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Set up the kinematic chain object.
        self.chain_left_leg = KinematicChain(node, 'pelvis', 'leftFoot', self.jointnames('left_leg'))
        self.chain_right_leg = KinematicChain(node, 'pelvis', 'rightFoot', self.jointnames('right_leg'))

        # Initial q0 is 0 for all joints
        self.q_pelvis = np.zeros((6, 1))
        self.q_left_leg = np.zeros((6, 1))
        self.q_right_leg = np.zeros((6, 1))
        self.q = np.vstack((self.q_pelvis, self.q_left_leg, self.q_right_leg))

        # Set up initial positions for the chain tips
        self.pos_left_leg = (np.array([-0.009844, 0.13769, -1.0834]).reshape((-1, 1)), R_from_quat(np.array([0, 0, 0, 1])))
        self.pos_right_leg = (np.array([-0.0098553, 0.13778, -1.0834]).reshape((-1, 1)), R_from_quat(np.array([0, 0, 0, 1])))

        self.amp = 1
        self.period = 2*pi
        self.lam = 20

    # Declare the joint names.
    def jointnames(self, chain = 'all'):
        # Return a list of joint names FOR THE EXPECTED URDF!

        # There are different chains stemming from the pelvis, tree structure can be seen in ValkyrieD.pdf
        chains = {
            'left_leg': ['leftHipYaw', 'leftHipRoll', 'leftHipPitch', 'leftKneePitch', 
                         'leftAnklePitch', 'leftAnkleRoll'],

            'right_leg': ['rightHipYaw', 'righHipRoll', 'rightHi,pPitch', 'righKneePitch', 'righAnklePitch',
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

        if chain == 'all':
            return chains['left_leg'] + chains['right_leg'] + chains['neck'] + chains['left_arm'][3:] + chains['right_arm'][3:]
        else:
            return chains[chain]

    def pelvis_movement(self, A, w, t):
        # Define path variable for motion of pelvis to be a cosine wave
        pos = np.array([0, 0, A - A * cos(w * t)]).reshape((-1,1))
        R_pos = R_from_quat(np.array([0, 0, 0, 1]))
        vel = np.array([0, 0, -A * w * sin(w * t)]).reshape((-1,1))
        return (pos, R_pos), vel

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        # desired position/velocities for pelvis only, everything else is stationary
        pd, vd = self.pelvis_movement(self.amp, self.period, t)
        Rd = R_from_quat(np.array([0, 0, 0, 1]))

        pelvis_qlast = self.q_pelvis
        left_leg_qlast = self.q_left_leg
        right_leg_qlast = self.q_right_leg

        qlast = self.q
        (p_p, R_p, Jv_p, Jw_p) = self.chain_world_pelvis.fkin(pelvis_qlast)
        (p_ll, R_ll, Jv_ll, Jw_ll) = self.chain_world_pelvis.fkin(left_leg_qlast)
        (p_rl, R_rl, Jv_rl, Jw_rl) = self.chain_world_pelvis.fkin(right_leg_qlast)
        J_p = np.stack((Jv_p, Jw_p))
        J_ll = np.stack((Jv_ll, Jw_ll))
        J_rl = np.stack((Jv_rl, Jw_rl))

        ep_p, er_p = ep(pd, p_p), eR(Rd, R_p)
        ep_ll, er_ll = ep(self.pos_left_leg[0], p_ll), eR(self.pos_left_leg[1], R_ll)
        ep_rl, er_rl = ep(self.pos_right_leg[0], p_rl), eR(self.pos_right_leg[1], R_rl)

        e = np.vstack((ep_p, er_p, ep_ll, er_ll, ep_rl, er_rl))

        J = np.block([[J_p, np.zeros_like(J_p), np.zeros_like(J_p)],
                      [np.zeros_like(J_ll), J_ll, np.zeros_like(J_ll)],
                      [np.zeros_like(J_rl), np.zeros_like(J_rl), J_rl]])

        v = np.zeros((18, 1))
        v[0:3] = vd

        qdot = np.linalg.pinv(J) @ (v + self.lam * e)
        q = qlast + dt * qdot
        self.q_pelvis = q[:6]
        self.q_left_leg = q[6:12]
        self.q_right_leg = q[12:]
        self.q = q

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
