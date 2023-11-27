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

from demos.TransformHelpers     import *


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
              'leftPinkyPitch1',
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
        ppelvis = pxyz(0.0, 0.5, 0.5 * np.sin(self.t/2))
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

        q[i_relbow,0]     = - pi/2 + pi/8 * sin(2*self.t)
        q[i_rshoulder_yaw,0]     = pi/2
        q[i_rforearm_yaw,0]     = -pi/2
        qdot[i_relbow, 0] =          pi/4 * cos(2*self.t)

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
