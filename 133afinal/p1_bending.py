# First we have to make valkyrie able to squat
# We have to define jacobians from pelvis to each leg, with legs
# being locked to the ground and the pelvis moving up and down. 

'''p1_bending.py

   This is the code for ME 133a Final Project. We are making Valkyrie squat
   as the first step

   Node:        /generator
   Publish:     /joint_states           sensor_msgs/JointState

'''

import rclpy
import numpy as np

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp, pow

# Grab the utilities
from hw5code.GeneratorNode      import GeneratorNode
from hw5code.TransformHelpers   import *
from hw5code.TrajectoryUtils    import *

# Grab the general fkin from HW5 P5.
from hw5code.KinematicChain     import KinematicChain

class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Set up the kinematic chain object.
        self.chain_world_pelvis = KinematicChain(node, 'world', 'pelvis', self.jointnames('pelvis_to_world'))
        self.chain_left_leg = KinematicChain(node, 'world', 'leftFoot', self.jointnames('left_leg'))
        self.chain_right_leg = KinematicChain(node, 'world', 'rightFoot', self.jointnames('right_leg'))

        # Initial q0 is 0 for all joints
        self.q_pelvis = np.zeros((6, 1))
        self.q_left_leg = np.zeros((6, 1))
        self.q_right_leg = np.zeros((6, 1))
        self.q = np.vstack((self.q_pelvis, self.q_left_leg, self.q_right_leg))

        # Set up initial positions for the chain tips
        self.pos_left_leg = (np.array([-0.009844, 0.13769, -1.0834]).reshape((-1, 1)), R_from_quat(np.array([0, 0, 0, 1])))
        self.pos_right_leg = (np.array([-0.0098553, 0.13778, -1.0834]).reshape((-1, 1)), R_from_quat(np.array([0, 0, 0, 1])))
        # self.left_arm = (np.array([0.010543, 0.86701, 0.31865]).reshape((-1, 1)), R_from_quat(np.array([0, 0, 0, 1])))
        # self.right_arm = (np.array([0.01012, -0.867, 0.31885]).reshape((-1, 1)), R_from_quat(np.array([0, 0, 0, 1])))
        # self.pos_pelvis = (np.array([0, 0, 0]).reshape((-1, 1)), R_from_quat(np.array([0, 0, 0, 1])))

        self.amp = 1
        self.period = 2*pi
        self.lam = 20

    # Declare the joint names.
    def jointnames(self, chain = 'all'):
        # Return a list of joint names FOR THE EXPECTED URDF!

        # There are different chains stemming from the torso, tree structure can be seen in ValkyrieD.pdf
        chains = {
            'left_leg': ['leftHipYaw', 'leftHipRoll', 'leftHipPitch', 'leftKneePitch', 'leftAnklePitch',
                        'leftAnkleRoll'],
            'right_leg': ['rightHipYaw', 'righHipRoll', 'rightHi,pPitch', 'righKneePitch', 'righAnklePitch',
                        'rightAnkleRoll'],
            'left_arm': ['torsoYaw', 'torsoPitch', 'torsoRoll', 'leftShoulderPitch', 'leftShoulderRoll',
                        'leftShoulderYaw', 'leftElbowPitch', 'leftForearmYaw', 'leftWristRoll', 'leftWristPitch'],
            'right_arm': ['torsoYaw', 'torsoPitch', 'torsoRoll', 'rightShoulderPitch', 'rightShoulderRoll',
                        'rightShoulderYaw', 'rightElbowPitch', 'rightForearmYaw', 'rightWristRoll', 'rightWristPitch'],
            'pelvis_to_world': ['mov_x', 'mov_y', 'mov_z', 'rotate_x', 'rotate_y', 'rotate_z']
        }

        if chain == 'all':
            return chains['pelvis_to_world'] + chains['left_leg'] + chains['right_leg']
        elif chain == 'pelvis_to_world':
            return chains['pelvis_to_world']
        else:
            return chains['pelvis_to_world'] + chains[chain]

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



