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

    # Declare the joint names.
    def jointnames(self, chain = 'all'):
        # Return a list of joint names FOR THE EXPECTED URDF!

        # There are different chains stemming from the torso, tree structure can be seen in ValkyrieD.pdf
        # We also have a 
        chains = {
            'left_leg': ['leftHipYaw', 'leftHipRoll', 'leftHipPitch', 'leftKneePitch', 'leftAnklePitch',
                        'leftAnkleRoll'],
            'right_leg': ['rightHipYaw', 'righHipRoll', 'rightHi,pPitch', 'righKneePitch', 'righAnklePitch',
                        'rightAnkleRoll'],
            'left_arm': ['torsoYaw', 'torsoPitch', 'torsoRoll', 'leftShoulderPitch', 'leftShoulderRoll',
                        'leftShoulderYaw', 'leftElbowPitch', 'leftForearmYaw', 'leftWristRoll', 'leftWristPitch'],
            'neck': ['torsoYaw', 'torsoPitch', 'torsoRoll', 'lowerNeckPitch', 'neckYaw', 'upperNeckPitch'],
            'right_arm': ['torsoYaw', 'torsoPitch', 'torsoRoll', 'rightShoulderPitch', 'rightShoulderRoll',
                        'rightShoulderYaw', 'rightElbowPitch', 'rightForearmYaw', 'rightWristRoll', 'rightWristPitch'],
            'pelvis_to_world': ['moveX', 'moveY', 'moveZ', 'rotX', 'rotY', 'rotZ'],
            'torso_to_world': ['moveX', 'moveY', 'moveZ', 'rotX', 'rotY', 'rotZ', 'torsoYaw', 'torsoPitch', 'torsoRoll']
        }

        if chain == 'all':
            return chains['pelvis_to_world'] + chains['right_arm'] + chains['left_arm'][3:] + chains['neck'][3:] + chains['left_leg'] + chains['right_leg'] + chains['right_leg']
        elif chain == 'pelvis_to_world':
            return chains['pelvis_to_world']
        elif chain == 'torso_to_world':
            return chains['torso_to_world']
        else:
            return chains['pelvis_to_world'] + chains[chain]
