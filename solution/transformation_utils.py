#!/usr/bin/env python

from math import pi, sin, cos
import numpy as np

# Note: Complete the following subfunctions to generate valid transformation matrices 
# from a translation vector and Euler angles, or a sequence of 
# successive rotations around z, y, and x.
class transformation():

    @staticmethod
    def trans(d):
        """
        Calculate pure translation homogenous transformation by d
        """

        # YOUR CODE STARTS HERE
        return np.array([
            [ 1, 0, 0, d[0] ],
            [ 0, 1, 0, d[1] ],
            [ 0, 0, 1, d[2] ],
            [ 0, 0, 0, 1    ],
        ])    
        # YOUR CODE ENDS HERE
    
    @staticmethod
    def roll(a):
        """
        Calculate homogenous transformation for rotation around x axis by angle a
        """

        # YOUR CODE STARTS HERE
        return np.array([
            [ 1,     0,       0,  0 ],
            [ 0, cos(a), -sin(a), 0 ],
            [ 0, sin(a),  cos(a), 0 ],
            [ 0,      0,       0, 1 ],
        ])    
        # YOUR CODE ENDS HERE

    @staticmethod
    def pitch(a):
        """
        Calculate homogenous transformation for rotation around y axis by angle a
        """

        # YOUR CODE STARTS HERE
        return np.array([
            [ cos(a), 0, -sin(a), 0 ],
            [      0, 1,       0, 0 ],
            [ sin(a), 0,  cos(a), 0 ],
            [ 0,      0,       0, 1 ],
        ])    
        # YOUR CODE ENDS HERE

    @staticmethod
    def yaw(a):
        """
        Calculate homogenous transformation for rotation around z axis by angle a
        """

        # YOUR CODE STARTS HERE
        return np.array([
            [ cos(a), -sin(a), 0, 0 ],
            [ sin(a),  cos(a), 0, 0 ],
            [      0,       0, 1, 0 ],
            [      0,       0, 0, 1 ],
        ])    
        # YOUR CODE ENDS HERE

    @staticmethod
    def transform(d,rpy):
        """
        Calculate a homogenous transformation for translation by d and
        rotation corresponding to roll-pitch-yaw euler angles
        """

        # YOUR CODE STARTS HERE
        return transformation.trans(d) @ transformation.roll(rpy[0]) @ transformation.pitch(rpy[1]) @ transformation.yaw(rpy[2])    
        # YOUR CODE ENDS HERE
    
if __name__ == "__main__":
    pass