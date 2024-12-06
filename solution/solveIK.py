#!/usr/bin/env python
import os
import sys
path_ws = os.path.abspath('../../..') 
sys.path.append(path_ws)
sys.path.append(path_ws + '/advance_robotics_assignment/')
sys.path.append(path_ws + '/advance_robotics_assignment/franka_ros_interface')
import numpy as np
from math import pi, acos, asin
from scipy.linalg import null_space
from solution.solveFK import FK

class IK:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])
    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint

    fk = FK()

    def __init__(self):
        """
        Set ik solver parameters:
        linear_tol - the maximum distance in meters between the target end
        effector origin and actual end effector origin for a solution to be
        considered successful
        angular_tol - the maximum angle of rotation in radians between the target
        end effector frame and actual end effector frame for a solution to be
        considered successful
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # solver parameters
        self.linear_tol = 1e-4
        self.angular_tol = 1e-3
        self.max_steps = 500
        self.min_step_size = 1e-5

    @staticmethod
    def calcJacobian(q):
        """
        Calculate the Jacobian of the end effector in a given configuration.
        INPUT:
        q - 1 x 7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]
        OUTPUT:
        J - the Jacobian matrix 
        """

        J = []       
        # YOUR CODE STARTS HERE
        fk = FK()
        joint_positions, T0e = fk.forward(q)

        o_n = np.matmul(T0e, np.array([0, 0, 0, 1]))[:3]
        o_i = (o_n.T - joint_positions)

        z_i = []
        T0e = np.identity(4)
        for i in range(7):
            # apply rotation matrix to z hat
            z = np.matmul(T0e[:3, :3], np.array([0, 0, 1]))
            z_i.append(z / np.linalg.norm(z))

            # forward kinematics to get rotation matrix
            a, alpha, d = fk.dh_params[i]
            T0e = np.matmul(T0e, fk.build_dh_transform(a, alpha, d, q[i]))

        J_v = np.array([np.cross(z_i[i], o_i[i]) for i in range(7)])
        z_i = np.array(z_i)
        J = np.append(J_v, z_i, axis=1).T
        # YOUR CODE ENDS HERE
        return J

    @staticmethod
    def cal_target_transform_vec(target, current):
        """
        Calculate the displacement vector and axis of rotation from 
        the current frame to the target frame

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
                 end effector to world

        current - 4x4 numpy array representing the current transformation from
                  end effector to world

        OUTPUTS:
        translate_vec - a 3-element numpy array containing the target translation vector from
                        the current frame to the target frame, expressed in the world frame

        rotate_vec - a 3-element numpy array containing the target rotation vector from
                     the current frame to the end effector frame
        """

        translate_vec = []
        rotate_vec = []
        # YOUR CODE STARTS HERE
        # compute displacement vector
        current_pos = np.matmul(current, np.array([0, 0, 0, 1]))
        target_pos  = np.matmul(target, np.array([0, 0, 0, 1]))
        translate_vec = (target_pos - current_pos)[:3]

        # get rotation matrices
        R_c_w = current[:3, :3]
        R_t_w = target[:3, :3]
        R_w_t = R_t_w.T

        # relative rotation from current to target
        R_c_t = np.matmul(R_w_t, R_c_w)

        # get skew symmetic matrix
        S = 0.5 * (R_c_t - R_c_t.T)

        # get vector a from S
        a = np.array([
            -S[2, 1],
            -S[0, 2],
            -S[1, 0]
        ])

        # transfrom axis to world coordinates
        rotate_vec = np.matmul(R_t_w, a)
        ## YOUR CODE ENDS HERE

        return translate_vec, rotate_vec

    def check_joint_constraints(self,q,target):
        """
        Check if the given candidate solution respects the joint limits.

        INPUTS
        q - the given solution (joint angles)

        target - 4x4 numpy array representing the desired transformation from
                 end effector to world

        OUTPUTS:
        success - True if some predefined certain conditions are met. Otherwise False
        """

        success = False

        # YOUR CODE STARTS HERE

        # check joint limits
        q = np.array(q)
        if ((q < IK.lower) | (q > IK.upper)).any():
            print("[IK SOLVER FAILURE] Violate joint limits!")
            return False

        # check distance and angle
        jointPositions, T0e = IK.fk.forward(q)
        # compute distance
        translate_vec, rotate_vec = IK.cal_target_transform_vec(target, T0e)
        distance = np.linalg.norm(translate_vec)

        # compute angle
        angle = asin(np.clip(np.linalg.norm(rotate_vec), -1, 1))
        if (distance > self.linear_tol) and (angle > self.angular_tol):
            print("[IK SOLVER FAILURE] Both linear and angular tolerance exceeded!")
        elif distance > self.linear_tol:
            print("[IK SOLVER FAILURE] Linear tolerance exceeded!")
        elif angle > self.angular_tol:
            print("[IK SOLVER FAILURE] Angular tolerance exceeded!")

        success = not bool((distance > self.linear_tol) or (angle > self.angular_tol))     
        # YOUR CODE ENDS HERE

        return success


    @staticmethod
    def solve_ik(q,target):
        """
        Uses the method you prefer to calculate the joint velocity 

        INPUTS:
        q - the current joint configuration, a "best guess" so far for the final solution

        target - a 4x4 numpy array containing the target end effector pose

        OUTPUTS:
        dq - a desired joint velocity
        Note: Make sure that it will smoothly decay to zero magnitude as the task is achieved.
        """

        dq = []
        # YOUR CODE STARTS HERE
        jointPositions, T0e = IK.fk.forward(q)
        translate_vec, rotate_vec = IK.cal_target_transform_vec(target, T0e)
        J = IK.calcJacobian(q)
        # target_delta_x
        v_w = np.append(translate_vec, rotate_vec, axis=0)

        # least square
        v_w_ = []
        J_ = []
        for i in range(len(v_w)):
            if not np.isnan(v_w[i]):
                v_w_.append(v_w[i])
                J_.append(J[i])

        if (len(v_w_) == 0):
            return np.zeros((7,))

        v_w_ = np.array(v_w_).reshape((len(v_w_), 1))

        # calculate the least squares error: J_ (6x7) is coefficient, v_w_(6x1) is the variables
        dq = np.linalg.lstsq(J_, v_w_, rcond=None)[0][:, 0]

        # # psudo inverse
        # dq = np.linalg.pinv(J) @ v_w
        
        # # damped least square
        # damping = 0.04
        # dq = J.T @ np.linalg.inv(J @ J.T + damping ** 2 * np.identity(6)) @ v_w
        # # transpose
        # JJte = J @ J.T @ v_w
        # alpha = np.dot(v_w, JJte) / np.dot(JJte, JJte)
        # dq = alpha * J.T @ v_w

        # max_delta = 0.032
        # delta = max_delta / max(max_delta, np.max(np.abs(dq)))
        # dq = delta * dq     
        # YOUR CODE ENDS HERE

        return dq

    @staticmethod
    # optional
    def solve_secondary_task(q,rate=5e-1):
        """
        Computes a joint velocity which will reduce the offset between each joint's angle and 
        the center of its range of motion and encourages the solver to choose solutions within 
        the allowed range of motion for the joints.

        INPUTS:
        q - the joint angles
        rate - a tunable parameter dictating how quickly to try to center the
        joints. Turning this parameter improves convergence behavior for the
        primary task, but also requires more solver iterations.

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """
        # normalize the offsets of all joints to range from -1 to 1 within the allowed range
        offset = 2 * (q - IK.center) / (IK.upper - IK.lower)
        
        # proportional term (implied quadratic cost)
        dq = rate * -offset

        return dq

    def inverse(self, target, initial_guess):
        """
        Solve the inverse kinematics of the robot arm

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world

        initial_guess - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], which
        is the "initial guess" from which to proceed with the solution process (has set up for you)

        OUTPUTS:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], giving the
        solution if success is True or the closest guess if success is False.

        success - True if IK is successfully solved. Otherwise False
        """

        q = initial_guess
        success = False

        # YOUR CODE STARTS HERE
        q_set = []
        while True:
            q_set.append(q)

            # Primary Task - Achieve End Effector Pose
            dq_ik = self.solve_ik(q, target)

            # Secondary Task - Center Joints
            dq_center = self.solve_secondary_task(q)

            # Project the soft constraint to null space of jacobian, flattens or reshapes the array representing 
            # the null space eigenvectors into a one-dimensional array
            nullspace = null_space(IK.calcJacobian(q)).ravel()
            dq_center_proj = nullspace * np.dot(dq_center, nullspace) / np.dot(nullspace, nullspace)

            # Task Prioritization
            dq = dq_ik + dq_center_proj

            # Termination Conditions
            exit_conditions = (len(q_set) == self.max_steps) or (np.linalg.norm(dq) < self.min_step_size)
            if exit_conditions:
                break
            
            # Clip joint angles to be within joint limits
            q = q + dq
            q[:-1] = np.clip(q, IK.lower, IK.upper)[:-1]

        # if angle is in joint_7's deadzone, clip it
        if (abs(q[-1]) - IK.upper[-1]) < 0.4887:
            q[-1] = np.clip(q[-1], IK.lower[-1], IK.upper[-1])

        # limit the angle
        if q[-1] < -np.radians(360):
            q[-1] = q[-1] % -np.radians(360)
        elif q[-1] > np.radians(360):
            q[-1] = q[-1] % np.radians(360)

        # limit the joints
        if q[-1] < IK.lower[-1]:
            print("before", q[-1])
            q[-1] = np.radians(360) + q[-1]
            print("after", q[-1])
        elif q[-1] > IK.upper[-1]:
            print("before", q[-1])
            q[-1] = -np.radians(360) + q[-1]
            print("after", q[-1])

        success = self.check_joint_constraints(q,target)   
        # YOUR CODE ENDS HERE

        return q_set, success

if __name__ == "__main__":
    pass