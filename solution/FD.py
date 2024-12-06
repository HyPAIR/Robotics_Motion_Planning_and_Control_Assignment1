import numpy as np
import os
import sys
path_ws = os.path.abspath('../../..') 
sys.path.append(path_ws + '/weijian/advanced_robotics/src/')
from solution.ID import ID
def FD(q, dq, g, param_dyn, DH):
    """
    Forward Dynamics computes the joint accelerations resulting from joint torques 
    and forces at the end-effector given the initial state of the system.

    Parameters:
        q: numpy array, joint positions (Nx1)
        dq: numpy array, joint velocities (Nx1)
        g: numpy array, gravity vector with respect to the base frame
        param_dyn: list of lists, dynamic parameters of each link
                   param_dyn[i] = [CoM, ICoM, m] where:
                       - CoM: Center of mass
                       - ICoM: Moment of inertia at the center of mass
                       - m: Mass
        DH: numpy array, Denavit-Hartenberg parameters

    Returns:
        ddq: numpy array, joint accelerations (Nx1)
    """
    # Initialize variables
    n = len(q)
    M = np.zeros((n, n))       # Inertia matrix
    h = np.zeros((n,))         # Joint torques
    tau_a = np.zeros((n,))     # Actuator torques (assumed negligible for now)

    # Compute the inertia matrix
    for i in range(n):
        q_0 = np.copy(q)
        dq_0 = np.zeros(n)
        ddq_0 = np.zeros(n)
        ddq_0[i] = 1  # e_i vector
        
        # TODO: Implement the ID function to compute the inertia matrix column
        # ID should return the torques for given q, dq, and ddq
        M[:, i] = ID(q_0, dq_0, ddq_0, 0 * g, param_dyn, DH)

    # TODO: Use ID function to compute coriolis and centrifugal components
    h = ID(q, dq, np.zeros(n), g, param_dyn, DH)

    # TODO: Include actuator torques or external forces if needed
    # Compute the joint accelerations resulting from the joint torques
    ddq = np.linalg.inv(M) @ (tau_a - h)

    return ddq