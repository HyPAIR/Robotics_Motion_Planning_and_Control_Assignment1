import os
import sys
path_ws = os.path.abspath('../../..') 
sys.path.append(path_ws + '/weijian/advanced_robotics/src/')
from solution.NEFoward import NEForward
from solution.NEBackwards import NEBackwards
def ID(q, dq, ddq, g, param_dyn, DH):
    """
    Computes the inverse dynamics of an n-DOF manipulator using the Newton-Euler algorithm.

    Parameters:
        q: numpy array, joint positions (Nx1)
        dq: numpy array, joint velocities (Nx1)
        ddq: numpy array, joint accelerations (Nx1)
        g: numpy array, gravity vector with respect to the base frame
        param_dyn: list of lists, dynamic parameters for each link
                   param_dyn[i] = [CoM, ICoM, m] where:
                       - CoM: Center of mass
                       - ICoM: Moment of inertia at the center of mass
                       - m: Mass
        DH: numpy array, Denavit-Hartenberg parameters

    Returns:
        tau: numpy array, joint torques (Nx1)
    """
    # Forward phase: propagate velocities and accelerations
    W, Wd, _, Pcdd, Rici, Rij, Rot = NEForward(q, dq, ddq, g, param_dyn, DH)

    # Backward phase: propagate forces and torques
    tau = NEBackwards(q, dq, ddq, param_dyn, W, Wd, Pcdd, Rij, Rici, Rot, DH)

    return tau
