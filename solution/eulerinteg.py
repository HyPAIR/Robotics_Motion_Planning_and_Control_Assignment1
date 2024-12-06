import numpy as np

def euler_integ(q, dq, ddq, dt):
    """
    Euler integration method to compute the velocity and position of joints
    given the joint accelerations.

    Parameters:
        q: numpy array or list, current joint positions
        dq: numpy array or list, current joint velocities
        ddq: numpy array or list, current joint accelerations
        dt: float, time step

    Returns:
        q: numpy array, updated joint positions
        dq: numpy array, updated joint velocities
    """
    dq = np.array(dq) + np.array(ddq) * dt
    q = np.array(q) + dq * dt + 0.5 * np.array(ddq) * (dt ** 2)
    
    return q, dq