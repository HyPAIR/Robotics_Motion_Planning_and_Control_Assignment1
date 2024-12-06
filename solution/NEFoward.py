import numpy as np

def NEForward(q, dq, ddq, g, param_dyn, DH):
    """
    Forward phase of inverse dynamics to propagate link velocities and accelerations.
    Parameters:
        q: Joint positions (list or numpy array)
        dq: Joint velocities (list or numpy array)
        ddq: Joint accelerations (list or numpy array)
        g: Gravity vector (numpy array)
        param_dyn: Dynamic parameters (list of lists, each containing the center of mass as the first element)
        DH: Denavit-Hartenberg parameters (numpy array)
    Returns:
        W: Angular velocities of each link
        Wd: Angular accelerations of each link
        Pdd: Linear accelerations
        Pcdd: Linear accelerations of the centers of mass
        Rici: Vectors from the origin of each frame to the center of mass of each link
        Rij: Vectors from the origin of each frame to the origin of the next frame
        R: Rotation matrices around the z-axis
    """
    # Initialize outputs
    W = []    # Angular velocity of each link
    Wd = []   # Angular acceleration of each link
    Pdd = []  # Linear accelerations
    Pcdd = [] # Linear acceleration of COMs
    Rici = [] # Vectors to COM
    Rij = []  # Vectors to next frame
    R = []    # Rotation matrices

    # Initialize base values
    w0 = np.zeros(3)
    wd0 = np.zeros(3)
    pdd0 = -g  # Gravity included
    z0 = np.array([0, 0, 1])  # Z-axis direction

    N = len(param_dyn)  # Number of links

    for j in range(N):
        # TODO: Compute the vector ri_ci from the origin of frame i to the center of mass of link i
        Rici.append(-np.array(param_dyn[j][0]))

        # TODO: Compute the vector ri_j from the origin of frame i-1 to the origin of frame i
        Rij.append(2 * np.array(param_dyn[j][0]))

        # TODO: Compute the rotation matrix
        theta = q[j]
        alpha = DH[j][1]
        Rj = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha)],
            [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha)],
            [0,              np.sin(alpha),                 np.cos(alpha)]
        ])
        R.append(Rj)

        # Compute angular velocity of link i
        Wj = Rj.T @ (w0 + dq[j] * z0)
        W.append(Wj)

        # Compute angular acceleration of link i
        Wdj = Rj.T @ (wd0 + ddq[j] * z0 + np.cross(w0, dq[j] * z0))
        Wd.append(Wdj)

        # Compute linear acceleration of link i
        Pddj = Rj.T @ pdd0 + np.cross(Wdj, Rij[j]) + np.cross(Wj, np.cross(Wj, Rij[j]))
        Pdd.append(Pddj)

        # Compute linear acceleration of the center of mass of link i
        Pcddj = Pddj + np.cross(Wdj, Rici[j]) + np.cross(Wj, np.cross(Wj, Rici[j]))
        Pcdd.append(Pcddj)

        # Update base values for the next iteration
        w0 = Wj
        wd0 = Wdj
        pdd0 = Pddj

    # TODO: Verify if additional transformations are required for final rotation matrices
    R.append(np.eye(3))

    return W, Wd, Pdd, Pcdd, Rici, Rij, R