import numpy as np

def NEBackwards(q, dq, ddq, param_dyn, W, Wd, Pcdd, Rij, Rici, Rot, DH):
    """
    Backward recursion of inverse dynamics to propagate forces.
    Parameters:
        q: Joint positions (list or numpy array)
        dq: Joint velocities (list or numpy array)
        ddq: Joint accelerations (list or numpy array)
        param_dyn: Dynamic parameters (list of dictionaries or lists)
        W: Angular velocities of each link (list of numpy arrays)
        Wd: Angular accelerations of each link (list of numpy arrays)
        Pcdd: Linear accelerations of COMs (list of numpy arrays)
        Rij: Vectors from the origin of each frame to the origin of the next frame (list of numpy arrays)
        Rici: Vectors from the origin of each frame to the center of mass (list of numpy arrays)
        Rot: Rotation matrices (list of numpy arrays)
        DH: Denavit-Hartenberg parameters (numpy array)
    Returns:
        tau: Joint torques (numpy array)
    """
    z0 = np.array([0, 0, 1])
    N = len(param_dyn)  # Number of links

    f0 = np.zeros(3)  # Store previously computed force
    u0 = np.zeros(3)  # Store previously computed moment
    f = np.zeros(3)   # Force exerted by link i-1 on link i
    u = np.zeros(3)   # Moment exerted by link i-1 on link i (wrt frame i-1)

    tau = np.zeros(N)  # Joint torques
    first = True

    # Backward phase
    for i in range(N - 1, -1, -1):
        if first:
            R_next = np.eye(3)  # Identity matrix for the last link
            first = False
        else:
            # TODO: Compute rotation matrix R_next for the next link
            theta_next = q[i + 1]
            alpha_next = DH[i + 1][1]
            R_next = np.array([
                [np.cos(theta_next), -np.sin(theta_next) * np.cos(alpha_next),  np.sin(theta_next) * np.sin(alpha_next)],
                [np.sin(theta_next),  np.cos(theta_next) * np.cos(alpha_next), -np.cos(theta_next) * np.sin(alpha_next)],
                [0,                   np.sin(alpha_next),                     np.cos(alpha_next)]
            ])

        # TODO: Compute the force exerted by link i-1 on link i
        f = R_next @ f0 + param_dyn[i][2] * Pcdd[i]  # param_dyn[i][2] is the mass

        # Compute the inertia tensor (assuming diagonal inertia matrix for simplicity)
        I_bar = np.diag(param_dyn[i][1][:3])  # param_dyn[i][1] contains inertia parameters

        # TODO: Compute the forces and moments at the joint
        u = (
            np.cross(-f, Rij[i] + Rici[i])
            + R_next @ u0
            + np.cross(R_next @ f0, Rici[i])
            + I_bar @ Wd[i]
            + 2 * np.cross(W[i], I_bar @ W[i])
        )

        # Compute joint torque
        tau[i] = u.T @ (Rot[i].T @ z0)

        # Update f0 and u0 for the next iteration
        f0 = f
        u0 = u

    return tau
