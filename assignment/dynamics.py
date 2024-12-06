import os
import sys
import numpy as np
from math import pi
from scipy.spatial.transform import Rotation as R
import copy
from typing import List
import time

# ROS2 Python API libraries
import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.client import Client
from rclpy.qos import qos_profile_system_default

# ROS2 message and service data structures
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from std_msgs.msg import Float64MultiArray

# path_ws = os.path.abspath('../../..') 
# sys.path.append(path_ws + '/weijian/advanced_robotics/src/')
sys.path.append('/home/weijian/advanced_robotics/src/')
from solution.FD import FD
from solution.solveFK import FK
from solution.eulerinteg import euler_integ

# --- Use your code to implement FK class in calculateFK.py ---
fk = FK()

# JOINT LIMITS
lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])
class Dynamics(Node):
    def __init__(self):
        super().__init__('panda_teleop_control')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('base_frame', None),
                ('end_effector_frame', None),
                ('end_effector_target_topic', None),
                ('end_effector_pose_topic', None)
            ]
        )

        # Create end effector target publisher
        self._joint_commands_publisher = self.create_publisher(Float64MultiArray, 'joint_group_position_controller/commands', 10)
        self._end_effector_target_publisher: Publisher = self.create_publisher(Odometry, 'end_effector_target_pose', qos_profile_system_default)
        self._end_effector_pose_subscriber: Subscription = self.create_subscription(Odometry, '/end_effector_pose', self.callback_end_effector_odom, 10)

        # Create a service for actuating the gripper. The service is requested via teleop
        self._actuate_gripper_client: Client = self.create_client(Empty, 'actuate_gripper')

        # The initial pose is just the end effector location in the base frame at the nominal joint angles
        self._end_effector_target_origin: Odometry = Odometry()
        self._end_effector_target_origin.pose.pose.position.x = 0.30701957005161057
        self._end_effector_target_origin.pose.pose.position.y = -5.934817164959582e-12
        self._end_effector_target_origin.pose.pose.position.z = 0.4872695582766443
        self._end_effector_target_origin.pose.pose.orientation.x = -0.00014170976139083377
        self._end_effector_target_origin.pose.pose.orientation.y = 0.7071045301233027
        self._end_effector_target_origin.pose.pose.orientation.z = 0.00014171064119222223
        self._end_effector_target_origin.pose.pose.orientation.w = 0.7071090038427887
        self._end_effector_target_origin.header.frame_id = 'panda_link0' 
        self._end_effector_target_origin.child_frame_id = 'end_effector_frame' 
        self._end_effector_target_origin.header.stamp = self.get_clock().now().to_msg()

        self._end_effector_target: Odometry = copy.deepcopy(self._end_effector_target_origin)
        self._end_effector_pose: Odometry = copy.deepcopy(self._end_effector_target)

        # publish the initial end effector target, which corresponds to the joints at their neutral position
        self._end_effector_target_publisher.publish(self._end_effector_target)

        self.MSG_TERMINAL = """
        Enter an end effector pose target:
        - The target should contain 7 values:
        - x, y, z target in CENTIMETERS
        - r, p, yaw target (x-y-z Euler angles) in DEGREES
        - change the gripper state (1: open->close or close->open, 0: keep current state)
        - Press ENTER to return the end effector to the HOME pose
        Enter a list separated by SPACES:
                """

        self.MSG_POSE = """CURRENT END EFFECTOR TARGET POSE:
        [x, y, z] = [{}, {}, {}] m
        [r, p, y] = [{}, {}, {}] ° (Euler)
        CURRENT END EFFECTOR POSE:
        [x, y, z] = [{}, {}, {}] m
        [r, p, y] = [{}, {}, {}] ° (Euler)"""

        self._translation_limits = [[0.0, 1.0], [-1.0, 1.0], [0.0, 1.0]] # xyz
        self._rotation_limits = [[-90., 90.], [-90., 90.], [-90., 90.]] # rpy
    def callback_end_effector_odom(self, odom: Odometry):
        self._end_effector_pose = odom

    def _publish(self):

        self._end_effector_target_publisher.publish(self._end_effector_target)

    def _set_pose_target(self, user_input):

        self._end_effector_target.header.stamp = self.get_clock().now().to_msg()
        for i, value in enumerate(user_input.split()[:7]):
            print(np.float_(value))
            if i < 3:
                # set the translation target
                self._end_effector_target.pose.pose.position.x = np.clip(np.float_(value) / 100., self._translation_limits[0][0], self._translation_limits[0][1])

                self._end_effector_target.pose.pose.position.y = np.clip(np.float_(value) / 100., self._translation_limits[1][0], self._translation_limits[1][1])

                self._end_effector_target.pose.pose.position.z = np.clip(np.float_(value) / 100., self._translation_limits[2][0], self._translation_limits[2][1])

            if i >= 3 and i < 6:
                # set the rotation target
                euler_target = [0., 0., 0.]
                for j in range(3):
                    euler_target[j] = np.clip(np.float_(value), self._rotation_limits[j][0], self._rotation_limits[j][1])

                self._end_effector_target.pose.pose.orientation = copy.deepcopy(rpy2quat(euler_target, input_in_degrees=True))

            if i == 6:
                if np.float_(value) > 0:
                    # Call the service to actuate the gripper
                    future = self._actuate_gripper_client.call_async(Empty.Request())
                    if future.done():
                        try:
                            response = future.result()
                        except Exception as e:
                            self.get_logger().info('SERVICE CALL TO ACTUATE GRIPPER SERVICE FAILED %r' % (e,))
                        else:
                            self.get_logger().info('GRIPPER ACTUATED SUCCESSFULLY')

    def move_joint_directly(self, joint_angles: np.ndarray):
        """
        Move joints directly based on the given angles.
        Args:
            joint_angles (np.ndarray): Target angles for all joints.
        """
        if len(joint_angles) != 9:
            self.get_logger().error("Invalid number of joint angles provided.")
            return

        # Create a message to publish joint commands
        msg = Float64MultiArray()
        msg.data = list(joint_angles)

        # Publish the joint angles to the joint controller
        self._joint_commands_publisher.publish(msg)

        # self.get_logger().info(f"Moving joints to angles: {joint_angles}")

    # Visualize the end effector
    def print_ee_pose(self, T0e):
        position = T0e[:3, 3]
        rotation_matrix = T0e[:3, :3]
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=True)  
        print("End Effector Position:")
        print(f"x: {position[0]:.3f}, y: {position[1]:.3f}, z: {position[2]:.3f}")
        print("\nEnd Effector Orientation (Euler Angles):")
        print(f"roll: {euler_angles[0]:.3f}°, pitch: {euler_angles[1]:.3f}°, yaw: {euler_angles[2]:.3f}°")

def main(DH, N, param_dyn, g, dt, T):
    """
    Newton-Euler recursive algorithm to create a simulation of a robot's motion.
    Parameters:
        DH: numpy array, Denavit-Hartenberg parameters [a, alpha, d]
        N: int, number of links
        param_dyn: list of dict, dynamic parameters for each link
                   Each dict contains:
                       - "CoM": numpy array, center of mass
                       - "Inertia": numpy array, inertia tensor
                       - "Mass": float, mass
        g: numpy array, gravity vector
        dt: float, time step
        T: float, total simulation time
    Returns:
        Q: numpy array, joint configurations over time
    """
    # Consistency check
    if len(DH) != len(param_dyn):
        raise ValueError("Inconsistent dimensions between DH parameters and dynamic parameters.")

    # Extract initial joint positions from DH parameters
    # q0 = np.array([row[2] for row in DH])
    q0 = np.array([ 0, 0, 0, -pi/2, 0, pi/2, pi/4 ])

    # Initialize variables
    Q = []  # To store joint configurations
    q = q0.copy()
    dq = np.zeros(N)  # Initial joint velocities

    # Simulation loop
    for t in np.arange(0, T, dt):
        # Forward Dynamics
        ddq = FD(q, dq, g, param_dyn, DH)
        
        # Euler integration
        q, dq = euler_integ(q, dq, ddq, dt)
        # Clip joint angles to be within joint limits
        q = q + dq
        q[:-1] = np.clip(q, lower, upper)[:-1]
        # if angle is in joint_7's deadzone, clip it
        if (abs(q[-1]) - upper[-1]) < 0.4887:
            q[-1] = np.clip(q[-1], lower[-1], upper[-1])
        # limit the joints
        if q[-1] < lower[-1]:
            q[-1] = np.radians(360) + q[-1]
        elif q[-1] > upper[-1]:
            q[-1] = -np.radians(360) + q[-1]
        # Store joint configuration for simulation purpose
        for q_ in q:
            if not np.isnan(q_):
                Q.append(q)

    return np.array(Q)

# Example test script
if __name__ == "__main__":
    rclpy.init() 
    # Number of links
    N = 7

    # DH parameters: [a, alpha, d]
    DH = [
            [0, -np.pi/2, 0.333],
            [0, np.pi/2, 0],
            [0.082, np.pi/2, 0.316],
            [-0.082, -np.pi/2, 0],
            [0, np.pi/2, 0.384],
            [0.088, np.pi/2, 0],
            [0, 0, 0.051 + 0.159]
        ]

    # Robot dynamic parameters
    # Initialize param_dyn as a list of lists
    param_dyn = [[None for _ in range(3)] for _ in range(N)]

    # Assign dynamic parameters
    # [x, y, z]
    param_dyn[0][0] = np.array([-0.025566, -2.88e-05, 0.057332])  # Center of Mass link 1
    param_dyn[1][0] = np.array([0, -0.0324958, -0.0675818])  # Center of Mass link 2
    param_dyn[2][0] = np.array([0, -0.06861, 0.0322285])  # Center of Mass link 3
    param_dyn[3][0] = np.array([0.0469893, 0.0316374, -0.031704])  # Center of Mass link 4
    param_dyn[4][0] = np.array([-0.0360446, 0.0336853, 0.031882])  # Center of Mass link 5
    param_dyn[5][0] = np.array([0, 0.0610427, -0.104176])  # Center of Mass link 6
    param_dyn[6][0] = np.array([0.5, 0.0, 0.0])  # Center of Mass link 7

    # [Ixx​,Iyy​,Izz​,Ixy​,Ixz​,Iyz​]
    param_dyn[0][1] = np.array([0.00782229414331, 0.0109027971813, 0.0102355503949, -1.56191622996e-05, -0.00126005738123, 1.08233858202e-05])  # Link 1 inertia at CoM
    param_dyn[1][1] = np.array([0.0180416958283, 0.0159136071891, 0.00620690827127, 0.0, 0.0, 0.0046758424612])  # Link 2 inertia at CoM
    param_dyn[2][1] = np.array([0.0182856182281,0.00621358421175, 0.0161514346309, 0.0, 0.0, -0.00472844221905])  # Link 3 inertia at CoM
    param_dyn[3][1] = np.array([0.00771376630908, 0.00989108008727, 0.00811723558464, -0.00248490625138, -0.00332147581033, -0.00217796151484])  # Link 4 inertia at CoM
    param_dyn[4][1] = np.array([0.00799663881132, 0.00825390705278, 0.0102515004345, 0.00347095570217, -0.00241222942995, 0.00235774044121])  # Link 5 inertia at CoM
    param_dyn[5][1] = np.array([0.030371374513, 0.0288752887402, 0.00444134056164, 6.50283587108e-07, -1.05129179916e-05, -0.00775653445787])  # Link 6 inertia at CoM
    param_dyn[6][1] = np.array([0.00303336450376, 0.00404479911567, 0.00558234286039, -0.000437276865508, 0.000629257294877, 0.000130472021025])  # Link 7 inertia at CoM
    
    # [mass]
    param_dyn[0][2] = 2.92  # Link 1 mass
    param_dyn[1][2] = 2.74  # Link 2 mass
    param_dyn[2][2] = 2.74  # Link 3 mass
    param_dyn[3][2] = 2.38  # Link 4 mass
    param_dyn[4][2] = 2.38  # Link 5 mass
    param_dyn[5][2] = 2.74  # Link 6 mass
    param_dyn[6][2] = 1.55  # Link 7 mass
    
    # Gravity vector
    g = np.array([0, 0, -9.81])

    # Simulation parameters
    dt = 0.5  # Time step
    T = 30      # Total time
    node = Dynamics()
    node.move_joint_directly(np.array([ 0, 0, 0, -pi/2, 0, pi/2, pi/4, 0, 0]))
    # Call the main function
    try:
        Q = main(DH, N, param_dyn, g, dt, T)
        for q_set in enumerate(Q):
            q = q_set[1]
            q_exe = np.append(q, [0, 0])
            node.move_joint_directly(q_exe) 
            joints, T0e = fk.forward(q)
            node.print_ee_pose(T0e)
            time.sleep(0.5)
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    except NotImplementedError as e:
        print(e)
