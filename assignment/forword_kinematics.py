import os
import sys
import numpy as np
from math import pi
from scipy.spatial.transform import Rotation as R
import copy
from typing import List

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
sys.path.append('/home/irlab/colcon_ws/src')
from solution.solveFK import FK

# Configure numpy output
np.set_printoptions(precision=4, suppress=True)

# --- Use your code to implement FK class in calculateFK.py ---
fk = FK()

class ForwardKinematics(Node):
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

### UNCOMMENT FOLLOWING LINES TO DEBUG
# def main(args=None):
#     if args is None:
#         args = sys.argv

#     execute(args=args)
def main():
    rclpy.init() 
    node = ForwardKinematics()  
    # Define a set of configurations below matches the dimensions given in the assignment.
    # Note: each configuration is 1 * 7
    configurations = [
        np.array([ 0,     0,     0,     -pi/2, 0,     pi/2, pi/4, 0, 0]),
        np.array([ pi/2,  0,     pi/4,  -pi/2, -pi/2, pi/2,    0, 0, 0 ]),
        np.array([ 0,     0,     -pi/2, -pi/4, pi/2,  pi,   pi/4, 0, 0 ]),
        np.array([ -pi/2, 0,     -pi/4, -pi/2, 0, pi, 0, 0, 0]),
        np.array([ 0, 0,  -pi/2, -pi/4,  pi/2, pi/2,  pi/4, 0, 0])
    ]
    for i, q in enumerate(configurations):
        if i < len(configurations):
            input("Press Enter to move to next configuration...")
        else:
            input("All configurations are complete!")
        node.move_joint_directly(q) 
        joints, T0e = fk.forward(q)
        node.print_ee_pose(T0e)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()
