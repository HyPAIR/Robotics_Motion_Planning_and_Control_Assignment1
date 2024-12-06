# Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

# TODO: Have a model that spawns which does not have any collision meshses. For now, just try to get it working with basic teleop
# TODO: This class only subscribes to joint states and publishes to end_effector_target

import numpy as np
import copy

# ROS2 Python API libraries
import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.client import Client

# ROS2 message and service data structures
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

# For teleop control
from rclpy.duration import Duration
from rclpy.qos import qos_profile_system_default

import sys, select, termios, tty

# For 'q' keystroke exit
import os
import signal

# Helpers
from .helpers import quat2rpy, rpy2quat

# Configure numpy output
np.set_printoptions(precision=4, suppress=True)

def execute(args):
    # rclpy.init(args=args)

    app = PandaTeleop()
    app.poll_keys()

    app.destroy_node()
    # rclpy.shutdown()

class PandaTeleop(Node):
    def __init__(self):
        super().__init__('panda_teleop_control')

        self._last_pressed = {}
        self._settings = termios.tcgetattr(sys.stdin)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('base_frame', None),
                ('end_effector_frame', None),
                ('end_effector_target_topic', None),
                ('end_effector_pose_topic', None)
            ]
        )

        # TODO: At the moment I'm unable to load the parameter directly from the parameter server. Later I need to modify this code to include that functionality.
        # Create end effector target publisher
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
        self._end_effector_target_origin.header.frame_id = 'panda_link0' # TODO: Remove hardcoded parameters
        self._end_effector_target_origin.child_frame_id = 'end_effector_frame' # TODO: Remove hardcoded parameters
        self._end_effector_target_origin.header.stamp = self.get_clock().now().to_msg()

        self._end_effector_target: Odometry = copy.deepcopy(self._end_effector_target_origin)
        self._end_effector_pose: Odometry = copy.deepcopy(self._end_effector_target)

        # publish the initial end effector target, which corresponds to the joints at their neutral position
        self._end_effector_target_publisher.publish(self._end_effector_target)

        self.MSG_TELEOP = """
Reading from the keyboard and publishing to end effector target!
---------------------------
Linear movement:
u    i    o               +x    
j    k    l   -->   +y          -y
m    ,    .               -x    
For rotation, hold down the shift key:
---------------------------
U    I    O               +p    
J    K    L   -->   -r          +r
M    <    >               -p    
t : up (+z)
b : down (-z)
y : +yaw
n : -yaw
z : open grippers/close grippers
r : return to home position
q : stop (quit)
CTRL-C to quit
        """
        self.MSG_POSE = """CURRENT END EFFECTOR TARGET POSE:
[x, y, z] = [{}, {}, {}] m
[r, p, y] = [{}, {}, {}] ° (Euler)
CURRENT END EFFECTOR POSE:
[x, y, z] = [{}, {}, {}] m
[r, p, y] = [{}, {}, {}] ° (Euler)"""

        self._planar_translation_bindings = { # +x, -x, +y, -y
            'u' : (1, 0, 1, 0),
            'i' : (1, 0, 0, 0),
            'o' : (1, 0, 0, -1),
            'j' : (0, 0, 1, 0),
            'l' : (0, 0, 0, -1),
            'm' : (0, -1, 1, 0),
            ',' : (0, -1, 0, 0),
            '.' : (0, -1, 0, -1)
        }

        self._vertical_translation_bindings = { # +z, -z
            't' : (1, 0),
            'b' : (0, -1)
        }

        self._planar_rotation_bindings = { # +p -p +r -r
            'U' : (1, 0, 0, -1),
            'I' : (1, 0, 0, 0),
            'O' : (1, 0, 1, 0),
            'J' : (0, 0, 0, -1),
            'L' : (0, 0, 1, 0),
            'M' : (0, -1, 0, -1),
            '<' : (0, -1, 0, 0),
            '>' : (0, -1, 1, 0)
        }

        self._yaw_rotation_bindings = { # +yaw, -yaw
            'y' : (1, 0),
            'n' : (0, -1)
        }

        self._open_close_gripper_bindings = {
            'z' : 0
        }

        self._translation_limits = [[0.0, 1.0], [-1.0, 1.0], [0.0, 1.0]] # xyz
        self._rotation_limits = [[-90., 90.], [-90., 90.], [-90., 90.]] # rpy
        self._dtheta = 1.0
        self._dx = 0.01

    def callback_end_effector_odom(self, odom: Odometry):
        self._end_effector_pose = odom

    def _key_pressed(self, keycode):
        if keycode == 'q':
            self._home()
            os.kill(os.getpid(), signal.SIGINT)

        if keycode == 'r':
            # return the end effector to the home position
            self._home()

        elif keycode in self._planar_translation_bindings.keys():
            self._last_pressed[keycode] = self.get_clock().now()

        elif keycode in self._vertical_translation_bindings.keys():
            self._last_pressed[keycode] = self.get_clock().now()

        elif keycode in self._planar_rotation_bindings.keys():
            self._last_pressed[keycode] = self.get_clock().now()

        elif keycode in self._yaw_rotation_bindings.keys():
            self._last_pressed[keycode] = self.get_clock().now()

        elif keycode in self._open_close_gripper_bindings.keys():

            # Call the service to actuate the gripper
            future = self._actuate_gripper_client.call_async(Empty.Request())
            if future.done():
                try:
                    response = future.result()
                except Exception as e:
                    self.get_logger().info('SERVICE CALL TO ACTUATE GRIPPER SERVICE FAILED %r' % (e,))
                else:
                    self.get_logger().info('GRIPPER ACTUATED SUCCESSFULLY')
        else:
            return

    def _home(self):
            self._end_effector_target = copy.deepcopy(self._end_effector_target_origin)
            self._end_effector_target.header.stamp = self.get_clock().now().to_msg()
            self._publish()

    def _set_pose_target(self):
        now = self.get_clock().now()
        self._end_effector_target.header.stamp = now.to_msg()
        keys = []
        for a in self._last_pressed:
            if now - self._last_pressed[a] < Duration(seconds=0.1):
                keys.append(a)

        for k in keys:
            if k in self._planar_translation_bindings.keys():
                binding = self._planar_translation_bindings[k]
                
                for i in range(2):
                    self._end_effector_target.pose.pose.position.x = np.clip(self._end_effector_target.pose.pose.position.x + binding[i] * self._dx, self._translation_limits[0][0], self._translation_limits[0][1])

                    self._end_effector_target.pose.pose.position.y = np.clip(self._end_effector_target.pose.pose.position.y + binding[i+2] * self._dx, self._translation_limits[1][0], self._translation_limits[1][1])

            elif k in self._vertical_translation_bindings.keys():
                binding = self._vertical_translation_bindings[k]

                self._end_effector_target.pose.pose.position.z = np.clip(self._end_effector_target.pose.pose.position.z + binding[0] * self._dx, self._translation_limits[2][0], self._translation_limits[2][1])

                self._end_effector_target.pose.pose.position.z = np.clip(self._end_effector_target.pose.pose.position.z + binding[1] * self._dx, self._translation_limits[2][0], self._translation_limits[2][1])

            elif k in self._planar_rotation_bindings.keys():
                binding = self._planar_rotation_bindings[k]

                # Get the current target rpy (Euler angles) for the end effector
                euler_target = quat2rpy(self._end_effector_target.pose.pose.orientation, degrees=True)

                for i in range(2):
                    euler_target[0] = np.clip(euler_target[0] + binding[i+2] * self._dtheta, self._rotation_limits[0][0], self._rotation_limits[0][1])

                    euler_target[1] = np.clip(euler_target[1] + binding[i] * self._dtheta, self._rotation_limits[1][0], self._rotation_limits[1][1])

                # Convert rotation back to radians and back to a quaternion representation
                quat = rpy2quat(euler_target, input_in_degrees=True)
                self._end_effector_target.pose.pose.orientation = copy.deepcopy(quat)

            elif k in self._yaw_rotation_bindings.keys(): # yaw rotation
                binding = self._yaw_rotation_bindings[k]

                euler_target = quat2rpy(self._end_effector_target.pose.pose.orientation, degrees=True)

                euler_target[2] = np.clip(euler_target[2] + binding[0] * self._dtheta, self._rotation_limits[2][0], self._rotation_limits[2][1])

                euler_target[2] = np.clip(euler_target[2] + binding[1] * self._dtheta, self._rotation_limits[2][0], self._rotation_limits[2][1])

                # Convert rotation back to radians and back to a quaternion representation
                quat = rpy2quat(euler_target, input_in_degrees=True)
                self._end_effector_target.pose.pose.orientation = copy.deepcopy(quat)

            else:
                return

    def _publish(self):

        self._end_effector_target_publisher.publish(self._end_effector_target)

    def _get_key(self):
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._settings)
        return key

    def poll_keys(self):

        try:
            while(1):
                keycode = self._get_key()

                print(self.MSG_TELEOP)

                rclpy.spin_once(self) # this is necessary, otherwise the odom callback will never be called within this while loop

                euler_current = quat2rpy(self._end_effector_pose.pose.pose.orientation, degrees=True)
                euler_target = quat2rpy(self._end_effector_target.pose.pose.orientation, degrees=True)

                print(self.MSG_POSE.format(
                self._end_effector_target.pose.pose.position.x,
                self._end_effector_target.pose.pose.position.y,
                self._end_effector_target.pose.pose.position.z,
                euler_target[0], euler_target[1], euler_target[2],
                self._end_effector_pose.pose.pose.position.x,
                self._end_effector_pose.pose.pose.position.y,
                self._end_effector_pose.pose.pose.position.z,
                euler_current[0], euler_current[1], euler_current[2]))

                self._key_pressed(keycode)
                self._set_pose_target()
                self._publish()

        except Exception as e:
            print(e)
        
        finally:
            self._publish()

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._settings)

### UNCOMMENT FOLLOWING LINES TO DEBUG
# def main(args=None):
#     if args is None:
#         args = sys.argv

#     execute(args=args)

# if __name__ == '__main__':
#     main()