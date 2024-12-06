# Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import sys

# ROS2 Python API libraries
import rclpy

# Teleop control modules
from . import panda_teleop_control
from . import  panda_setpoint_control

def main(args=None):
    if args is None:
        args = sys.argv
    rclpy.init(args=args)

    if "setpoint" in sys.argv[2]:
        panda_setpoint_control.execute(args)
    else:
        panda_teleop_control.execute(args)

    rclpy.shutdown()

if __name__ == '__main__':
    main()