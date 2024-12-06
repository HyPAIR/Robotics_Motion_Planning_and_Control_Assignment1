from typing import List
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Quaternion

def quat2rpy(quat: Quaternion, degrees=True) -> List[float]:

    return list(R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler('xyz', degrees=degrees))

def rpy2quat(rpy: List[float], input_in_degrees=False) -> Quaternion:

    quat = R.from_euler('xyz', rpy, degrees=input_in_degrees).as_quat()

    out = Quaternion()
    out.x = quat[0]
    out.y = quat[1]
    out.z = quat[2]
    out.w = quat[3]

    return out # return roll-pith-yaw angles as quaternion