#!/usr/bin/env /home/zion/py3.7_ws/py3.7_venv/bin/python

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np

def convert_depth_image(ros_image):
    bridge = CvBridge()
    # Use cv_bridge() to convert the ROS image to OpenCV format
    try:
    #Convert the depth image using the default passthrough encoding
        depth_image = bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
        depth_array = np.array(depth_image, dtype=np.float32)
        
        center_idx = np.array(depth_array.shape) / 2
        #rospy.loginfo(center_idx)
        rospy.loginfo(depth_array[int(center_idx[0])][int(center_idx[1])])
        
    except CvBridgeError as e:
        print(e)
    #Convert the depth image to a Numpy array


def pixel2depth():
	rospy.init_node('pixel2depth',anonymous=True)
	rospy.Subscriber("/zedm/zed_node/depth/depth_registered", Image,callback=convert_depth_image, queue_size=1)
	rospy.spin()

if __name__ == '__main__':
	pixel2depth()
