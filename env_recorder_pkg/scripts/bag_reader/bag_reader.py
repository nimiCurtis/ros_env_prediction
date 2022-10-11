# reading bag files
from bagpy import bagreader
import numpy as np
import pandas as pd

# extracting images
import os
import argparse
import cv2
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class READER():
    
    def __init__(self, bag_file):
        self.bag_file = bag_file
        self.bag_read = bagreader(bag_file)
        
        self.topic_df = self.bag_read.topic_table
        self.imu_df = imu = self.bag_read.message_by_topic('/zedm/zed_node/imu/data')

        
        # creating camera df , maybe change source code. adding links to jpg files. 
        #self.camera_df = 
    def extract_images(self):
        # loop topic df topics.if type == sensor/Image --> extract to the relevant folder

        """

            bag = rosbag.Bag(self.bag_file, "r")
            bridge = CvBridge()
            count = 0
            for topic, msg, t in bag.read_messages(topics=[<image_topic>]):
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

                cv2.imwrite(os.path.join(<output_dir>, "frame%06i.jpg" % count), cv_img)
                print ("Wrote image %i" % count)

                count += 1

            bag.close()
        """        
        
        pass
        
    def get_synced_df(self):
        # make a df depend on timestamp
        
        pass 
    
    

    