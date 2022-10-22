# reading bag files dependencies
from bagpy import bagreader
import numpy as np
import pandas as pd

# extracting images dependendcies
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

    def read(self):
        
        self.topic_df = self.bag_read.topic_table
        
        for index, topic_row in self.topic_df.iterrows():
            if (topic_row['Types']=='sensor_msgs/Imu') and  topic_row['Message Count']!=0:
                self.imu_df = pd.read_csv(self.bag_read.message_by_topic(topic_row['Topics']))

            if (topic_row['Types']=='sensor_msgs/Image') and  topic_row['Message Count']!=0:
                self.init_image_df(topic_row['Topics'])
                        
    def init_image_df(self,topic):
        
        topic_split = topic.split('/')
        
        if 'left' in topic_split: # change it to rgb , and add depth init
            dir = os.path.join(self.bag_read.datafolder,'rgb')
            os.mkdir(dir)
            
            rgb_tmp_file = self.bag_read.message_by_topic(topic)
            self.rgb_df = pd.read_csv(rgb_tmp_file)
            self.rgb_df.drop('data',inplace = True , axis =1)
            
            self.rgb_df['frame_path'] = self.extract_images(topic, dir)
            self.rgb_df.to_csv(rgb_tmp_file)
        
        if 'depth' in topic_split:
            dir = os.path.join(self.bag_read.datafolder,'depth')
            os.mkdir(dir)
            
            depth_tmp_file = self.bag_read.message_by_topic(topic)
            self.depth_df = pd.read_csv(rgb_tmp_file)
            self.depth_df.drop('data',inplace = True , axis =1)
            
            self.depth_df['frame_path'] = self.extract_images(topic, dir)
            self.depth_df.to_csv(rgb_tmp_file)
            


    def extract_images(self, topic, dir):

            bag = rosbag.Bag(self.bag_file, "r")
            bridge = CvBridge()
            count = 0
            path_list = []
            for topic, msg, t in bag.read_messages(topics=topic):
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") # check desired encoding
                frame_path = os.path.join(dir, "frame%06i.jpg" % count)
                
                path_list.append(frame_path)
                cv2.imwrite(frame_path, cv_img)
                
                print ("Wrote image %i" % count)

                count += 1

            bag.close()
            return path_list
        
    def get_synced_df(self):
        # make a df depend on timestamp
        
        pass 
    
    

if __name__ == '__main__':
    read = READER('/home/zion/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/bag_2022-10-11-18-18-54.bag')
    read.read()
    a=1