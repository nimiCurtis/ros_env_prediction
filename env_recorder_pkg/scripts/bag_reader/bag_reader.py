############## 
## TO DO ##
# read with no extract
##############


# reading bag files dependencies
from bagpy import bagreader
import numpy as np
import pandas as pd
import json

# extracting images dependendcies
import os
import argparse
import cv2
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError



class READER():
    
    def __init__(self, bag_file):
        
        self.bag_file = bag_file
        self.bag_read = bagreader(bag_file)
        
        self.metadata_file = os.path.join(self.bag_read.datafolder, "metadata.json")    
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file,"r") as json_file:
                self.MetaData = json.load(json_file)
        else:
            self.MetaData = {"rgb":"","depth":"","tf":"", "imu":"","confidence":""}

    def read(self):
        
        self.topic_df = self.bag_read.topic_table
        
        for index, topic_row in self.topic_df.iterrows():
            if (topic_row['Types']=='sensor_msgs/Imu') and  topic_row['Message Count']!=0:
                if os.path.exists(self.metadata_file):
                    self.imu_df = pd.read_csv(self.MetaData["imu"])
                else:
                    imu_file = self.bag_read.message_by_topic(topic_row['Topics'])
                    self.imu_df = pd.read_csv(imu_file)
                    self.MetaData["imu"] = imu_file

            if (topic_row['Types']=='sensor_msgs/Image') and  topic_row['Message Count']!=0:
                self.init_image_df(topic_row['Topics'])
        
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, "w") as json_file:
                json.dump(self.MetaData, json_file, indent=3)
        else:
            pass

    def init_image_df(self,topic):
        
        topic_split = topic.split('/')
        
        if 'rgb' in topic_split: # change it to rgb , and add depth init
            if os.path.exists(self.metadata_file):
                self.rgb_df = pd.read_csv(self.MetaData["rgb"])
            else:

                dir = os.path.join(self.bag_read.datafolder,'rgb')
                if not os.path.exists(dir):
                    os.mkdir(dir)
                
                rgb_tmp_file = self.bag_read.message_by_topic(topic)
                self.rgb_df = pd.read_csv(rgb_tmp_file)
                self.rgb_df.drop('data',inplace = True , axis =1)
                
                self.rgb_df['frame_path'], _ = self.extract_images(topic, dir, "rgb")
                self.rgb_df.to_csv(rgb_tmp_file)
                self.MetaData["rgb"] = rgb_tmp_file
        
        if 'depth' in topic_split:
            if os.path.exists(self.metadata_file):
                self.depth_df = pd.read_csv(self.MetaData["depth"])
            else:

                dir = os.path.join(self.bag_read.datafolder,'depth')
                dir_vals = os.path.join(dir,'vals')
                if not os.path.exists(dir):
                    os.mkdir(dir)
                    if not os.path.exists(dir_vals):
                        os.mkdir(dir_vals) 
                
                depth_tmp_file = self.bag_read.message_by_topic(topic)
                self.depth_df = pd.read_csv(depth_tmp_file)
                self.depth_df.drop('data',inplace = True , axis =1)
                
                self.depth_df['frame_path'], self.depth_df['np_path'] = self.extract_images(topic, dir, "depth")
                self.depth_df.to_csv(depth_tmp_file)
                self.MetaData["depth"] = depth_tmp_file
            
        if 'confidence' in topic_split:
            if os.path.exists(self.metadata_file):
                self.confidence_df = pd.read_csv(self.MetaData["confidence"])
            else:
                dir = os.path.join(self.bag_read.datafolder,'confidence')
                dir_vals = os.path.join(dir,'vals')
                if not os.path.exists(dir):
                    os.mkdir(dir)
                    if not os.path.exists(dir_vals):
                        os.mkdir(dir_vals) 
                
                confidence_tmp_file = self.bag_read.message_by_topic(topic)
                self.confidence_df = pd.read_csv(confidence_tmp_file)
                self.confidence_df.drop('data',inplace = True , axis =1)
            
                self.confidence_df['frame_path'], self.confidence_df['np_path'] = self.extract_images(topic, dir, "confidence")
                self.confidence_df.to_csv(confidence_tmp_file)
                self.MetaData["confidence"] = confidence_tmp_file


    def extract_images(self, topic, dir, img_type):

            bag = rosbag.Bag(self.bag_file, "r")
            bridge = CvBridge()
            count = 0
            frame_path_list = []
            numpy_path_list = []
            for topic, msg, t in bag.read_messages(topics=topic):
                try:
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                except CvBridgeError as e:
                    print(e)
                
                frame_path = os.path.join(dir, "frame%06i.jpg" % count)
                if img_type == "depth":
                    depth_array = np.array(cv_img, dtype=np.float32)
                    numpy_path = os.path.join(dir, "vals/np_values%06i.npy" % count)
                    np.save(numpy_path, depth_array)
                    numpy_path_list.append(numpy_path)
                
                if img_type == "confidence":
                    confidence_array = np.array(cv_img, dtype=np.float32)
                    numpy_path = os.path.join(dir, "vals/np_values%06i.npy" % count)
                    np.save(numpy_path, confidence_array)
                    numpy_path_list.append(numpy_path)

                frame_path_list.append(frame_path)
                cv2.imwrite(frame_path, cv_img)

                count += 1

            bag.close()
            print("images saved")
            return frame_path_list, numpy_path_list
        
    def get_synced_df(self):
        # make a df depend on timestamp
        
        pass 
    
    

if __name__ == '__main__':
    read = READER('/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-10-24-00-07-50.bag')
    read.read()
    a=1