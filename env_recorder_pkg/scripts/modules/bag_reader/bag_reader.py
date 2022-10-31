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
        self.MetaData = {}
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file,"r") as json_file:
                self.MetaData = json.load(json_file)
        
        self.rgb_df = pd.DataFrame()
        self.depth_df = pd.DataFrame()
        self.imu_df = pd.DataFrame()
        self.confidence_df = pd.DataFrame()

        self.frame_count = 0

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


    def init_image_df(self,topic):
        
        topic_split = topic.split('/')
        

        if 'depth' in topic_split:
            self.depth_df = self.set_image_df(topic,'depth')

        if 'rgb' in topic_split: 
            self.rgb_df = self.set_image_df(topic,'rgb')
            
        if 'confidence' in topic_split:
            self.confidence_df = self.set_image_df(topic,'confidence')


    def extract_images(self, topic, dir, img_type):

            bag = rosbag.Bag(self.bag_file, "r")
            bridge = CvBridge()
            frame_path_list = []
            numpy_path_list = []
            for topic, msg, t in bag.read_messages(topics=topic):
                try:
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                except CvBridgeError as e:
                    print(e)
                
                frame_path = os.path.join(dir, "frame%06i.jpg" % self.frame_count)

                if (img_type != "rgb"):
                    numpy_path_list.append(self.save_np_data(cv_img,dir))

                frame_path_list.append(frame_path)
                cv2.imwrite(frame_path, cv_img)

                self.frame_count += 1
            bag.close()
            print("images saved")
            return frame_path_list, numpy_path_list
        
    def get_synced_df(self):
        # make a df depend on timestamp
        
        pass 
    
    def set_image_df(self,topic,image_type):
        self.frame_count = 0
        if os.path.exists(self.metadata_file):
            df = pd.read_csv(self.MetaData[image_type])
            return df
        else:
            dir = os.path.join(self.bag_read.datafolder,image_type)
            dir_vals = os.path.join(dir,'vals')
            if not os.path.exists(dir):
                os.mkdir(dir)
                if not os.path.exists(dir_vals) and image_type!='rgb':
                    os.mkdir(dir_vals) 
            
            tmp_file = self.bag_read.message_by_topic(topic)
            df = pd.read_csv(tmp_file)
            df.drop('data',inplace = True , axis =1)
            if image_type == 'rgb':
                df['frame_path'], _ = self.extract_images(topic, dir, image_type)
            else:
                df['frame_path'], df['np_path'] = self.extract_images(topic, dir, image_type)

            df.to_csv(tmp_file)
            self.MetaData[image_type] = tmp_file
            return df

        
        
    def save_np_data(self,img,dir, d_type=np.float32):
            img_array = np.array(img, dtype=d_type)
            numpy_path = os.path.join(dir, "vals/np_values%06i.npy" % self.frame_count)
            np.save(numpy_path, img_array)
            return numpy_path
            
    

if __name__ == '__main__':
    read = READER('/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-10-27-12-42-56.bag')
    read.read()
    a=1