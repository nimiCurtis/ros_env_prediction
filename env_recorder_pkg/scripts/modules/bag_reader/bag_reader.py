############## 
## TO DO ##
# documentation
# path and folder organizing

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
    """_summary_
        """        
    
    def __init__(self, bag_file):
        """ Constructor of READER object

            Args:
                bag_file (string): path string of .bag file
            """            
        
        self.bag_file = bag_file
        self.bag_read = bagreader(bag_file) # get bagreader object using bagpy library 
        
        # create or use exist metadata json file
        self.metadata_file = os.path.join(self.bag_read.datafolder, "metadata.json")    
        self.MetaData = {}
        if os.path.exists(self.metadata_file): # if already exist load the content of its file to self.Metadata dict
            with open(self.metadata_file,"r") as json_file:
                self.MetaData = json.load(json_file)
        
        # initialize data frames
        self.rgb_df = pd.DataFrame()
        self.depth_df = pd.DataFrame()
        self.imu_df = pd.DataFrame()
        self.confidence_df = pd.DataFrame()

        # initialize counter
        self.frame_count = 0

    def read(self):
        """This function read the data from the bag and modify the data frames acordingly
            """        
        
        # set topic df
        self.topic_df = self.bag_read.topic_table
        
        # read and set imu_df
        for index, topic_row in self.topic_df.iterrows():
            if (topic_row['Types']=='sensor_msgs/Imu') and  topic_row['Message Count']!=0: # stop when topic is the imu topic and its not empty
                if os.path.exists(self.metadata_file):  # if already exist load the data from its csv
                    self.imu_df = pd.read_csv(self.MetaData["imu"]) 
                else:
                    imu_file = self.bag_read.message_by_topic(topic_row['Topics']) # create the csv file  
                    self.imu_df = pd.read_csv(imu_file) # set the df
                    self.MetaData["imu"] = imu_file # save the path to metadata

            if (topic_row['Types']=='sensor_msgs/Image') and  topic_row['Message Count']!=0: # stop when topic is Image kind and its not empty
                self.init_image_df(topic_row['Topics']) 
        
        if not os.path.exists(self.metadata_file): # if meta data not exist, create it using json.dump
            with open(self.metadata_file, "w") as json_file:
                json.dump(self.MetaData, json_file, indent=3)


    def init_image_df(self,topic):
        """_summary_

            Args:
                topic (string): name of the topic
            """        

        topic_split = topic.split('/')

        # handle with depth topic
        if 'depth' in topic_split:
            self.depth_df = self.set_image_df(topic,'depth')

        # handle with rgb topic
        if 'rgb' in topic_split: 
            self.rgb_df = self.set_image_df(topic,'rgb')
        
        # handle with confidence
        if 'confidence' in topic_split:
            self.confidence_df = self.set_image_df(topic,'confidence')

        # add disparity

    def extract_images(self, topic, dir, img_type):
        """_summary_

            Args:
                topic (string): name of topic
                dir (string): path to the directory
                img_type (string): type of image rgb/depth/confidence/disparity

            Returns:
                frame_path_list (string array): list of the paths to the .jpg files
                np_path_list (string array): list of the path to the .npy files of the values
            """

        bag = rosbag.Bag(self.bag_file, "r") # read the bag file using rosbag library
        bridge = CvBridge() # create bridge object
        
        # initialize paths lists
        frame_path_list = [] 
        numpy_path_list = []

        # iterate the topic msgs
        for topic, msg, t in bag.read_messages(topics=topic):
            
            # convert image msgs to opencv format
            try:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") # "passthrough" = keep the same encoding format of the image
            except CvBridgeError as e:
                print(e)
            
            frame_path = os.path.join(dir, "frame%06i.jpg" % self.frame_count) 

            if (img_type != "rgb"): # if not rgb type save the real values as .npy file, and update np list
                numpy_path_list.append(self.save_np_data(cv_img,dir))

            frame_path_list.append(frame_path) # update frame path list
            cv2.imwrite(frame_path, cv_img)    # save img

            self.frame_count += 1

        bag.close()
        
        print("images saved") # convert to log
        
        return frame_path_list, numpy_path_list
        
    def get_synced_df(self):
        # make a df depend on timestamp
        
        pass 
    
    def set_image_df(self,topic,image_type):
        """ This function creating the image type dataframes and csv files including the outputs from 
        extract_images function 
        
        Args:
            topic (string): name of topic 
            image_type (strip): type of image rgb/depth/confidence/disparity 

        Returns:
            df (data frame): pandas data frame depends on the type of image
        """   

        self.frame_count = 0 # initialize counter every time using this function

        if os.path.exists(self.metadata_file): # if metadata exist load from existing csv
            df = pd.read_csv(self.MetaData[image_type])
            
        else:
            dir = os.path.join(self.bag_read.datafolder,image_type)
            dir_vals = os.path.join(dir,'vals')

            # crating data folder for the relevant image type
            if not os.path.exists(dir):
                os.mkdir(dir)
                if not os.path.exists(dir_vals) and image_type!='rgb': # if not rgb type create values directory
                    os.mkdir(dir_vals) 
            
            # create csv temp file using the bagreader library - temporarily because it deosnt handle good with imgs 
            tmp_file = self.bag_read.message_by_topic(topic)
            df = pd.read_csv(tmp_file)
            df.drop('data',inplace = True , axis =1) # drop the data column because its containing garbage data 
            
            if image_type == 'rgb': # if rgb save only frame paths
                df['frame_path'], _ = self.extract_images(topic, dir, image_type)
            else:
                df['frame_path'], df['np_path'] = self.extract_images(topic, dir, image_type)

            df.to_csv(tmp_file) # create updated csv file
            self.MetaData[image_type] = tmp_file # save to metadata
        
        return df

    def save_np_data(self,img,dir, d_type=np.float32):
        """This function save values of the image into .npy file 

            Args:
                img (numpy array): image matrice 
                dir (string): path to directory
                d_type (np.type , optional): saving in specific format. Defaults to np.float32.

            Returns:
                numpy_path (string): path to the saved file
            """

        img_array = np.array(img, dtype=d_type) # convert type to d_type
        numpy_path = os.path.join(dir, "vals/np_values%06i.npy" % self.frame_count)
        np.save(numpy_path, img_array)
        return numpy_path
            
    

if __name__ == '__main__':
    read = READER('/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-10-27-12-42-56.bag')
    read.read()
 