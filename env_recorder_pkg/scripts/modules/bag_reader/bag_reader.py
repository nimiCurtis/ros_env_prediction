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
        self.disparity_df = pd.DataFrame()

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

            if ((topic_row['Types']=='sensor_msgs/Image')or(topic_row['Types']=='stereo_msgs/DisparityImage')) and  topic_row['Message Count']!=0: # stop when topic is Image/Stereo kind and its not empty
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

        if 'disparity' in topic_split:
            self.disparity_df = self.set_image_df(topic, 'disparity')

    def set_image_df(self,topic,img_type):
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
            df = pd.read_csv(self.MetaData[img_type])
            
        else:
            dir = os.path.join(self.bag_read.datafolder,img_type)
            dir_vals = os.path.join(dir,'vals')

            # crating data folder for the relevant image type
            if not os.path.exists(dir):
                os.mkdir(dir)
                if not os.path.exists(dir_vals) and img_type != 'rgb': # if not rgb type create values directory
                    os.mkdir(dir_vals) 
            
            # create csv temp file using the bagreader library - temporarily because it deosnt handle good with imgs 
            tmp_file = self.bag_read.message_by_topic(topic)
            df = pd.read_csv(tmp_file)
            if (img_type != 'disparity'):
                df.drop('data',inplace = True , axis =1) # drop the data column because its containing garbage data 
            else: 
                df.drop('image.data',inplace = True , axis =1) # drop image.data column from disparity df

            if img_type == 'rgb' : # if rgb save only frame paths
                df['frame_path'], _ = self.extract_images(topic, dir, img_type)
            else:
                df['frame_path'], df['np_path'] = self.extract_images(topic, dir, img_type)

            df.to_csv(tmp_file) # create updated csv file
            self.MetaData[img_type] = tmp_file # save to metadata
        
        return df

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
                if img_type != 'disparity':
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") # "passthrough" = keep the same encoding format of the image
                else:
                    cv_img = bridge.imgmsg_to_cv2(msg.image, desired_encoding="passthrough")
            except CvBridgeError as e:
                print(e)
            
            frame_path = os.path.join(dir, "frame%06i.jpg" % self.frame_count) 

            if (img_type != "rgb"): # if not rgb type save the real values as .npy file, and update np list
                values_array = np.array(cv_img, dtype=np.float32) # convert type to d_type --> extract depth values
                numpy_path_list.append(self.save_np_data(values_array,dir)) # save values
                if (img_type == "depth"):
                    cv_img = self.get_depth_normalization(values_array) # get normalized image of depth values

                if (img_type == "disparity"):
                    cv_img = self.get_disparity_colormap(values_array,msg) # get color map of dispatity values
                    


            frame_path_list.append(frame_path) # update frame path list
            cv2.imwrite(frame_path, cv_img)    # save img

            self.frame_count += 1

        bag.close()
        
        print(f"[INFO] {img_type} folder saved") # convert to log
        
        return frame_path_list, numpy_path_list
        
    def get_synced_df(self):
        # make a df depend on timestamp
        
        pass 
    
    def get_depth_normalization(self, img):
        """Normalize the depth image to fall between 0 (black) and 1 (white)

        Args:
            img (numpy array): image matrice to be normalized

        Returns:
            img (numpy array): normalized image
        """        
        
        cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
        img = img*255
        
        return img

    def get_disparity_colormap(self,values_array, msg):
        """Get 

            Args:
                values_array (numpy array): image disparity values
                msg (stereo_msgs/Disparity): ROS msg of type Disparity. see: http://docs.ros.org/en/melodic/api/stereo_msgs/html/msg/DisparityImage.html

            Returns:
                (numpy array): color map image of the disparity values
            """        

        multiplier = 255/(msg.max_disparity - msg.min_disparity)
        scaledDisparity = (values_array - msg.min_disparity)*multiplier + 0.5
        scaledDisparity = np.clip(scaledDisparity,0,255)
        scaledDisparity = scaledDisparity.astype(np.uint8)
        scaledDisparity = cv2.applyColorMap(scaledDisparity,cv2.COLORMAP_JET)
        return cv2.cvtColor(scaledDisparity, cv2.COLOR_BGR2RGB)


    def save_np_data(self,values_array,dir):
        """This function save values of the image into .npy file 

            Args:
                values_array (numpy array): depth_values matrice 
                dir (string): path to directory
                

            Returns:
                numpy_path (string): path to the saved file
            """

        numpy_path = os.path.join(dir, "vals/np_values%06i.npy" % self.frame_count)
        np.save(numpy_path, values_array)
        return numpy_path
            
    

if __name__ == '__main__':
    read = READER('/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-11-06-19-06-55.bag')
    read.read()
