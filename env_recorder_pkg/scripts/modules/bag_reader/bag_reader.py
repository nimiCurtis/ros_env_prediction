
# reading bag files dependencies
from bagpy import bagreader
import numpy as np
import pandas as pd
import json
from functools import reduce

# extracting images dependendcies
import os
import argparse
import cv2
import rosbag
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError

class BagReader():
    """_summary_
        """    
    
    def __init__(self, bag_file):
        """ Constructor of BagReader object

            Args:
                bag_file (string): path string of .bag file
            """            
        
        self.bag_file = bag_file
        self.bag_read = bagreader(bag_file) # get bagreader object using bagpy library 
        
        # create or use exist metadata json file
        self.metadata_file = os.path.join(self.bag_read.datafolder, "metadata.json")    
        self.MetaData ={}
        if os.path.exists(self.metadata_file): # if already exist load the content of its file to self.Metadata dict
            with open(self.metadata_file,"r") as json_file:
                self.MetaData = json.load(json_file)
        else:
            self.MetaData["exported"] = False
        
        # initialize data frames dictionary
        self.dfs = {}

        # initialize counter
        self.frame_count = 0
    
    def get_data(self):
        """This function export/read bag data based on metadata 'exported' status
            """        

        if self.MetaData["exported"]:
            self.read()
        else:
            self.export()

    def read(self):
        """This function reading bag data from existing folders
            """        

        print("[INFO]  Bag already exported, Reading data ...")
        
        names = ["imu","rgb","depth","confidence","disparity","pointclod","tf","synced"] # change based on configuration
        
        for name in names:
            if name in self.MetaData:
                self.dfs[name] = pd.read_csv(self.MetaData[name])

    def export(self):
        """This function export the data from the bag and modify the data frames accordingly
            """        
        

        print("[INFO]  Bag doesn't exported, Exporting data ...")

        # set topic df
        self.topic_df = self.bag_read.topic_table 
        
        # read and set imu_df
        for index, topic_row in self.topic_df.iterrows():
            if (topic_row['Types']=='sensor_msgs/Imu') and  topic_row['Message Count']!=0: # stop when topic is the imu topic and its not empty
                    imu_file = self.bag_read.message_by_topic(topic_row['Topics']) # create the csv file  
                    self.dfs["imu"] = pd.read_csv(imu_file) # set the df
                    self.dfs["imu"].drop_duplicates(subset=['header.stamp.secs','header.stamp.nsecs'],ignore_index=True,inplace=True)
                    self.dfs["imu"].to_csv(imu_file) # rewrite imu csv
                    self.MetaData["imu"] = imu_file # save the path to metadata

            if ((topic_row['Types']=='sensor_msgs/Image')or(topic_row['Types']=='stereo_msgs/DisparityImage')) and  topic_row['Message Count']!=0: # stop when topic is Image/Stereo kind and its not empty
                self.init_image_df(topic_row['Topics']) 
        
        self.dfs["synced"] = self.sync_imu_with([v for k, v in self.dfs.items() if k!='imu'])
        synced_file = os.path.join(self.bag_read.datafolder,"synced.csv")
        self.dfs["synced"].to_csv(synced_file)
        self.MetaData["synced"] = synced_file  

        # save the path to metadata
        # change exported status and dump MetaData to a json file
        self.MetaData["exported"] = True
        with open(self.metadata_file, "w") as json_file:
            json.dump(self.MetaData, json_file, indent=3)

    def init_image_df(self,topic):
        """This function initializing image data frames using set_image_df function per topic

            Args:
                topic (string): name of the topic
            """        

        topic_split = topic.split('/')

        names = ["rgb","depth","confidence","disparity","pointclod"] # change based on configuration

        for name in names:
            if name in topic_split:
                self.dfs[name] = self.set_image_df(topic,name)
                break

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

        # set the img data folder and numpy values folder
        dir = os.path.join(self.bag_read.datafolder,img_type)
        dir_vals = os.path.join(dir,'vals')

        # crating data folder for the relevant image type
        if not os.path.exists(dir):
            os.mkdir(dir)
            if not os.path.exists(dir_vals) : # create values directory
                os.mkdir(dir_vals) 
        
        # create csv temp file using the bagreader library - temporarily because it deosnt handle good with imgs 
        tmp_file = self.bag_read.message_by_topic(topic)
        df = pd.read_csv(tmp_file)
        if (img_type == 'disparity'):
            df.columns = df.columns.str.replace('image.data','data',regex=True) # rename disparity columns starting with 'image.'
        df.drop('data',inplace = True , axis =1) # drop the data column because its containing garbage data

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
            

                if (img_type == "depth"):
                    #values_array = values_array/1000.0 # convert to meters
                    cv_img = self.get_depth_normalization(values_array.copy()) # get normalized image of depth values

                if (img_type == "disparity"):
                    cv_img = self.get_disparity_colormap(values_array.copy(),msg) # get color map of dispatity values
                
            else:
                values_array = np.array(cv_img, dtype=np.int32)

            numpy_path_list.append(self.save_np_data(values_array,dir)) # save values
            frame_path_list.append(frame_path) # update frame path list
            cv2.imwrite(frame_path, cv_img)    # save img

            self.frame_count += 1

        bag.close()
        print(f"[INFO]  {img_type} folder saved") # convert it to log in the future
        
        return frame_path_list, numpy_path_list
        
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
        normal_dist = msg.max_disparity - msg.min_disparity
        shifted_disparity = (values_array - msg.min_disparity)                    # shift values to get rid from negetive vals | current_min = 0
        scaled_disparity = (shifted_disparity*255)/normal_dist            # normalize to fall between (0,255) 
        scaled_disparity = np.clip(scaled_disparity,0,255)                        # clip , not sure if totaly necssary
        scaled_disparity = scaled_disparity.astype(np.uint8)                      # change format 
        colormap_disparity= cv2.applyColorMap(scaled_disparity,cv2.COLORMAP_JET)  # apply colormap

        return cv2.cvtColor(colormap_disparity, cv2.COLOR_BGR2RGB)                # convert to rgb --> red = close dist, blue = far dist

    def sync_imu_with(self,dfs):
        cols = ['np_path', 'frame_path']
        df_sync = self.dfs["imu"].copy()
        for df in dfs:
            key = [k for k, v in self.dfs.items() if v.equals(df)]
            df = df.rename(columns={c: c+f'_{key[0]}' for c in df.columns if c in cols})
            df_sync = pd.merge(df_sync,df[["header.stamp.secs","header.stamp.nsecs",f"np_path_{key[0]}",f"frame_path_{key[0]}"]],on=["header.stamp.secs","header.stamp.nsecs"],how="right")    
        
        return df_sync

if __name__ == '__main__':
    bag_obj = BagReader('/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-11-08-10-13-11.bag')
    bag_obj.get_data()
    a=1
