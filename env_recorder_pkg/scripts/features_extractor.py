import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from modules.bag_reader.bag_reader import READER
from modules.bag_processor.bag_processor import DepthProcessor

def main():

    pass

if __name__ == '__main__':
    bag_read = READER('/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-10-27-12-42-56.bag')
    bag_read.read()
    bag_read.depth_df

    dp = DepthProcessor()
    feat_df = dp.get_features_df(bag_read.depth_df)
    dp.imshow_with_grid(feat_df.frame_path[20])
    
