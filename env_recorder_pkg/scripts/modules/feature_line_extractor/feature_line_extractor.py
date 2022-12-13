# imports
import os
import hydra
from omegaconf import DictConfig
from datetime import datetime
from typing import Union
import cv2
import numpy as np
import scipy.signal as ss
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class FeatLineExtract:

    def __init__(self) -> None:
        
        pass

    def extract_dir(self,dir_path):
        # arrays init
        d_mean = []
        d_std = []
        d_max = []
        d_min = []
        d_peaks_num = []
        d_peaks_mean = []
        d_peaks_idx_mean = []
        s_mean = []
        s_std = []
        s_max = []
        s_argmax = []
        s_peaks_num = []
        s_peaks_mean = []
        s_clpeak_idx = []
        s_peaks_idx_mean = []


        # iterate dir
        for filename in os.scandir(dir_path): 
            if filename.is_file() and filename.path.split('.')[-1]=='npy':
            #load np arrays
                depth = np.load(filename)
            
                features_dic = self.extract(depth)
            

        # append
                d_mean.append(features_dic['depth_mean'])
                d_std.append(features_dic['depth_std'])
                d_max.append(features_dic['depth_max'])
                d_min.append(features_dic['depth_min'])
                d_peaks_num.append(features_dic['depth_peaks_num'])
                d_peaks_mean.append(features_dic['depth_peaks_mean'])
                d_peaks_idx_mean.append(features_dic['depth_peaks_idx_mean'])
                s_mean.append(features_dic['subtracted_mean'])
                s_std.append(features_dic['subtracted_std'])
                s_max.append(features_dic['subtracted_max'])
                s_argmax.append(features_dic['subtracted_argmax'])
                s_peaks_num.append(features_dic['subtracted_peaks_num'])
                s_peaks_mean.append(features_dic['subtracted_peaks_mean'])
                s_clpeak_idx.append(features_dic['subtracted_clpeak'])
                s_peaks_idx_mean.append(features_dic['subtracted_peaks_idx_mean'])

        # create data_frame
        df = pd.DataFrame({'d_mean':d_mean,
                            'd_std':d_std,
                            'd_max':d_max,
                            'd_min':d_min,
                            'd_peaks_num':d_peaks_num,
                            'd_peaks_mean':d_peaks_mean,
                            'd_peaks_idx_mean':d_peaks_idx_mean,
                            's_mean':s_mean,
                            's_std':s_std,
                            's_max':s_max,
                            's_argmax':s_argmax,
                            's_peaks_num':s_peaks_num,
                            's_peaks_mean':s_peaks_mean,
                            's_clpeak_idx':s_clpeak_idx,
                            's_peaks_idx_mean':s_peaks_idx_mean})
        
        # save to csv
        df.to_csv(dir_path+"/features.csv")

    def extract(self,depth):
        subtracted = np.subtract(depth[1:],depth[:-1])
        subtracted = np.concatenate([[0],subtracted])

        # extract depth features
        features_dic = {}
        features_dic['subtracted'] = subtracted

        features_dic['depth_mean'] = np.mean(depth)
        features_dic['depth_std'] = np.std(depth)
        features_dic['depth_max'] = np.max(depth)
        features_dic['depth_min'] = np.min(depth)
        #peaks
        depth_peaks, properties_line = ss.find_peaks(depth,distance=10)
        features_dic['depth_peaks'] = depth_peaks
        features_dic['depth_peaks_num'] = len(depth_peaks)
        features_dic['depth_peaks_mean'] = np.mean(depth[depth_peaks])
        features_dic['depth_peaks_idx_mean'] = np.mean(depth_peaks)
    
    # extract depth features
        features_dic['subtracted_mean'] = np.mean(subtracted)
        features_dic['subtracted_std'] = np.std(subtracted)
        features_dic['subtracted_max'] = np.max(subtracted)
        features_dic['subtracted_argmax'] = np.argmax(subtracted)
        #peaks
        subtracted_peaks, properties_line = ss.find_peaks(subtracted,distance=20)
        features_dic['subtracted_peaks'] = subtracted_peaks
        features_dic['subtracted_peaks_num'] = len(subtracted_peaks)
        features_dic['subtracted_peaks_mean'] = np.mean(subtracted[subtracted_peaks])
        features_dic['subtracted_clpeak'] = subtracted_peaks[-1]
        features_dic['subtracted_peaks_idx_mean'] = np.mean(subtracted_peaks)

        return features_dic


def main():
    extractor = FeatLineExtract()
    extractor.extract_dir('/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-11-08-10-13-11/plots/feature')

if __name__ == '__main__':
    main()

