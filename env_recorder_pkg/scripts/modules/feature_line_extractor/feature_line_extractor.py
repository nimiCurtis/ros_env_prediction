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

        g_mean = []
        g_std = []
        g_max = []
        g_min = []
        g_argmax = []
        g_peaks_num = []
        g_peaks_mean = []
        g_clpeak_idx = []
        g_peaks_idx_mean = []


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
                g_mean.append(features_dic['gradient_mean'])
                g_std.append(features_dic['gradient_std'])
                g_max.append(features_dic['gradient_max'])
                g_min.append(features_dic['gradient_min'])
                g_argmax.append(features_dic['gradient_argmax'])
                g_peaks_num.append(features_dic['gradient_peaks_num'])
                g_peaks_mean.append(features_dic['gradient_peaks_mean'])
                g_clpeak_idx.append(features_dic['gradient_clpeak'])
                g_peaks_idx_mean.append(features_dic['gradient_peaks_idx_mean'])

        # create data_frame
        df = pd.DataFrame({'d_mean':d_mean,
                            'd_std':d_std,
                            'd_max':d_max,
                            'd_min':d_min,
                            'd_peaks_num':d_peaks_num,
                            'd_peaks_mean':d_peaks_mean,
                            'd_peaks_idx_mean':d_peaks_idx_mean,
                            'g_mean':g_mean,
                            'g_std':g_std,
                            'g_max':g_max,
                            'g_max':g_min,
                            'g_argmax':g_argmax,
                            'g_peaks_num':g_peaks_num,
                            'g_peaks_mean':g_peaks_mean,
                            'g_clpeak_idx':g_clpeak_idx,
                            'g_peaks_idx_mean':g_peaks_idx_mean})


        
        df = df.round(decimals=4)

        if os.path.exists(dir_path+"/features.csv"):
            df0 = pd.read_csv(dir_path+"/features.csv")
            if 'labels' in df0.keys():
                df['labels'] = df0['labels']
        
        # save to csv
        df.to_csv(dir_path+"/features.csv")

    def extract(self,depth):
        delta = int(len(depth)/3)
        depth_up = depth[:delta]
        depth_mid = depth[delta:2*delta]
        depth_bot = depth[2*delta:]

        gradient = np.gradient(depth)
        # extract depth features
        features_dic = {}

        features_dic['gradient'] = gradient

        features_dic['depth_mean'] = np.mean(depth_mid)
        features_dic['depth_std'] = np.std(depth_mid)
        features_dic['depth_max'] = np.max(depth_mid)
        features_dic['depth_min'] = np.min(depth_mid)
        
        #peaks
        depth_peaks, d_peaks_properties = ss.find_peaks(depth_mid,distance=3,width=3)
        depth_minima, d_min_peaks_properties =  ss.find_peaks(-depth_mid,distance=3,width=3)
        depth_peaks  = np.concatenate([depth_peaks,depth_minima]) + delta
        features_dic['depth_peaks'] = depth_peaks
        features_dic['depth_peaks_num'] = len(depth_peaks)
        if features_dic['depth_peaks_num'] == 0:
            features_dic['depth_peaks_mean'] = 0
            features_dic['depth_peaks_idx_mean'] = 0 
        else:
            features_dic['depth_peaks_mean'] = np.mean((depth[depth_peaks]))
            features_dic['depth_peaks_idx_mean'] = np.mean(depth_peaks)

        features_dic['gradient_mean'] = np.mean(gradient)
        features_dic['gradient_std'] = np.std(gradient)
        features_dic['gradient_max'] = np.max(gradient)
        features_dic['gradient_min'] = np.min(gradient)
        features_dic['gradient_argmax'] = np.argmax(np.abs(gradient))

        gradient_peaks, g_peaks_properties = ss.find_peaks(gradient,width=1,distance=1,threshold=0.003)
        gradient_minima, d_min_peaks_properties =  ss.find_peaks(-gradient,width=1,distance=1,threshold=0.003)
        gradient_peaks  = np.concatenate([gradient_peaks,gradient_minima])
        
        
        features_dic['gradient_peaks'] = gradient_peaks
        features_dic['gradient_peaks_num'] = len(gradient_peaks)
        if features_dic['gradient_peaks_num'] ==0:
            features_dic['gradient_peaks_mean'] = 0
            features_dic['gradient_clpeak'] = 0
            features_dic['gradient_peaks_idx_mean'] = 0
        else:
            features_dic['gradient_peaks_mean'] = np.mean(np.abs(gradient[gradient_peaks]))
            features_dic['gradient_clpeak'] = gradient_peaks[-1]
            features_dic['gradient_peaks_idx_mean'] = np.mean(gradient_peaks)

        return features_dic


def main():
    extractor = FeatLineExtract()
    extractor.extract_dir('/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-12-12-15-23-00/plots/feature')

if __name__ == '__main__':
    main()

