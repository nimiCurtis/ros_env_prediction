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

        # all
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

        # up
        ud_mean = []
        ud_std = []
        ud_max = []
        ud_min = []
        ud_delta = []
        ud_peaks_num = []
        ud_peaks_mean = []
        ug_mean = []
        ug_std = []
        ug_max = []
        ug_min = []
        ug_peaks_num = []
        ug_peaks_mean = []


        # mid
        md_mean = []
        md_std = []
        md_max = []
        md_min = []
        md_peaks_num = []
        md_peaks_mean = []
        mg_mean = []
        mg_std = []
        mg_max = []
        mg_min = []
        mg_peaks_num = []
        mg_peaks_mean = []

        # bot
        bd_mean = []
        bd_std = []
        bd_max = []
        bd_min = []
        bd_peaks_num = []
        bd_peaks_mean = []
        bg_mean = []
        bg_std = []
        bg_max = []
        bg_min = []
        bg_peaks_num = []
        bg_peaks_mean = []
        # iterate dir
        for i in range(len(os.listdir(dir_path))-2):
            filename = f'plot_{i}.npy'
            file_path =  os.path.join(dir_path, filename)
            if os.path.isfile(file_path):


            

            #load np arrays
                depth = np.load(file_path)
            
                features_dic,_ = self.extract(depth)
            

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

                ### up
                ud_mean.append(features_dic['up_depth_mean'])
                ud_std.append(features_dic['up_depth_std'])
                ud_max.append(features_dic['up_depth_max'])
                ud_min.append(features_dic['up_depth_min'])
                ud_delta.append(features_dic['up_depth_delta'])
                ud_peaks_num.append(features_dic['up_depth_peaks_num'])
                ud_peaks_mean.append(features_dic['up_depth_peaks_mean'])
                ug_mean.append(features_dic['up_gradient_mean'])
                ug_std.append(features_dic['up_gradient_std'])
                ug_max.append(features_dic['up_gradient_max'])
                ug_min.append(features_dic['up_gradient_min'])
                ug_peaks_num.append(features_dic['up_gradient_peaks_num'])
                ug_peaks_mean.append(features_dic['up_gradient_peaks_mean'])

                ### mid
                md_mean.append(features_dic['mid_depth_mean'])
                md_std.append(features_dic['mid_depth_std'])
                md_max.append(features_dic['mid_depth_max'])
                md_min.append(features_dic['mid_depth_min'])
                md_peaks_num.append(features_dic['mid_depth_peaks_num'])
                md_peaks_mean.append(features_dic['mid_depth_peaks_mean'])
                mg_mean.append(features_dic['mid_gradient_mean'])
                mg_std.append(features_dic['mid_gradient_std'])
                mg_max.append(features_dic['mid_gradient_max'])
                mg_min.append(features_dic['mid_gradient_min'])
                mg_peaks_num.append(features_dic['mid_gradient_peaks_num'])
                mg_peaks_mean.append(features_dic['mid_gradient_peaks_mean'])

                #### bot
                bd_mean.append(features_dic['bot_depth_mean'])
                bd_std.append(features_dic['bot_depth_std'])
                bd_max.append(features_dic['bot_depth_max'])
                bd_min.append(features_dic['bot_depth_min'])
                bd_peaks_num.append(features_dic['bot_depth_peaks_num'])
                bd_peaks_mean.append(features_dic['bot_depth_peaks_mean'])
                bg_mean.append(features_dic['bot_gradient_mean'])
                bg_std.append(features_dic['bot_gradient_std'])
                bg_max.append(features_dic['bot_gradient_max'])
                bg_min.append(features_dic['bot_gradient_min'])
                bg_peaks_num.append(features_dic['bot_gradient_peaks_num'])
                bg_peaks_mean.append(features_dic['bot_gradient_peaks_mean'])
                
            else:
                pass

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
                            'g_min':g_min,
                            'g_argmax':g_argmax,
                            'g_peaks_num':g_peaks_num,
                            'g_peaks_mean':g_peaks_mean,
                            'g_clpeak_idx':g_clpeak_idx,
                            'g_peaks_idx_mean':g_peaks_idx_mean,
                            # up
                            'ud_mean':ud_mean,
                            'ud_std':ud_std,
                            'ud_max':ud_max,
                            'ud_min':ud_min,
                            'ud_delta':ud_delta,
                            'ud_peaks_num':ud_peaks_num,
                            'ud_peaks_mean':ud_peaks_mean,
                            'ug_mean':ug_mean,
                            'ug_std':ug_std,
                            'ug_max':ug_max,
                            'ug_min':ug_min,
                            'ug_peaks_num':ug_peaks_num,
                            'ug_peaks_mean':ug_peaks_mean,
                            # mid
                            'md_mean':md_mean,
                            'md_std':md_std,
                            'md_max':md_max,
                            'md_min':md_min,
                            'md_peaks_num':md_peaks_num,
                            'md_peaks_mean':md_peaks_mean,
                            'mg_mean':mg_mean,
                            'mg_std':mg_std,
                            'mg_max':mg_max,
                            'mg_min':mg_min,
                            'mg_peaks_num':mg_peaks_num,
                            'mg_peaks_mean':mg_peaks_mean,
                            ## bot
                            'bd_mean':bd_mean,
                            'bd_std':bd_std,
                            'bd_max':bd_max,
                            'bd_min':bd_min,
                            'bd_peaks_num':bd_peaks_num,
                            'bd_peaks_mean':bd_peaks_mean,
                            'bg_mean':bg_mean,
                            'bg_std':bg_std,
                            'bg_max':bg_max,
                            'bg_min':bg_min,
                            'bg_peaks_num':bg_peaks_num,
                            'bg_peaks_mean':bg_peaks_mean
                            })


        
        df = df.round(decimals=4)

        if os.path.exists(dir_path+"/features.csv"):
            df0 = pd.read_csv(dir_path+"/features.csv")
            if 'labels' in df0.keys():
                df['labels'] = df0['labels']
        
        # save to csv
        df.to_csv(dir_path+"/features.csv")

    def extract(self,depth):
        delta = int(len(depth)/3)
        depth_up = depth[:delta].copy()
        depth_mid = depth[delta:2*delta].copy()
        depth_bot = depth[2*delta:].copy()

        
        # extract depth features
        features_dic = {}
        return_dic = {}
        # features_dic.update(self.extract(depth))
        # features_dic.update(self.extract(depth,stop=delta,name='up'))
        # features_dic.update(self.extract(depth,start=delta,stop=2*delta,name='mid'))
        # features_dic.update(self.extract(depth,start=2*delta,name='bot'))
        
        gradient = np.gradient(depth)
        return_dic['gradient'] = gradient.copy()
        gradient_up = gradient[:delta].copy()
        gradient_mid = gradient[delta:2*delta].copy()
        gradient_bot = gradient[2*delta:].copy()

        ##### all
        features_dic['depth_mean'] = np.mean(depth)
        features_dic['depth_std'] = np.std(depth)
        features_dic['depth_max'] = np.max(depth)
        features_dic['depth_min'] = np.min(depth)
        
        #peaks
        depth_peaks, d_peaks_properties = ss.find_peaks(depth,distance=3,width=3,threshold=0.007)
        depth_minima, d_min_peaks_properties =  ss.find_peaks(-depth,distance=3,width=3,threshold=0.003)
        # diff between max and min
        depth_peaks  = np.concatenate([depth_peaks,depth_minima])
        return_dic['depth_peaks'] = depth_peaks
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

        gradient_peaks, g_peaks_properties = ss.find_peaks(gradient,width=1,distance=1,threshold=0.0035)
        gradient_minima, d_min_peaks_properties =  ss.find_peaks(-gradient,width=1,distance=1,threshold=0.0035)
        gradient_peaks  = np.concatenate([gradient_peaks,gradient_minima])
        
        return_dic['gradient_peaks'] = gradient_peaks
        features_dic['gradient_peaks_num'] = len(gradient_peaks)
        if features_dic['gradient_peaks_num'] ==0:
            features_dic['gradient_peaks_mean'] = 0
            features_dic['gradient_clpeak'] = 0
            features_dic['gradient_peaks_idx_mean'] = 0
        else:
            features_dic['gradient_peaks_mean'] = np.mean(np.abs(gradient[gradient_peaks]))
            features_dic['gradient_clpeak'] = gradient_peaks[-1]
            features_dic['gradient_peaks_idx_mean'] = np.mean(gradient_peaks)


        ######## up
        features_dic['up_depth_mean'] = np.mean(depth_up)
        features_dic['up_depth_std'] = np.std(depth_up)
        features_dic['up_depth_max'] = np.max(depth_up)
        features_dic['up_depth_min'] = np.min(depth_up)
        features_dic['up_depth_delta'] = features_dic['up_depth_max'] - features_dic['up_depth_min']
    

        #peaks
        depth_peaks, d_peaks_properties = ss.find_peaks(depth_up,distance=3,width=3)
        depth_minima, d_min_peaks_properties =  ss.find_peaks(-depth_up,distance=3,width=3)
        depth_peaks  = np.concatenate([depth_peaks,depth_minima])
        #features_dic['up_depth_peaks'] = depth_peaks
        features_dic['up_depth_peaks_num'] = len(depth_peaks)
        if features_dic['up_depth_peaks_num'] == 0:
            features_dic['up_depth_peaks_mean'] = 0
        else:
            features_dic['up_depth_peaks_mean'] = np.mean((depth[depth_peaks]))

        features_dic['up_gradient_mean'] = np.mean(gradient_up)
        features_dic['up_gradient_std'] = np.std(gradient_up)
        features_dic['up_gradient_max'] = np.max(gradient_up)
        features_dic['up_gradient_min'] = np.min(gradient_up)

        gradient_peaks, g_peaks_properties = ss.find_peaks(gradient_up,width=1,distance=1,threshold=0.003)
        gradient_minima, g_min_peaks_properties =  ss.find_peaks(-gradient_up,width=1,distance=1,threshold=0.003)
        gradient_peaks  = np.concatenate([gradient_peaks,gradient_minima])
        
        features_dic['up_gradient_peaks_num'] = len(gradient_peaks)
        if features_dic['up_gradient_peaks_num'] ==0:
            features_dic['up_gradient_peaks_mean'] = 0
        else:
            features_dic['up_gradient_peaks_mean'] = np.mean(np.abs(gradient[gradient_peaks]))

        ######## mid
        features_dic['mid_depth_mean'] = np.mean(depth_mid)
        features_dic['mid_depth_std'] = np.std(depth_mid)
        features_dic['mid_depth_max'] = np.max(depth_mid)
        features_dic['mid_depth_min'] = np.min(depth_mid)
        # add delta

        #peaks
        depth_peaks, d_peaks_properties = ss.find_peaks(depth_mid,distance=3,width=3)
        depth_minima, d_min_peaks_properties =  ss.find_peaks(-depth_mid,distance=3,width=3)
        depth_peaks  = np.concatenate([depth_peaks,depth_minima]) + delta
        #features_dic['mid_depth_peaks'] = depth_peaks
        features_dic['mid_depth_peaks_num'] = len(depth_peaks)
        if features_dic['mid_depth_peaks_num'] == 0:
            features_dic['mid_depth_peaks_mean'] = 0
        else:
            features_dic['mid_depth_peaks_mean'] = np.mean((depth[depth_peaks]))

        features_dic['mid_gradient_mean'] = np.mean(gradient_mid)
        features_dic['mid_gradient_std'] = np.std(gradient_mid)
        features_dic['mid_gradient_max'] = np.max(gradient_mid)
        features_dic['mid_gradient_min'] = np.min(gradient_mid)

        gradient_peaks, g_peaks_properties = ss.find_peaks(gradient_mid,width=1,distance=1,threshold=0.003)
        gradient_minima, g_min_peaks_properties =  ss.find_peaks(-gradient_mid,width=1,distance=1,threshold=0.003)
        gradient_peaks  = np.concatenate([gradient_peaks,gradient_minima]) +delta
        
        #features_dic['up_gradient_peaks'] = gradient_peaks
        features_dic['mid_gradient_peaks_num'] = len(gradient_peaks)
        if features_dic['mid_gradient_peaks_num'] ==0:
            features_dic['mid_gradient_peaks_mean'] = 0
        else:
            features_dic['mid_gradient_peaks_mean'] = np.mean(np.abs(gradient[gradient_peaks]))


        ######## bot
        features_dic['bot_depth_mean'] = np.mean(depth_bot)
        features_dic['bot_depth_std'] = np.std(depth_bot)
        features_dic['bot_depth_max'] = np.max(depth_bot)
        features_dic['bot_depth_min'] = np.min(depth_bot)
        
        #peaks
        depth_peaks, d_peaks_properties = ss.find_peaks(depth_bot,distance=3,width=3)
        depth_minima, d_min_peaks_properties =  ss.find_peaks(-depth_bot,distance=3,width=3)
        depth_peaks  = np.concatenate([depth_peaks,depth_minima]) + 2*delta
        #features_dic['bot_depth_peaks'] = depth_peaks
        features_dic['bot_depth_peaks_num'] = len(depth_peaks)
        if features_dic['bot_depth_peaks_num'] == 0:
            features_dic['bot_depth_peaks_mean'] = 0
        else:
            features_dic['bot_depth_peaks_mean'] = np.mean((depth[depth_peaks]))

        return_dic['bot_gradient'] = gradient_bot
        features_dic['bot_gradient_mean'] = np.mean(gradient_bot)
        features_dic['bot_gradient_std'] = np.std(gradient_bot)
        features_dic['bot_gradient_max'] = np.max(gradient_bot)
        features_dic['bot_gradient_min'] = np.min(gradient_bot)

        gradient_peaks, g_peaks_properties = ss.find_peaks(gradient_bot,width=1,distance=1,threshold=0.003)
        gradient_minima, g_min_peaks_properties =  ss.find_peaks(-gradient_bot,width=1,distance=1,threshold=0.003)
        gradient_peaks  = np.concatenate([gradient_peaks,gradient_minima]) +2*delta
        
        #return_dic['bot_gradient_peaks'] = gradient_peaks
        features_dic['bot_gradient_peaks_num'] = len(gradient_peaks)
        if features_dic['bot_gradient_peaks_num'] ==0:
            features_dic['bot_gradient_peaks_mean'] = 0
            return_dic["stair"] = None
        else:
            return_dic["stair"] = [depth[np.argmax(gradient_bot)+2*delta],np.argmax(gradient_bot)+2*delta]
            features_dic['bot_gradient_peaks_mean'] = np.mean(np.abs(gradient[gradient_peaks]))



        return features_dic, return_dic



def main():
    extractor = FeatLineExtract()
    extractor.extract_dir('/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-12-12-15-24-09/plots/feature')

if __name__ == '__main__':
    main()

