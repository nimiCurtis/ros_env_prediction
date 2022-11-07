
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

class DepthProcessor():
    
    def __init__(self):
        pass
    
    def zero_to_nan(self, depth_vals_array):
        depth_vals_array[depth_vals_array==0] = np.nan
        return depth_vals_array

    def split_to_regions(self, depth_vals_array,grid_shape=(3,4)):

        h_grid = np.linspace(0,depth_vals_array.shape[0],grid_shape[0]+1,dtype=np.uint16)
        w_grid = np.linspace(0,depth_vals_array.shape[1],grid_shape[1]+1,dtype=np.uint16)

        regions_arr = [depth_vals_array[x:x+int(h_grid[1]),y:y+int(w_grid[1])] for x in range(0,depth_vals_array.shape[0],int(h_grid[1])) for y in range(0,depth_vals_array.shape[1],int(w_grid[1]))]
        regions = np.array(regions_arr)

        return regions, h_grid, w_grid

    def get_regions_mean(self,depth_vals_splited,grid_shape=(3,4)):
        arr_len = grid_shape[0]*grid_shape[1]
        mean_arr = np.array([np.nanmean(depth_vals_splited[x])/1000 for x in range(arr_len)]).reshape(grid_shape)

        return mean_arr
    
    def get_regions_std(self,depth_vals_splited,grid_shape=(3,4)):
        arr_len = grid_shape[0]*grid_shape[1]
        std_arr = np.array([np.nanstd(depth_vals_splited[x])/1000 for x in range(arr_len)]).reshape(grid_shape)

        return std_arr

    def extract_features_singleImg(self,depth_vals,grid_shape=(3,4)):
        depth_vals = self.zero_to_nan(depth_vals)
        depth_vals_splited = self.split_to_regions(depth_vals,grid_shape)
        mean = self.get_regions_mean(depth_vals_splited, grid_shape)
        std = self.get_regions_std(depth_vals_splited,grid_shape)
        return mean, std

    def extract_features_batchImg(self,df):
        mean_arr = []
        std_arr = []
        grid_shape = (3,4)

        for index in range(df.shape[0]):
            depth_vals = np.load(df.np_path[index])
            mean, std = self.extract_features_singleImg(depth_vals,grid_shape)
            
            mean_arr.append(mean)
            std_arr.append(std)
        
        return mean_arr, std_arr
    
    def get_features_df(self,df):

        feat_df = df 
        feat_df["mean"], feat_df["std"] = self.extract_features_batchImg(df)

        return feat_df

    def imshow_with_grid(self,img_path,grid_shape=(3,4), color=(0, 255, 0), thickness=1):
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        rows, cols = grid_shape
        dy, dx = h / rows, w / cols

        # draw vertical lines
        for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
            x = int(round(x))
            cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

        # draw horizontal lines
        for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
            y = int(round(y))
            cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

        plt.imshow(img)
        plt.show()


def main(): 
    pass


if __name__ == '__main__':
    main()
