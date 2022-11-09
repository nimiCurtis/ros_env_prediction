# Import libraries
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

class ImageHandler():
    """_summary_
        """    
    
    def __init__(self):
        """_summary_
            """        
        
        self.grid_shape = (3,4)           # shape of the image grid

    def split_to_regions(self, img):
        """This function split generic image matrice to regions

            Args:
                img (numpy array): generic image matrice

            Returns:
                regions (numpy array): splited matrice by defined regions
                h_grid (numpy array): hight regions 
                w_grid (numpy array): width regions
            """        
        # split hight and width indexes by requierd grid shape
        h_grid = np.linspace(0,img.shape[0],self.grid_shape[0]+1,dtype=np.uint16) 
        w_grid = np.linspace(0,img.shape[1],self.grid_shape[1]+1,dtype=np.uint16)

        # create array of the img regions
        regions_arr = [img[x:x+int(h_grid[1]),y:y+int(w_grid[1])] for x in range(0,img.shape[0],int(h_grid[1])) for y in range(0,img.shape[1],int(w_grid[1]))]
        regions = np.array(regions_arr)

        return regions, h_grid, w_grid

    def get_regions_mean(self,img):
        """This function calculate image regions means

            Args:
                img (numpy array): generic image matrice

            Returns:
                mean_arr (numpy array): means of image splited regions
            """

        arr_len = self.grid_shape[0]*self.grid_shape[1] 
        mean_arr = np.array([np.nanmean(img[x]) for x in range(arr_len)]).reshape(self.grid_shape)

        return mean_arr
    
    def get_regions_std(self,img):
        """This function calculate image regions std

            Args:
                img (numpy array): generic image matrice

            Returns:
                std_arr (numpy array): standard deviation of image splited regions
            """    

        arr_len = self.grid_shape[0]*self.grid_shape[1]
        std_arr = np.array([np.nanstd(img[x]) for x in range(arr_len)]).reshape(self.grid_shape)

        return std_arr

    def extract_features_singleImg(self,img):
        """This function extract relevant features from image matrice

            Args:
                img (numpy array): generic image matrice

            Returns:
                mean (numpy array): means array from 'get_regions_mean()'
                std (numpy array): std array from 'get_regions_std()'
            """        
        
        # first split the image
        img_splited = self.split_to_regions(img) 

        # extract featurs
        mean = self.get_regions_mean(img_splited)
        std = self.get_regions_std(img_splited)

        return mean, std
    
    def imshow_with_grid(self,img_path, color=(0, 255, 0), thickness=1):
        """This function show image with the grid 

        Args:
            img_path (string): path of image
            color (tuple, optional): RGB values of the grid lines. Defaults to (0, 255, 0).
            thickness (int, optional): grid lines thickness value. Defaults to 1.
        """        

        img = cv2.imread(img_path)
        h, w, _ = img.shape
        rows, cols = self.grid_shape
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


class DepthHandler(ImageHandler):
    """_summary_

        Args:
            ImageHandler (object): parent class
        """        

    def __init__(self):
        """_summary_
            """        

        super().__init__()

    def mm2meter(self, depth):
        """This function covert depth matrice to meters units

            Args:
                depth (numpy array): depth values in milimeters

            Returns:
                depth (numpy array): depth values in meters
            """        

        return depth/1000

    def meter2mm(self, depth):
        """This function covert depth matrice to mm units

            Args:
                depth (numpy array): depth values in meters

            Returns:
                depth (numpy array): depth values in milimeters
            """ 

        return depth*1000

    def zero_to_nan(self, depth):
        """This function convert zero values to nan values in given depth matrice

        Args:
            depth (numpy array): depth vals matrice

        Returns:
            depth (numpy array): depth matrice with zero values converted to nan
        """        

        depth[depth==0] = np.nan
        return depth

    def extract_depth_features_single(self,depth):
        """This function extract features from a depth matrice
        using the extract_features_singleImg

        Args:
            depth (numpy array): depth vals matrice

        Returns:
            mean (numpy array): means array from 'get_regions_mean()'
            std (numpy array): std array from 'get_regions_std()'
        """        

        depth0 = self.zero_to_nan(depth)
        mean , std = self.extract_features_singleImg(depth0)
        return mean, std

    def extract_depth_features_batch(self,df):
        """This function extract features from given data frame of depth matrices

        Args:
            df (pandas dataframe): data frame contains depth matrices

        Returns:
            mean_arr (list): mean array off depth regions means
            std_arr (list): std array off depth regions std
        """        

        mean_arr = []
        std_arr = []

        for index in range(df.shape[0]):
            depth = np.load(df.np_path[index])
            depth = self.mm2meter(depth)                             # converted until config ready
            mean, std = self.extract_depth_features_single(depth)
            
            mean_arr.append(mean)
            std_arr.append(std)
        
        return mean_arr, std_arr


    def get_depth_features_df(self,df):
        """This function add extracted regions features columns to a given depth images dataframe 

            Args:
                df (pandas dataframe): data frame contains depth matrices

            Returns:
                df (pandas dataframe): modified dataframe to include features columns
            """        

        feat_df = df 
        feat_df["mean"], feat_df["std"] = self.extract_depth_features_batch(df)

        return feat_df

def main(): 
    pass

if __name__ == '__main__':
    main()
