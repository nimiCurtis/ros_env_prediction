# imports
from typing import Union
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import bag reader and processor modules
from bag_reader.bag_reader import BagReader
from bag_processor.bag_processor import DepthHandler
dp = DepthHandler()

class StairDetector:
    """_summary_
        """    

    def __init__(self):
        """_summary_
            """        
        pass

    def detect(self,img:np.ndarray,depth:np.ndarray,vis:bool=True)->list:
        """This function detect stairs lines using and image and depth values

            Args:
                img (np.ndarray): image matrice
                depth (np.ndarray): depth matrice
                vis (bool, optional): visualise relevant images for debug. Defaults to True.

            Returns:
                list: list of the detected stairs lines
            """        


        # init params and vars
        stairs_lines = []

        # init thresholds
        threshold_sobel = 100     

        # pre-process the image
        blured = cv2.GaussianBlur(img,(11,11),0,0)        # get blured img
        #https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html
        laplacian = cv2.Laplacian(blured,cv2.CV_64F)      # get laplacian img
        sobelx = cv2.Sobel(blured,cv2.CV_64F,1,0,ksize=5) # get sobelx img
        sobely = cv2.Sobel(blured,cv2.CV_64F,0,1,ksize=5) # get sobely img
        
        # normalized sobel matrice
        # shift the matrice by the min val
        sobely_shifted = sobely - np.min(sobely)
        # scaled the matrice to fall between 0-255
        sobely_scaled = (sobely_shifted/np.max(sobely_shifted))*255
        # convert matrice to uint8
        sobely_u8 = sobely_scaled.astype("uint8")
        sobely_u8[sobely_u8<threshold_sobel] = 0

        # sobely_abs = np.abs(sobely)
        # sobely0 = sobely_abs + np.abs(sobely_abs.min())
        # sobely0 = (sobely0/sobely0.max())*255
        # sobely0 = sobely0.astype("uint8")
        # sobely0[sobely0<threshold_sobel] = 0 
        
        # apply canny edge detection
        edges = cv2.Canny(sobely_u8,100 ,250,apertureSize = 3)
        
        # apply Houghline detector
        minLineLength = 90
        maxLineGap = 10  
        lines = cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength,maxLineGap)
        
        # eliminate irelevant lines
        if lines is not None:           
            for line in lines:
                for x1,y1,x2,y2 in line:            
                    if x1 != x2 and ((y1 > 10 and y2 > 10)and (y1 < 240 and y2 < 240)):                   
                        m = (y1-y2)/(x1-x2)                    
                        if np.rad2deg(np.arctan(m))<20 and np.rad2deg(np.arctan(m))>-20 and depth[y1,x1]>300.0 and depth[y2,x2]> 300.0 and depth[y1,x1]!=np.inf :
                            stairs_lines.append(line) 
        
        # visualize relevant images
        if vis:
            self.vis({"blured":blured,"edges":edges,"sobely":sobely})

        return stairs_lines

    def vis(self,**kwargs):
        """This function visualize relevant images for debug
            """        

        for image in kwargs.keys():
            cv2.imshow(image,kwargs[image])
        
class NormalEstimation: # will continue in future work

    def __init__(self):
        pass

    def estimate(self,img_depth,depth, vis=True):
        # if vis:
        #     self.vis(normal)
        # return normal
        pass
    
    def vis(self,normal):
        cv2.imshow("normal",normal)

class AlgoRunner:

    def __init__(self,bag_obj:BagReader):
        self.bag_obj = bag_obj
        
        # init detectors/estimators
        self.stair_detector = StairDetector()
        self.normal_estimator = NormalEstimation()
        
        # set thresholds
        self.static_thresholds = [0.08,0.95]    
        self.dynamic_thresholds = {} 
        
    def __len__(self): ## need to change 
        return len(self.bag_obj.depth_df)

    def get_current_step(self, step:int)->dict:
        """This function insert input data into a dictionary

            Args:
                step (int): iteratiokn number

            Returns:
                dict: input dict
            """        
        
        in_data = {}
        in_data["depth"] = np.load(self.bag_obj.depth_df.np_path[step])
        in_data["depth_img"] = cv2.imread(self.bag_obj.depth_df.frame_path[step])
        
        return in_data

    def crop_regions(self,img,h_grid, w_grid): ## need to change function

        img[:,:w_grid[1]] = 0
        img[:,w_grid[3 ]:] = 0
        #img[h_grid[2]:,:] = 0          

        return img

    def is_SS(self,std_grid:np.ndarray,mean_grid:np.ndarray)->bool:
        """This function determine wether intent state is SS

            Args:
                std_grid (np.ndarray): regions stds
                mean_grid (np.ndarray): regions means

            Returns:
                bool: True if SS
            """        

        # apply logic
        if((std_grid[1,1]<self.static_thresholds[0])
            and(std_grid[1,2]<self.static_thresholds[0])
            and(std_grid[2,1]<self.static_thresholds[1])
            and(std_grid[2,2]<self.static_thresholds[1])):
            
            # update dynamic thresholds
            self.dynamic_thresholds["sa"] = [(mean_grid[0,j] -2.5*std_grid[0,j]) for j in range(std_grid.shape[1])]
            self.dynamic_thresholds["sd"] = [(mean_grid[0,j] +2.5*std_grid[0,j]) for j in range(std_grid.shape[1])]

            return True
        
        else:
            return False

    def is_GLW(self,mean_grid:np.ndarray,stairs_lines:list)->bool:
        """This function determine wether intent state is GLW 

            Args:
                mean_grid (np.ndarray): regions means
                stairs_lines (list): list of detected staires lines

            Returns:
                bool: True if GLW
            """        
        # apply logic
        if((((self.dynamic_thresholds["sa"][1]<mean_grid[0,1])and(mean_grid[0,1]<self.dynamic_thresholds["sd"][1]))
            or
            ((self.dynamic_thresholds["sa"][2]<mean_grid[0,2])and(mean_grid[0,2]<self.dynamic_thresholds["sd"][2])))
            and(len(stairs_lines)==0)):
            
            return True
        
        else:
            return False

    def is_SA(self,mean_grid:np.ndarray,stairs_lines:list)->bool:
        """This function determine wether intent state is SA

            Args:
                mean_grid (np.ndarray): regions means
                stairs_lines (list): list of detected staires lines

            Returns:
                bool: True if SA
            """        

        # apply logic
        if((mean_grid[0,1]<=self.dynamic_thresholds["sa"][1])
            and(mean_grid[0,2]<=self.dynamic_thresholds["sa"][2])
            and(len(stairs_lines)!=0)):

            return True
        
        else:
            return False
    
    def is_SD(self,mean_grid:np.ndarray,stairs_lines:list)->bool:
        """This function determine wether intent state is SD

            Args:
                mean_grid (np.ndarray): regions means
                stairs_lines (list): list of detected staires lines

            Returns:
                bool: True if SD
            """

        # apply logic
        if((mean_grid[0,1]>=self.dynamic_thresholds["sd"][1])
            and(mean_grid[0,2]>=self.dynamic_thresholds["sd"][2])
            and(len(stairs_lines)!=0)):

            return True
        
        else:
            return False

    def intent_recognizer(self,out_data:dict)->int:
        """This function determine the intent state using state machine logic

            Args:
                out_data (dict): output dictionary contaians the relevant features of the current step

            Returns:
                int: integer representing the state. 0:SS | 1:GLW | 2:SA | 3:SD | 10:Dynamic
            """        

        state = ""
        # extarct features of current step
        mean_grid, std_grid, lines = out_data["mean"], out_data["std"],out_data["lines"] 
        
        # apply intent recognition system acording to its logic
        if self.is_SS(std_grid,mean_grid):
            state = 0

        else:
            
            if self.is_GLW(mean_grid,lines):
                state = 1
            
            elif self.is_SA(mean_grid,lines):
                state = 2
            
            elif self.is_SD(mean_grid,lines):
                state = 3

            else:
                state = 10

        return state

    def run(self):
        """This function run the main algorithem of the intention recognition system
            """        
        
        # init buffer for saving data
        algo_buffer = []
        
        # iterating the frames
        for step in range(len(self)):
            # set input/output dictionaries
            out_data = {}
            in_data = self.get_current_step(step)
            
            # copy imgs and depth data
            img = in_data["depth_img"].copy()
            depth =  in_data["depth"].copy()

            # split depth 
            depth_grid, h_grid, w_grid = dp.split_to_regions(depth)
 
            image_cropped = self.crop_regions(img, h_grid, w_grid) ## need to change cropping function

            # extract features
            mean_grid = dp.get_regions_mean(depth_grid)    
            std_grid = dp.get_regions_std(depth_grid)
            
            # detect staires lines
            lines = self.stair_detector.detect(img,depth, vis= True)
            
            # update output dictionary and apply intent recognition system
            out_data["lines"], out_data["mean"], out_data["std"] = lines, mean_grid, std_grid
            out_data["intent"] = self.intent_recognizer(out_data)

            # visualize output
            self.vis_step(in_data,out_data)
            
            # update buffer
            algo_buffer.append(out_data)

        # save output data of this run
        self.save_runner(algo_buffer)

    def save_runner(self,algo_buffer):
        pass

    def vis_step(self,in_data:dict,out_data:dict):
        """This function visualize the output of the algorithem

        Args:
            in_data (dict): input dictionary
            out_data (dict): output dictionary

        Raises:
            Exception: _description_
        """                

        # printing intentiokn state
        print(out_data["intent"])

        # plot staires lines 
        if out_data["lines"] is not None:           
            for line in out_data["lines"]:        
                for x1,y1,x2,y2 in line:            
                    cv2.line(in_data["depth_img"],(x1,y1),(x2,y2),(0,255,0),2)

        cv2.imshow("rgb", in_data["depth_img"])

        # 'q' key to stop
        if cv2.waitKey(33) == ord('q'): 
            cv2.destroyAllWindows()   
            raise Exception()
        else:
            cv2.waitKey(0)

def main():
    bag_obj = BagReader('/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-11-08-10-07-30.bag')
    bag_obj.get_data()
    algo_runner = AlgoRunner(bag_obj)
    algo_runner.run()

if __name__ == "__main__":
    main()