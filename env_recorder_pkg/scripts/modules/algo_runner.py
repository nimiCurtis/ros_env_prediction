# imports
import hydra
from omegaconf import DictConfig
from typing import Union
import cv2
import numpy as np
import scipy.signal as ss
import pandas as pd
import matplotlib.pyplot as plt

# import bag reader and processor modules
from bag_reader.bag_reader import BagReader
from bag_processor.bag_processor import DepthHandler, ImageHandler
dp = DepthHandler()
ih = ImageHandler()

class StairDetector:
    """_summary_
        """    

    def __init__(self,cfg : DictConfig):
        """StairDetector constructor

        Args:
            cfg (DictConfig): config dictionary
        """        

        # init config
        self.cfg = cfg
        
        self.max_line = 0

    def detect(self,img:np.ndarray, depth:np.ndarray,vis:bool=True)->list:
        """This function detect stairs lines using and image and depth values

            Args:
                img (np.ndarray): image matrice
                depth (np.ndarray): depth matrice
                vis (bool, optional): visualise relevant images for debug. Defaults to True.

            Returns:
                list: list of the detected stairs lines
            """        
        # init configs
        sobel_config = self.cfg.Sobel
        gauss_config = self.cfg.GaussianBlur
        canny_config = self.cfg.Canny
        hough_config = self.cfg.HoughLinesP
        gabor_config = self.cfg.Gabor
        bilateral_config = self.cfg.Bilateral
        eliminate_config = self.cfg.Eliminate

        # init params and vars
        blured, sobeld, canny = None, None, None
        stairs_lines = []

        # pre-process the image
        # convert img to gray 
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # get blured img
        if self.cfg.blur.enable:
            b_typ = self.cfg.blur.type
            if b_typ == 0:
                #blured = ss.medfilt2d(img.copy(),5)
                blured = ih.GaussianBlur(img.copy(), gauss_config)
            elif b_typ == 1:
                blured = ih.GaborFilter(img.copy(), gabor_config)
            elif b_typ == 2:
                blured = ih.BilateralFilter(img.copy(), bilateral_config)
            img = blured

        #https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html
        # apply sobel functions
        if self.cfg.Sobel.enable:
            sobeld = ih.Sobel(img.copy(), sobel_config)
            #sobeld = ih.GaussianBlur(sobeld.copy(), gauss_config)
            img = sobeld
            

        # apply canny edge detection
        if self.cfg.Canny.enable:
            canny = ih.Canny(img, canny_config)
            img = canny

        # apply Houghline detector
        lines = ih.Hough(img, hough_config)

        # eliminate candidats
        eliminate_top = eliminate_config.top
        eliminate_bottom = eliminate_config.bottom
        eliminate_right = eliminate_config.right
        eliminate_left = eliminate_config.left
        eliminate_theta = eliminate_config.theta
        eliminate_depth = eliminate_config.depth

        if lines is not None:
            #lines = self.small_edges_eliminate(lines)           
            for line in lines:
                for x1,y1,x2,y2 in line:            
                    if (x1 != x2) and ((y1 > eliminate_top and y2 > eliminate_top) and (y1 < eliminate_bottom and y2 < eliminate_bottom)) and (x1 > eliminate_left and x2>eliminate_left) and (x1 < eliminate_right and x2<eliminate_right):                   
                        m = (y1-y2)/(x1-x2)                    
                        if np.rad2deg(np.arctan(m))<eliminate_theta and np.rad2deg(np.arctan(m))>-eliminate_theta and depth[y1,x1]>eliminate_depth and depth[y1,x1] < 1.4 and depth[y2,x2]> eliminate_depth and depth[y1,x1]!=np.inf :
                            stairs_lines.append(line) 
        
        stairs_lines = self.link_lines(stairs_lines)
        
        
        if len(stairs_lines)>0:
            stairs_lines = self.small_edges_eliminate(depth,stairs_lines).tolist()
            

        # visualize relevant images     
        if vis:
            self.vis(blur=blured,
                        canny=canny,
                        sobel=sobeld)

        return stairs_lines

    def link_lines(self, lines:list, pgap:int=15):
        link_lines = []
        
        for i in range(len(lines)-1):

            xl_i,yl_i,xr_i,yr_i = lines[i][0]
            for j in range(i+1,len(lines)): 
                xl_j,yl_j,xr_j,yr_j = lines[j][0]
                
                if xl_j>xr_i:
                    if (np.abs(yl_j-yr_i)<=pgap) and np.abs((xl_j-xr_i))<=pgap:
                        xr_i,yr_i = xr_j,yr_j
                
                elif xr_j<xl_i:
                    if (np.abs(yr_j-yl_i)<=pgap) and np.abs((xr_j-xl_i))<=pgap:
                        xl_i,yl_i = xl_j,yl_j
                        

            
            link_lines.append([[xl_i,yl_i,xr_i,yr_i]])


        return link_lines

    def get_feature_line(self,lines:list,depth):
        nlines = np.array(lines)
        max_line = self.get_max_line(nlines)
        xmid,ymid = self.get_mid(max_line)
        feature_vals = depth[:,xmid]
        feature_index = np.ones((depth.shape[0],2),dtype=np.int16)*xmid
        feature_index[:,1] = np.arange(0,depth.shape[0])

        return [feature_vals,feature_index, (xmid,ymid)]

    def get_feature_region(self,lines:list,depth):
        nlines = np.array(lines)
        max_line = self.get_max_line(nlines)
        xmid,ymid = self.get_mid(max_line)

        feature_vals = depth[:,xmid-5:xmid+5]
        feature_vals = feature_vals.mean(axis=1)
        feature_vals = feature_vals[feature_vals!=0]
        feature_index = np.ones((feature_vals.shape[0],2),dtype=np.int16)*xmid
        feature_index[:,1] = np.arange(0,feature_vals.shape[0])
        
        return [feature_vals,feature_index, (xmid,ymid)]

    def small_edges_eliminate(self,depth, lines:list):
        nlines = np.array(lines)
        p1 = nlines[:,0][:,:2]
        p2 = nlines[:,0][:,2:]
        dist = np.linalg.norm(p1-p2,axis=1)
        max_length = np.max(dist)
        self.max_line = max(max_length,self.max_line)
        line_thresh = self.max_line/4 
        eliminate = nlines[dist>line_thresh]

        return eliminate

    def get_max_line(self,lines:np.ndarray):
        p1 = lines[:,0][:,:2]
        p2 = lines[:,0][:,2:]
        dist = np.linalg.norm(p1-p2,axis=1)
        max_length = np.max(dist)
        max_line= lines[dist==max_length]
        return max_line 
    
    def get_mid(self,line):
        x1,y1,x2,y2 = line[0][0]
        return int((x2+x1)/2),int((y2+y1)/2)

    def vis(self,**kwargs):
        """This function visualize relevant images for debug
            """        

        for image in kwargs.keys():
            if kwargs[image] is not None:
                cv2.imshow(image, kwargs[image])
        
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

    def __init__(self,bag_obj:BagReader,cfg:DictConfig ):
        """AlgoRunner constructor

        Args:
            bag_obj (BagReader): bag object
            cfg (DictConfig): config dictionary
        """        

        # init bag object
        self.bag_obj = bag_obj
        
        #init algo config
        self.intent_config = cfg.AlgoRunner.IntentRecognition
        self._save_run = cfg.AlgoRunner.save_run
        self._save_vid = cfg.AlgoRunner.save_vid
        self._viz_plot = cfg.AlgoRunner.viz_plot
        self._vid_name = cfg.AlgoRunner.vid_name

        # init detectors/estimators
        self.stair_detector = StairDetector(cfg.StairDetector)
        self.normal_estimator = NormalEstimation()
        
        # set thresholds
        self.static_thresholds = self.intent_config.static_thresholds    
        self.dynamic_thresholds = {} 
        
    def __len__(self): ## need to change 
        return len(self.bag_obj.dfs["depth"])

    def get_current_step(self, step:int)->dict:
        """This function insert input data into a dictionary

            Args:
                step (int): iteratiokn number

            Returns:
                dict: input dict
            """        
        
        in_data = {}
        in_data["depth"] = np.load(self.bag_obj.dfs["depth"].np_path[step])
        in_data["depth_img"] = cv2.imread(self.bag_obj.dfs["rgb"].frame_path[step])
        
        return in_data

    def crop_regions(self,img,h_grid, w_grid): ## need to change function

        img_cropped = img[h_grid[1]:h_grid[2],w_grid[1]:w_grid[3]]

        return img_cropped

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
            sa_cof = self.intent_config.dynamic_thresholds.sa 
            sd_cof = self.intent_config.dynamic_thresholds.sd

            self.dynamic_thresholds["sa"] = [(mean_grid[0,j] +sa_cof*std_grid[0,j]) for j in range(std_grid.shape[1])]
            self.dynamic_thresholds["sd"] = [(mean_grid[0,j] +sd_cof*std_grid[0,j]) for j in range(std_grid.shape[1])]

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
        frame_buffer = []

        # iterating the frames
        for step in range(len(self)):
            # set input/output dictionaries
            out_data = {}
            in_data = self.get_current_step(step)

            # copy imgs and depth data
            img = in_data["depth_img"].copy()
            depth =  dp.mm2meter(in_data["depth"].copy()) ## change in releas to be depend on config

            # split depth 
            depth_grid, h_grid, w_grid = dp.split_to_regions(depth)

            # depth_cropped = self.crop_regions(depth, h_grid, w_grid) 
            # image_cropped = self.crop_regions(img, h_grid, w_grid) ## need to change cropping function

            # extract features
            mean_grid = dp.get_regions_mean(depth_grid)    
            std_grid = dp.get_regions_std(depth_grid)

            # detect staires lines
            lines = self.stair_detector.detect(img, depth, vis=True)
            if len(lines)>0:
                #d = ss.medfilt2d(depth.copy(),3)
                feature_line = self.stair_detector.get_feature_region(lines,depth)
                out_data["feature_line"] = feature_line

            # update output dictionary and apply intent recognition system
            out_data["lines"], out_data["mean"], out_data["std"] = lines, mean_grid, std_grid
            out_data["intent"] = self.intent_recognizer(out_data)

            # update buffer 
            algo_buffer.append(out_data)         
            frame_buffer.append(in_data["depth_img"])

            # visualize output
            self.vis_step(in_data,out_data)
            # re-drawing the figure
            cv2.imshow("rgb", in_data["depth_img"])
            

            # 'q' key to stop
            if cv2.waitKey(10) & 0xFF == ord('q'): 
                    cv2.destroyAllWindows()   
                    raise Exception()
            else:
                cv2.waitKey(1) 
            
            if self._viz_plot:
                plt.show()
            
        # save output data of this run
        if self._save_run:
            self.save_runner(algo_buffer)

        if self._save_vid:
            print("[Info] Saving video..")    
            ih.write_video(self.bag_obj.bag_read.datafolder,self._vid_name, frame_buffer, 10)
            print("[Info] Video saved")

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
        #plt.ylim((0,5))
        if out_data["lines"] is not None:           
            for line in out_data["lines"]:        
                for x1,y1,x2,y2 in line:            
                    cv2.line(in_data["depth_img"],(x1,y1),(x2,y2),(0,255,0),2)
                    pt1 = (out_data["feature_line"][1][0][0],out_data["feature_line"][1][0][1])
                    pt2 = (out_data["feature_line"][1][-1][0],out_data["feature_line"][1][-1][1])
                    cv2.line(in_data["depth_img"],pt1,pt2,(255,0,0),1)
                    plt.plot(out_data["feature_line"][1][:,1],out_data["feature_line"][0][::-1],'b')
                    plt.ylim((0,3))

# Use hydra for configuration managing
@hydra.main(version_base=None, config_path="../../config", config_name = "algo")
def main(cfg):
    bag_obj = BagReader('/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-11-08-10-13-11.bag')
    bag_obj.get_data()
    algo_runner = AlgoRunner(bag_obj,cfg)
    algo_runner.run()

if __name__ == "__main__":
    main() 