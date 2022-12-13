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


# import bag reader and processor modules
from bag_reader.bag_reader import BagReader
from bag_processor import DepthHandler, ImageHandler
from feature_line_extractor.feature_line_extractor import FeatLineExtract

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
        self.enable = cfg.enable
        
        self.max_line = 0
        #self.line_detect = cv2.ximgproc.createFastLineDetector()

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
                blured = ih.GaussianBlur(img, gauss_config)
            elif b_typ == 1:
                blured = ih.GaborFilter(img, gabor_config)
            elif b_typ == 2:
                blured = ih.BilateralFilter(img, bilateral_config)
            img = blured

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
        elim_top_bot = eliminate_config.top_bottom
        elim_sides = eliminate_config.sides
        elim_theta = eliminate_config.theta
        elim_depth = eliminate_config.depth

        if lines is not None:
            #lines = self.small_edges_eliminate(lines)           
            for line in lines:
                for x1,y1,x2,y2 in line:            
                    if (x1 != x2) and ((y1 > elim_top_bot[0] and y2 > elim_top_bot[0]) and (y1 < elim_top_bot[1] and y2 < elim_top_bot[1])) and (x1 > elim_sides[1] and x2>elim_sides[1]) and (x1 < elim_sides[0] and x2 < elim_sides[0]):                   
                        m = (y1-y2)/(x1-x2)                    
                        if np.rad2deg(np.arctan(m))<elim_theta and np.rad2deg(np.arctan(m))>-elim_theta and depth[y1,x1]>elim_depth[0] and depth[y1,x1] < elim_depth[1] and depth[y2,x2]> elim_depth[0] and depth[y1,x1]!=np.inf :
                            stairs_lines.append(line[0]) 
        
        
        
        stairs_lines = self.link_lines2(stairs_lines)

        if len(stairs_lines)>0:
            stairs_lines = self.small_edges_eliminate(stairs_lines)

        # visualize relevant images     
        if vis:
            self.vis(blur=blured,
                        canny=canny,
                        sobel=sobeld)

        return stairs_lines

    def link_lines1(self, lines:list, pgap:int=15)->list:
        """Linking close lines

        Args:
            lines (list): input lines list
            pgap (int, optional): threshold gap. Defaults to 15.

        Returns:
            list: output lines list with linked lines
        """        

        # change logic on the next dev session
        link_lines = []
        
        for i in range(len(lines)-1):

            xl_i,yl_i,xr_i,yr_i = lines[i]
            for j in range(i+1,len(lines)): 
                xl_j,yl_j,xr_j,yr_j = lines[j]
                
                if xl_j>xr_i:
                    if (np.abs(yl_j-yr_i)<=pgap) and np.abs((xl_j-xr_i))<=pgap:
                        xr_i,yr_i = xr_j,yr_j
                
                elif xr_j<xl_i:
                    if (np.abs(yr_j-yl_i)<=pgap) and np.abs((xr_j-xl_i))<=pgap:
                        xl_i,yl_i = xl_j,yl_j
                        

            
            link_lines.append(np.array((xl_i,yl_i,xr_i,yr_i)))

        return link_lines

    def link_lines2(self,lines:list)->list:
        link_lines = []
        lines0 = lines
        while(len(lines0)>0):
            xl_i,yl_i,xr_i,yr_i = lines0.pop(0)
            ym = int(np.mean([yl_i,yr_i]))
            m_i = np.rad2deg(np.arctan((yl_i-yr_i)/(xl_i-xr_i)))
            
            if len(lines0)>0:
                nlines = np.array(lines0[:]) 
                nlines_candidates = nlines[(nlines[:,1]>ym-5)&(nlines[:,3]>ym-5)&(nlines[:,1]<ym+5)&(nlines[:,3]<ym+5)]
                if len(nlines_candidates>0):
                    m_j = np.rad2deg(np.arctan((nlines_candidates[:,1]-nlines_candidates[:,3])/(nlines_candidates[:,0]-nlines_candidates[:,2]))) 
                    nlines_candidates = nlines_candidates[(m_j<m_i+3)&(m_j>m_i-3)]
                    if len(nlines_candidates>0):
                        xl = min(nlines_candidates[:,0].min(),xl_i)
                        xr = max(nlines_candidates[:,2].max(),xr_i)
                        if xl==xl_i:yl = yl_i
                        else:yl = nlines_candidates[nlines_candidates[:,0]==xl][:,1][0]
                        if xr==xr_i:yr = yr_i
                        else:yr = nlines_candidates[nlines_candidates[:,2]==xr][:,3][0]
                    
                        link_lines.append(np.array((xl,yl,xr,yr)))
                    
                    else:
                        link_lines.append(np.array((xl_i,yl_i,xr_i,yr_i)))
                else:
                    link_lines.append(np.array((xl_i,yl_i,xr_i,yr_i)))
            else:
                link_lines.append(np.array((xl_i,yl_i,xr_i,yr_i)))
            
            lines0 = [elem for elem in lines0 if elem not in nlines_candidates]
        
        if len(link_lines)>0:
            a=1
            
        return link_lines


    def small_edges_eliminate(self, lines:list)->np.ndarray:
        """Eliminate small edges assuming they are noise

        Args:
            lines (list): input lines list

        Returns:
            np.ndarray: output lines list with out noise
        """        

        nlines = np.array(lines)
        p1 = nlines[:,:2]
        p2 = nlines[:,2:]
        dist = np.linalg.norm(p1-p2,axis=1)
        max_length = np.max(dist)
        self.max_line = max(max_length,self.max_line)
        line_thresh = self.max_line/5 
        eliminate = nlines[dist>line_thresh]

        return eliminate

    def feature_line_filter(self,depth_line:np.ndarray)->np.ndarray:
        fdepth_line = depth_line
        j_flag=False
        if fdepth_line[0]==0:fdepth_line[0]= fdepth_line[fdepth_line!=0][0]
        if fdepth_line[-1]==0:fdepth_line[-1]= fdepth_line[fdepth_line!=0][-1]

        for i in range(len(fdepth_line-1)):
            if fdepth_line[i]==0:
                if not j_flag:
                    for j in range(i+1,len(fdepth_line)):
                        if fdepth_line[j]!=0:
                            j_flag=True
                            fdepth_line[i]=np.mean([fdepth_line[i-1],fdepth_line[j]])
                            break
                else:
                    fdepth_line[i]=np.mean([fdepth_line[i-1],fdepth_line[j]])
            else:
                if j_flag:
                    j_flag=False
        
        return fdepth_line

    def find_stair(self,depth_line:np.ndarray)->np.ndarray:
        subtracted = np.subtract(depth_line[1:],depth_line[:-1])
        subtracted = np.concatenate([[0],subtracted])
        subtracted_abs = np.abs(subtracted)
        id_max = np.argmax(subtracted_abs)
        return depth_line[id_max], id_max

    def vis(self,**kwargs:dict):

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
        self._bag_obj = bag_obj 
        self._dfs = self._bag_obj.get_dfs()
        
        #init algo config
        self.intent_config = cfg.AlgoRunner.IntentRecognition
        self._save_run = cfg.AlgoRunner.save_run

        self._vid_config = cfg.AlgoRunner.video
        self._plots_config = cfg.AlgoRunner.plots

        # init detectors/estimators
        self.stair_detector = StairDetector(cfg.StairDetector)
        self.normal_estimator = NormalEstimation()
        self.feature_line_extractor = FeatLineExtract()

        # set thresholds
        self.static_thresholds = self.intent_config.static_thresholds    
        self.dynamic_thresholds = {} 
        
        if cfg.AlgoRunner.run_from is not None:
            self.start_step = cfg.AlgoRunner.run_from
        else:
            self.start_step = 0

    def __len__(self): ## need to change 
        return len(self._dfs["depth"])

    def get_current_step(self, step:int)->dict:
        """This function insert input data into a dictionary

            Args:
                step (int): iteration number

            Returns:
                dict: input dict
            """        

        in_data = {}
        in_data["depth"] = np.load(self._dfs["depth"].np_path[step])
        in_data["depth_img"] = cv2.imread(self._dfs["depth"].frame_path[step])

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

        if self._plots_config.save_mode:
            if not os.path.exists(self._bag_obj.bag_read.datafolder+"/plots"):
                os.mkdir(self._bag_obj.bag_read.datafolder+"/plots")
                if not os.path.exists(self._bag_obj.bag_read.datafolder+"/plots/feature"):
                    os.mkdir(self._bag_obj.bag_read.datafolder+"/plots/feature")
            else:
                if not os.path.exists(self._bag_obj.bag_read.datafolder+"/plots/feature"):
                    os.mkdir(self._bag_obj.bag_read.datafolder+"/plots/feature")
        
        # iterating the frames

        for step in range(self.start_step,len(self)):
            # set input/output dictionaries
            out_data = {}
            in_data = self.get_current_step(step)

            # copy imgs and depth data
            img = in_data["depth_img"].copy()
            depth =  dp.mm2meter(in_data["depth"].copy()) ## change in release to be depend on config

            # split depth 
            depth_grid, h_grid, w_grid = dp.split_to_regions(depth)

            # extract features
            mean_grid = dp.get_regions_mean(depth_grid)    
            std_grid = dp.get_regions_std(depth_grid)

            # detect staires lines
            if self.stair_detector.enable:
                lines = self.stair_detector.detect(img, depth, vis=self._vid_config.debug)
                if len(lines)>0:
                    #d = ss.medfilt2d(depth.copy(),3)
                    feature_line = dp.get_feature_line(lines,depth)
                    feature_line[0] = self.stair_detector.feature_line_filter(feature_line[0])
                    stair_dist = self.stair_detector.find_stair(feature_line[0])
                    
                    out_data["feature_line"], out_data["stair_dist"] = feature_line, stair_dist

                out_data["lines"] = lines
            
            else:
                feature_line = dp.get_feature_line(depth)
                feature_line[0] = self.stair_detector.feature_line_filter(feature_line[0])
                stair_dist = self.stair_detector.find_stair(feature_line[0])
                
                
                out_data["feature_line"], out_data["stair_dist"] = feature_line, stair_dist

            # update output dictionary and apply intent recognition system
            out_data["mean"], out_data["std"] =  mean_grid, std_grid
            #out_data["intent"] = self.intent_recognizer(out_data)

            # update buffer 
            algo_buffer.append(out_data)         
            frame_buffer.append(in_data["depth_img"])
            

            # visualize output
            self.vis_step(step,in_data,out_data)
            
            if self._plots_config.save_mode:
                    self.plot_step(step,in_data,out_data,
                                save=True,
                                pltshow=self._plots_config.debug.online,
                                imshow=False)

            else:
                if self._plots_config.debug.online or self._plots_config.debug.offline:
                    self.plot_step(step,in_data,out_data,
                                    save=False,
                                    pltshow=self._plots_config.debug.online,
                                    imshow=self._plots_config.debug.offline)
                else:
                    pass


            # 'q' key to stop
            if cv2.waitKey(10) & 0xFF == ord('q'): 
                    cv2.destroyAllWindows()   
                    raise Exception()
            else:
                cv2.waitKey(1) 

        # save output data of this run
        if self._save_run:
            self.save_runner(algo_buffer)

        if self._vid_config.save:
            print("[Info] Saving video..")    
            ih.write_video(self._bag_obj.bag_read.datafolder,self._vid_config.name, frame_buffer, 10)
            print("[Info] Video saved")

    def save_runner(self,algo_buffer):
        pass

    def vis_step(self,step,in_data:dict,out_data:dict):
        """This function visualize the output of the algorithem

        Args:
            in_data (dict): input dictionary
            out_data (dict): output dictionary

        Raises:
            Exception: _description_
        """                
        
        # printing intentiokn state

        #print(out_data["intent"])

        # plot staires lines
        pt1 = (out_data["feature_line"][1][0][0],out_data["feature_line"][1][0][1])
        pt2 = (out_data["feature_line"][1][-1][0],out_data["feature_line"][1][-1][1])
        cv2.line(in_data["depth_img"],pt1,pt2,(255,0,0),1)
        text = f"Distance to POI: {out_data['stair_dist'][0]:.3f},frame: {step}"
        y0, dy = 20, 15
        for i, line in enumerate(text.split(',')):
            y = y0 + i*dy
            cv2.putText(in_data["depth_img"],line,org = (20, y ),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4,color=(0,0,255),thickness=1)

        cv2.circle(in_data["depth_img"],out_data["feature_line"][1][out_data["stair_dist"][1]],radius=5,color=(0,0,255))


        if self.stair_detector.enable:
            if out_data["lines"] is not None:           
                for line in out_data["lines"]:        
                    x1,y1,x2,y2 = line            
                    cv2.line(in_data["depth_img"],(x1,y1),(x2,y2),(0,255,0),2)

        # re-drawing the figure
        cv2.imshow("rgb", in_data["depth_img"])

    def plot_step(self,step,in_data,out_data,save,pltshow,imshow):
        file_path = self._bag_obj.bag_read.datafolder+f"/plots/plot_{step}.png"
        np_path = self._bag_obj.bag_read.datafolder+f"/plots/feature/plot_{step}.npy"
        if save or pltshow:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            # make a little extra space between the subplots
            fig.subplots_adjust(hspace=0.5)
            px_indexes = out_data["feature_line"][1][:,1]
            depth_line = out_data["feature_line"][0][::]

            features_dic = self.feature_line_extractor.extract(depth_line)
            ax1.plot(px_indexes,depth_line,'b')
            #ax1.plot(out_data["stair_dist"][1],out_data["stair_dist"][0],marker="o", markersize=8, color="red")
            ax1.plot(features_dic['depth_peaks'],depth_line[features_dic["depth_peaks"]],"x")

            ax1.set_title(f"depth vs pixel index | frame: {step}")
            
            
            ax2.plot(px_indexes,features_dic['subtracted'])
            ax2.plot(features_dic['subtracted_peaks'],features_dic['subtracted'][features_dic['subtracted_peaks']],"x")
            ax2.set_title(f"depth diff vs pixel index")

            if save:
                np.save(np_path,np.array((depth_line)))
                #plt.savefig(file_path)
            
            if pltshow:
                plt.show()
            
            plt.close(fig)
        
        if imshow:
            img = cv2.imread(file_path)
            cv2.imshow("plot",img)



# Use hydra for configuration managing
@hydra.main(version_base=None, config_path="../../config", config_name = "algo")
def main(cfg):
    bag_obj = BagReader()
    bag_obj.bag = '/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-11-08-10-13-11.bag'
    algo_runner = AlgoRunner(bag_obj,cfg)
    algo_runner.run()

if __name__ == "__main__":
    main() 