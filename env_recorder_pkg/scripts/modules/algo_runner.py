# imports
import os
import hydra
from omegaconf import DictConfig
from typing import Union
import cv2
import numpy as np
import scipy.signal as ss
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from bag_label_tool.label_tool import EnvLabel
# import bag reader and processor modules
from bag_reader.bag_reader import BagReader
from feature_line_extractor.feature_line_extractor import FeatLineExtract
from stair_detector.stair_detector import StairDetector
from image_data_handler.image_data_handler import DepthHandler, ImageHandler
from env_classifier.env_classifier import EnvClassifierPipe
dp = DepthHandler()
ih = ImageHandler()



class AnalyticEnvRecognition:

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
        self.intent_config = cfg.AlgoRunner.AnalyticEnvRecognition
        self._save_run = cfg.AlgoRunner.save_run

        self._vid_config = cfg.AlgoRunner.video

        # init detectors/estimators
        self.stair_detector = StairDetector(cfg.StairDetector)

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
            lines = self.stair_detector.detect(img, depth, vis=self._vid_config.debug)

            # update output dictionary and apply intent recognition system
            out_data["lines"] = lines
            out_data["mean"], out_data["std"] =  mean_grid, std_grid
            out_data["intent"] = self.intent_recognizer(out_data)

            algo_buffer.append(out_data)         
            frame_buffer.append(in_data["depth_img"])
            
            # visualize output
            self.vis_step(step,in_data,out_data)

            # 'q' key to stop
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                    cv2.destroyAllWindows()   
                    raise Exception()
            else:
                cv2.waitKey(10) 
            
            

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
        img = in_data["depth_img"].copy()
        print(out_data["intent"])
        # plot staires lines
        if out_data["lines"] is not None:           
            for line in out_data["lines"]:        
                x1,y1,x2,y2 = line            
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        
        # re-drawing the figure
        cv2.imshow("depth", img)

class SVMEnvRecognition:

    def __init__(self,bag_obj:BagReader,cfg:DictConfig ):

        """AlgoRunner constructor
        Args:
            bag_obj (BagReader): bag object
            cfg (DictConfig): config dictionary
        """        

        # init bag object
        self._bag_obj = bag_obj 
        self._dfs = self._bag_obj.get_dfs()
        self._labels = pd.read_csv(self._bag_obj.bag_read.datafolder+"/feature_line/features.csv")['labels'].to_list() 
        #init algo config
        self._save_run = cfg.AlgoRunner.save_run

        self._vid_config = cfg.AlgoRunner.video
        self._plots_config = cfg.AlgoRunner.plots

        # init detectors/estimators
        self.stair_detector = StairDetector(cfg.StairDetector)
        self.feature_line_extractor = FeatLineExtract()
        self.clf_pipline = EnvClassifierPipe()
        self.clf_pipline.load('best_29-12-2022_20-10-56.joblib')
        #self.transformer = load('/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/models/transformer.joblib')
        
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
        in_data["depth_img"] = cv2.imread(self._dfs["rgb"].frame_path[step])

        return in_data

    def run(self):
        """This function run the main algorithem of the intention recognition system
            """

        # init buffer for saving data
        algo_buffer = []
        frame_buffer = []

        # iterating the frames

        for step in range(self.start_step,len(self)):
            # set input/output dictionaries
            out_data = {}
            in_data = self.get_current_step(step)

            # copy imgs and depth data
            img = in_data["depth_img"].copy()
            depth =  dp.mm2meter(in_data["depth"].copy()) ## change in release to be depend on config

            # detect staires lines
            if self.stair_detector.enable:
                lines = self.stair_detector.detect(img, depth, vis=self._vid_config.debug)
                if len(lines)>0:
                    #d = ss.medfilt2d(depth.copy(),3)
                    feature_line = dp.get_feature_line(lines,depth)
                    feature_line[0] = dp.feature_line_filter(feature_line[0])
                    stair_dist = self.stair_detector.find_stair(feature_line[0])
                    
                    
                else:
                    feature_line = dp.get_feature_line(depth)
                    feature_line[0] = dp.feature_line_filter(feature_line[0])
                
                out_data["feature_line"] = feature_line
                out_data["lines"] = lines
            
            else:
                feature_line = dp.get_feature_line(depth)
                feature_line[0] = dp.feature_line_filter(feature_line[0])
                
            #             X_test = column_transformer.transform(X_test)
            # X_test = pd.DataFrame(data=X_test, columns=column_transformer.get_feature_names_out())
                
                

            features_dic,ret_dic = self.feature_line_extractor.extract(feature_line[0])
            features_input = np.array(list(features_dic.values())).reshape(1,-1)
            #features_input = self.transformer.transform(features_input)
            # predict_env = self.clf.predict(features_input)
            #x = pd.DataFrame(features_dic,index=[0])
            #numerical_cols = x.columns.to_list()
            # Create a transformer object
            

            #x = self.transformer.transform(x)
            
            predict_env = self.clf_pipline.predict(features_input)
            out_data["predict_env"] = predict_env[0]

            # update buffer 
            stair_dist = ret_dic["stair"]
            out_data["feature_line"], out_data["stair_dist"] = feature_line, stair_dist

            algo_buffer.append(out_data)         
            if self._vid_config.save:
                frame = self.vis_step(step,in_data,out_data)
                frame_buffer.append(frame)
            else:
                self.vis_step(step,in_data,out_data)

            
            if self._plots_config.save_mode:
                    self.plot_step(step,ret_dic,out_data,
                                save=True,
                                pltshow=self._plots_config.debug.online,
                                imshow=False)

            else:
                if self._plots_config.debug.online or self._plots_config.debug.offline:
                    self.plot_step(step,ret_dic,out_data,
                                    save=False,
                                    pltshow=self._plots_config.debug.online,
                                    imshow=self._plots_config.debug.offline)
                else:
                    pass
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                cv2.destroyAllWindows()   
                raise Exception()
            else:
                cv2.waitKey(0)

        # save output data of this run
        if self._save_run:
            self.save_runner(algo_buffer)

        # visualize output and save video
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
        
        img = self.feature_line_extractor.draw_feature_line(in_data["depth_img"])
        
        predict = out_data["predict_env"]
        if predict == self._labels[step]:
            tcolor = (0,255,0)
        else:
            tcolor = (0,0,255)

        if predict!=1 and out_data['stair_dist'] is not None:
            text = f"Distance to POI: {out_data['stair_dist'][0]:.3f}[meters],frame: {step},env predict: {EnvLabel(predict).name},env real: {EnvLabel(self._labels[step]).name}"
            y0, dy = 20, 15
            for i, line in enumerate(text.split(',')):
                y = y0 + i*dy
                cv2.putText(img,line,org = (20, y ),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4,color=tcolor,thickness=1)

            cv2.circle(img,out_data["feature_line"][1][out_data["stair_dist"][1]],radius=5,color=(0,0,255))
        else:
            text = f"frame: {step},env predict: {EnvLabel(predict).name},env real: {EnvLabel(self._labels[step]).name}"
            y0, dy = 20, 15
            for i, line in enumerate(text.split(',')):
                y = y0 + i*dy
                cv2.putText(img,line,org = (20, y ),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4,color=tcolor,thickness=1)

        # plot staires lines
        if self.stair_detector.enable:
            if out_data["lines"] is not None:           
                for line in out_data["lines"]:        
                    x1,y1,x2,y2 = line            
                    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

        # re-drawing the figure
        cv2.imshow("depth", img)
        return img


    def plot_step(self,step,ret_dic,out_data,save,pltshow,imshow):
        
        file_path = self._bag_obj.bag_read.datafolder+f"/feature_line/plots/plot_{step}.png"
        dline = out_data['feature_line'][0]
        
        if save or pltshow:
            self.feature_line_extractor.plot_feature_line(step,dline,ret_dic,file_path=file_path,save_plots=save,pltshow=pltshow)

        if imshow:
            img = cv2.imread(file_path)
            cv2.imshow("plot",img)



# Use hydra for configuration managin
@hydra.main(version_base=None, config_path="../../config", config_name = "algo")
def main(cfg):
    bag_obj = BagReader()
    bag_obj.bag = '/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-12-12-15-24-09.bag'
    algo_runner = SVMEnvRecognition(bag_obj,cfg)
    #algo_runner = AnalyticEnvRecognition(bag_obj,cfg)
    algo_runner.run()

if __name__ == "__main__":
    main()
