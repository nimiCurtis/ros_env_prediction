#### TO DO ####
# improve code structure - makes it generic with params
# documentation
# create config for algo runner
# review logic

# improve other code files and documents

# test algo:
#   - with different algo params (stairs detection / grid / crop) 
#   - with different recording params (ROS yaml files of the camera)
#   - with stair detector implementation on rgb/depth/disparity
# create a local records (bag file) on jetson (with no wifi dependency) with better light conditions
# add norm estimation
# video writer - https://www.programcreek.com/python/example/72134/cv2.VideoWriter
#  

########################

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bag_reader.bag_reader import READER
from bag_processor.bag_processor import DepthProcessor
dp = DepthProcessor()

#feat_df = dp.get_features_df(bag_read.depth_df)

class StairDetector:

    def __init__(self):
        pass

    def detect(self,img_depth,depth, vis=True):
        stairs_lines = []
        blured = cv2.GaussianBlur(img_depth,(11,11),0,0)
        edges = cv2.Canny(blured,0,0,apertureSize = 7)
        minLineLength = 0
        maxLineGap = 0
        lines = cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength,maxLineGap)
        
        if lines is not None:           
            for line in lines:
                for x1,y1,x2,y2 in line:            
                    if x1 != x2 and (y1 > 10 and y2 > 10):                   
                        m = (y1-y2)/(x1-x2)                    
                        if np.rad2deg(np.arctan(m))<15 and np.rad2deg(np.arctan(m))>-15 and depth[y1,x1]>100.0 and depth[y2,x2]> 100.0:
                            stairs_lines.append(line)
        if vis:
            self.vis(blured,edges)

        return stairs_lines

    def vis(self,blured, edges):
        cv2.imshow("blured",blured)
        cv2.imshow("edges",edges)


class NormalEstimation:

    def __init__(self):
        pass

    def estimate(self,img_depth,depth, vis=True):
        stairs_lines = []
        threshold_sobel = 60  

        blured = cv2.GaussianBlur(img_depth,(13,13),0,0)
        laplacian = cv2.Laplacian(blured,cv2.CV_64F)
        sobelx = cv2.Sobel(blured,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(blured,cv2.CV_64F,0,1,ksize=5)
        sobely0 = sobely.astype("float32")
        sobely0 = np.abs(sobely0)
        sobely0 = sobely0 + np.abs(sobely0.min())
        sobely0 = (sobely0/sobely0.max())*255
        sobely0 = sobely0.astype("uint8")
        sobely0[sobely0<threshold_sobel] = 0 

        edges = cv2.Canny(sobely0,100 ,255,apertureSize = 3)
        
        minLineLength = 50
        maxLineGap = 5

        lines = cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength,maxLineGap)
        
        if lines is not None:           
            for line in lines:
                for x1,y1,x2,y2 in line:            
                    if x1 != x2 and (y1 > 10 and y2 > 10):                   
                        m = (y1-y2)/(x1-x2)                    
                        if np.rad2deg(np.arctan(m))<15 and np.rad2deg(np.arctan(m))>-15 and depth[y1,x1]>100.0 and depth[y2,x2]> 100.0:
                            stairs_lines.append(line)


        #d_im = d_im.astype("float64")

        # normals = np.array(d_im, dtype="float32")
        # h,w,d = d_im.shape
        # for i in range(1,w-1):
        #     for j in range(1,h-1):
        #         t = np.array([i,j-1,d_im[j-1,i,0]],dtype="float64")
        #         f = np.array([i-1,j,d_im[j,i-1,0]],dtype="float64")
        #         c = np.array([i,j,d_im[j,i,0]] , dtype = "float64")
        #         d = np.cross(f-c,t-c)
        #         n = d / np.sqrt((np.sum(d**2)))
        #         normals[j,i,:] = n

        if vis:
            self.vis(laplacian , edges, sobely0)
            

        return stairs_lines
    
    def vis(self,laplacian,edges, sobely0):
        cv2.imshow("laplacian",laplacian)
        cv2.imshow("edges",edges)
        cv2.imshow("sobely",sobely0)


class AlgoRunner:
    def __init__(self,bag_obj):
        self.bag_obj = bag_obj
        
        self.stair_detector = StairDetector()
        self.normal_estimator = NormalEstimation()

        self.static_thresholds = [0.065,1]
        self.dynamic_thresholds = {}
        
        

    def __len__(self):
        return len(self.bag_obj.depth_df)

    def get_current_step(self, step):
        data = {}
        data["depth"] = np.load(self.bag_obj.depth_df.np_path[step])
        data["depth_img"] = cv2.imread(self.bag_obj.depth_df.frame_path[step])
        #data["rgb_img"] = cv2.imread(bag_read.rgb_df.frame_path[step])
        
        return data

    def crop_regions(self,img,h_grid, w_grid):

        img[:,:w_grid[1]] = 0
        img[:,w_grid[3 ]:] = 0
        #img[:h_grid[1],:] = 0          

        return img
        

    def is_SS(self,std_grid,mean_grid):
        if((std_grid[1,1]<self.static_thresholds[0])
            and(std_grid[1,2]<self.static_thresholds[0])
            and(std_grid[2,1]<self.static_thresholds[1])
            and(std_grid[2,2]<self.static_thresholds[1])):
            self.dynamic_thresholds["sa"] = [(mean_grid[0,j] -3*std_grid[0,j]) for j in range(std_grid.shape[1])]
            self.dynamic_thresholds["sd"] = [(mean_grid[0,j] +3*std_grid[0,j]) for j in range(std_grid.shape[1])]
            
            return True
        else:
            return False

    def is_GLW(self,mean_grid,stairs_lines):
        

        if((((self.dynamic_thresholds["sa"][1]<mean_grid[0,1])and(mean_grid[0,1]<self.dynamic_thresholds["sd"][1]))
            or
            ((self.dynamic_thresholds["sa"][2]<mean_grid[0,2])and(mean_grid[0,2]<self.dynamic_thresholds["sd"][2])))
            and(len(stairs_lines)==0)):
            
            return True
        else:
            return False


    def is_SD(self,mean_grid,stairs_lines):
        
        if((mean_grid[0,1]>=self.dynamic_thresholds["sd"][1])
            and(mean_grid[0,2]>=self.dynamic_thresholds["sd"][2])
            and(len(stairs_lines)!=0)):

            return True
        else:
            return False


    def is_SA(self,mean_grid,stairs_lines):
        
        if((mean_grid[0,1]<=self.dynamic_thresholds["sa"][1])
            and(mean_grid[0,2]<=self.dynamic_thresholds["sa"][2])
            and(len(stairs_lines)!=0)):

            return True
        else:
            return False

    def intent_recognizer(self,out_data):
        state = ""
        mean_grid, std_grid, lines = out_data["mean"], out_data["std"],out_data["lines"] 
        
        if self.is_SS(std_grid,mean_grid):
            state = "SS"
        else:
            
            if self.is_GLW(mean_grid,lines):
                state = "GLW"
            
            elif self.is_SA(mean_grid,lines):
                state = "SA"
            
            elif self.is_SD(mean_grid,lines):
                state = "SD"

            else:
                state = "Dynamic"

        return state

    def run(self):
        
        
        algo_buffer = []
        
        for step in range(len(self)):
            out_data = {}

            in_data = self.get_current_step(step)
            img_depth = in_data["depth_img"].copy()
            depth =  in_data["depth"].copy()

            img_grid, h_grid, w_grid = dp.split_to_regions(depth)

            image_cropped = self.crop_regions(img_depth, h_grid, w_grid)

            mean_grid = dp.get_regions_mean(img_grid)    
            std_grid = dp.get_regions_std(img_grid)
            #lines = self.stair_detector.detect(image_cropped,in_data["depth"], vis=True)
            
            lines = self.normal_estimator.estimate(img_depth,depth, vis= True)
            
            out_data["lines"], out_data["mean"], out_data["std"] = lines, mean_grid, std_grid
            out_data["intent"] = self.intent_recognizer(out_data)

            self.vis_step(in_data,out_data)
            algo_buffer.append(out_data)

        self.save_runner(algo_buffer)

    def save_runner(self,algo_buffer):
        pass

    def vis_step(self,in_data,out_data):
        
        print(out_data["intent"])
        if out_data["lines"] is not None:           
            for line in out_data["lines"]:
                for x1,y1,x2,y2 in line:            
                    cv2.line(in_data["depth_img"],(x1,y1),(x2,y2),(0,255,0),2)
                    #cv2.line(in_data["rgb_img"],(x1,y1),(x2,y2),(0,255,0),2)

        cv2.imshow("depth", in_data["depth_img"])
        #cv2.imshow("depth", in_data["rgb_img"])
        
        
        if cv2.waitKey(33) == ord('q'): 
            cv2.destroyAllWindows()   # Esc key to stop
            raise Exception()
        
        else:
            cv2.waitKey(0)
        


def main():
    bag_read = READER('/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-10-27-12-42-56.bag')
    bag_read.read()
    algo_runner = AlgoRunner(bag_read)
    algo_runner.run()

if __name__ == "__main__":
    main()