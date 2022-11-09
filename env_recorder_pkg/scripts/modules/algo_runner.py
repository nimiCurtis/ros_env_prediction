
# test algo:
#   - with different algo params (stairs detection / grid / crop) 
#   - with different recording params (ROS yaml files of the camera)
#   - with stair detector implementation on rgb/depth/disparity


########################

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bag_reader.bag_reader import BagReader
from bag_processor.bag_processor import DepthProcessor
dp = DepthProcessor()

#feat_df = dp.get_features_df(bag_read.depth_df)

class StairDetector:

    def __init__(self):
        pass

    def detect(self,img_depth,depth, vis=True):
        stairs_lines = []
        threshold_sobel = 100     

        blured = cv2.GaussianBlur(img_depth,(11,11),0,0)
        #https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html
        laplacian = cv2.Laplacian(blured,cv2.CV_64F)
        sobelx = cv2.Sobel(blured,cv2.CV_64F,1,0,ksize=5) 
        sobely = cv2.Sobel(blured,cv2.CV_64F,0,1,ksize=5)
        #sobely0 = sobely.astype("float32")
        sobely_abs = np.abs(sobely)
        sobely0 = sobely_abs + np.abs(sobely_abs.min())
        sobely0 = (sobely0/sobely0.max())*255
        sobely0 = sobely0.astype("uint8")
        sobely0[sobely0<threshold_sobel] = 0 

        # Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
        

        edges = cv2.Canny(sobely0,100 ,250,apertureSize = 3)
        
        minLineLength = 90
        maxLineGap = 10  

        lines = cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength,maxLineGap)
        
        if lines is not None:           
            for line in lines:
                for x1,y1,x2,y2 in line:            
                    if x1 != x2 and ((y1 > 10 and y2 > 10)and (y1 < 240 and y2 < 240)):                   
                        m = (y1-y2)/(x1-x2)                    
                        if np.rad2deg(np.arctan(m))<20 and np.rad2deg(np.arctan(m))>-20 and depth[y1,x1]>300.0 and depth[y2,x2]> 300.0 and depth[y1,x1]!=np.inf :
                            stairs_lines.append(line) 
        if vis:
            self.vis(blured,edges,sobely)

        return stairs_lines

    def vis(self,blured,edges,sobely):
        cv2.imshow("blured",blured)
        cv2.imshow("edges",edges)
        cv2.imshow("sobely",sobely)
        


class NormalEstimation:

    def __init__(self):
        pass

    def estimate(self,img_depth,depth, vis=True):
        zy, zx, _ = np.gradient(img_depth)  
        # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
        # to reduce noise
        #zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)     
        #zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)

        normal = np.dstack((-zx, -zy, np.ones_like(img_depth)))
        n = np.linalg.norm(normal, axis=2)
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n

        # offset and rescale values to be in 0-255
        normal += 1
        normal /= 2

        if vis:
            self.vis(normal)
            

        return normal
    
    def vis(self,normal):
        cv2.imshow("normal",normal[:, :, ::-1])




class AlgoRunner:
    def __init__(self,bag_obj):
        self.bag_obj = bag_obj
        
        self.stair_detector = StairDetector()
        self.normal_estimator = NormalEstimation()
        self.static_thresholds = [0.08,0.95]    
        self.dynamic_thresholds = {} 
        

    def __len__(self):
        return len(self.bag_obj.depth_df)

    def get_current_step(self, step):
        data = {}
        data["depth"] = np.load(self.bag_obj.depth_df.np_path[step])
        data["depth_img"] = cv2.imread(self.bag_obj.depth_df.frame_path[step])
        data["disparity_img"] = cv2.imread(self.bag_obj.rgb_df.frame_path[step])
        
        return data

    def crop_regions(self,img,h_grid, w_grid):

        img[:,:w_grid[1]] = 0
        img[:,w_grid[3 ]:] = 0
        #img[h_grid[2]:,:] = 0          

        return img
        

    def is_SS(self,std_grid,mean_grid):
        if((std_grid[1,1]<self.static_thresholds[0])
            and(std_grid[1,2]<self.static_thresholds[0])
            and(std_grid[2,1]<self.static_thresholds[1])
            and(std_grid[2,2]<self.static_thresholds[1])):
            
            self.dynamic_thresholds["sa"] = [(mean_grid[0,j] -2.5*std_grid[0,j]) for j in range(std_grid.shape[1])]
            self.dynamic_thresholds["sd"] = [(mean_grid[0,j] +2.5*std_grid[0,j]) for j in range(std_grid.shape[1])]

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
            img_depth = in_data["disparity_img"].copy()
            depth =  in_data["depth"].copy()

            img_grid, h_grid, w_grid = dp.split_to_regions(depth)

            image_cropped = self.crop_regions(img_depth, h_grid, w_grid)

            mean_grid = dp.get_regions_mean(img_grid)    
            std_grid = dp.get_regions_std(img_grid)
            
            lines = self.stair_detector.detect(img_depth,depth, vis= True)
            #normal = self.normal_estimator.estimate(img_depth,depth, vis= True)
            
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
                    cv2.line(in_data["disparity_img"],(x1,y1),(x2,y2),(0,255,0),2)
                    #cv2.line(in_data["rgb_img"],(x1,y1),(x2,y2),(0,255,0),2)

        cv2.imshow("rgb", in_data["disparity_img"])
        #cv2.imshow("depth", in_data["rgb_img"])

        if cv2.waitKey(33) == ord('q'): 
            cv2.destroyAllWindows()   # Esc key to stop
            raise Exception()
        
        else:
            cv2.waitKey(0)
        


def main():
    bag_read = READER('/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-11-08-10-07-30.bag')
    bag_read.read()
    algo_runner = AlgoRunner(bag_read)
    algo_runner.run()

if __name__ == "__main__":
    main()