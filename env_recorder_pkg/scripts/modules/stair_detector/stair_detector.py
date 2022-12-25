# imports
import os
import sys
from omegaconf import DictConfig
from datetime import datetime
from typing import Union
import cv2
import numpy as np
import scipy.signal as ss
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# import from parallel modules
# sys.path.insert(0, '/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/scripts/modules/bag_parser')
# from bag_parser.bag_parser import Parser

sys.path.insert(0, '/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/scripts/modules')
from bag_reader.bag_reader import BagReader

sys.path.insert(0, '/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/scripts/modules/image_data_handler')
from image_data_handler.image_data_handler import DepthHandler, ImageHandler
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