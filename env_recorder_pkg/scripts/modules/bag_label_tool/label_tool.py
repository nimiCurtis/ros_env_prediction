import os
import argparse
import sys
import pandas as pd
from enum import Enum
import cv2
import numpy as np

sys.path.insert(0, '/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/scripts/modules')
from bag_parser.bag_parser import Parser, is_bag_dir, is_bag_file
from bag_reader.bag_reader import BagReader

class EnvLabel(Enum):
    GL = 1
    SA = 2
    SD = 3
    
    
    
    W = 9


class LabelTool:

    def __init__(self):
        pass
    

    def set_bag_labels(self,bag_obj,video_name =None):
        video_folder = bag_obj.bag_read.datafolder+'/video'
        if video_name is not None: video_file = os.path.join(video_folder,video_name)
        else: video_file = os.path.join(video_folder,os.listdir(video_folder)[0])
        
        dataset_file = bag_obj.bag_read.datafolder+'/feature_line/features.csv'

        capture = cv2.VideoCapture(video_file)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print('[INFO]  Frame count:', frame_count)
        
        frame_i = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        labels_dict = {}
        
        _, frame = capture.read()

        while True:
            k= cv2.waitKey(0)    

            if k==ord('f'):
                frame_i+=1
                if frame_i>frame_count-1:
                    frame_i = frame_count-1

                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
                _, frame = capture.read()

            elif k==ord('b'):
                frame_i-=1
                if frame_i<0:
                    frame_i = 0

                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
                _, frame = capture.read()
                
            elif k==ord('c'):
                labels_dict[f"frame{frame_i}"] = label

            elif k==ord('q'):
                cv2.destroyAllWindows()
                break
            
            elif k==ord('l'):
                label = self.set_frame_label(frame_i)
                labels_dict[f"frame{frame_i}"] = label

            elif k==ord('s'):
                
                start = int(input("Insert index of starting frame : "))
                stop = int(input("Insert index of ending frame : "))
                while (not self.isvalid_frame(start,frame_count)) or (not self.isvalid_frame(stop,frame_count)) or (start>=stop):
                    start = int(input("Insert index of starting frame : "))
                    stop = int(input("Insert index of ending frame : "))
                
                label = int(input(f"Insert label for frames {start} - {stop} : "))
                self.set_frames_label(labels_dict,start,stop,label)
                
            elif k==ord('m'):
                
                start = int(input("Insert index of starting frame : "))
                stop = int(input("Insert index of ending frame : "))
                while (not self.isvalid_frame(start,frame_count)) or (not self.isvalid_frame(stop,frame_count)) or (start>=stop):
                    start = int(input("Insert index of starting frame : "))
                    stop = int(input("Insert index of ending frame : "))
                
                label_list = [int(input("Insert top label: ")),int(input("Insert mid label: ")),int(input("Insert bottom label: "))]
                self.set_frames_label(labels_dict,start,stop,label_list)


            keys = labels_dict.keys()
            if f"frame{frame_i}" in keys:
                label = labels_dict[f'frame{frame_i}']
                if isinstance(label,list):
                    text = f"frame : {frame_i} | label: [{EnvLabel(label[0]).name},{EnvLabel(label[1]).name},{EnvLabel(label[2]).name}]"
                elif isinstance(label,list):
                    text = f"frame : {frame_i} | label: {EnvLabel(label).name}"

            else:
                text = f"frame : {frame_i}"

            clone = frame.copy()
            cv2.putText(clone,text,org=(20,30),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,thickness=2,color=(0,0,255))
            cv2.imshow('frame', clone)
            if len(labels_dict)==frame_count:
                print("[Info]  All frames are labeld")

        if len(labels_dict)==frame_count:
            labels_list = self.return_labels_as_list(labels_dict)
            if len(labels_list[0])==1: ### need to change in refactoring
                df = pd.read_csv(dataset_file,index_col=0)
                df['labels'] = labels_list
            elif len(labels_list[0])==3: ### need to change in refactoring
                labels_numpy = np.array(labels_list)
                
                df = pd.read_csv(dataset_file,index_col=0)
                df['top_labels'] = labels_numpy[:,0].tolist()
                df['mid_labels'] = labels_numpy[:,1].tolist()
                df['bot_labels'] = labels_numpy[:,2].tolist()

            df.to_csv(dataset_file)
            print("[Info]  Labels saved")
        else:
            print("[Info]  Exiting without saving labels")

    def isvalid_frame(self,frame_i,frame_count):
        if frame_i in range(frame_count):
            return True
        else:
            return False

    def set_frame_label(self,frame_i):
        label = int(input(f"Insert label of frame {frame_i}: "))
        
        return label
    
    def set_frames_label(self,labels_dict,start,stop,label):
        for i in range(start,stop+1):
            labels_dict[f"frame{i}"] = label
        pass

    def return_labels_as_list(self,labels_dict):
        labels_list  = [v for k, v in labels_dict.items()]

        return labels_list

class MultiLabelTool(LabelTool):

    def __init__(self):
        super().__init__()

    def set_bag_labels(self,bag_obj,video_name =None):
        video_folder = bag_obj.bag_read.datafolder+'/video'
        if video_name is not None: video_file = os.path.join(video_folder,video_name)
        else: video_file = os.path.join(video_folder,os.listdir(video_folder)[0])
        
        multi_dataset_file = bag_obj.bag_read.datafolder+'/feature_line/multi_features.csv'
        single_dataset_file = bag_obj.bag_read.datafolder+'/feature_line/features.csv'
        capture = cv2.VideoCapture(video_file)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print('[INFO]  Frame count:', frame_count)
        
        frame_i = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        labels_dict = {}
        
        _, frame = capture.read()

        while True:
            k= cv2.waitKey(0)    

            if k==ord('f'):
                frame_i+=1
                if frame_i>frame_count-1:
                    frame_i = frame_count-1

                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
                _, frame = capture.read()
                
                
                
            elif k==ord('b'):
                frame_i-=1
                if frame_i<0:
                    frame_i = 0

                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
                _, frame = capture.read()
                
            elif k==ord('c'):
                labels_dict[f"frame{frame_i}"] = label

            elif k==ord('q'):
                cv2.destroyAllWindows()
                break
            
            elif k==ord('l'):
                label = self.set_frame_label(frame_i)
                labels_dict[f"frame{frame_i}"] = label

            elif k==ord('s'):

                start = int(input("Insert index of starting frame : "))
                stop = int(input("Insert index of ending frame : "))
                while (not self.isvalid_frame(start,frame_count)) or (not self.isvalid_frame(stop,frame_count)) or (start>=stop):
                    start = int(input("Insert index of starting frame : "))
                    stop = int(input("Insert index of ending frame : "))
                
                label_list = [int(input("Insert top label: ")),int(input("Insert mid label: ")),int(input("Insert bottom label: "))]
                self.set_frames_label(labels_dict,start,stop,label_list)


            keys = labels_dict.keys()
            if f"frame{frame_i}" in keys:
                label = labels_dict[f'frame{frame_i}']
                if isinstance(label,list):
                    text = f"frame : {frame_i} | label: [{EnvLabel(label[0]).name},{EnvLabel(label[1]).name},{EnvLabel(label[2]).name}]"
                elif isinstance(label,list):
                    text = f"frame : {frame_i} | label: {EnvLabel(label).name}"

            else:
                text = f"frame : {frame_i}"

            clone = frame.copy()
            cv2.putText(clone,text,org=(20,30),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,thickness=2,color=(0,0,255))
            cv2.imshow('frame', clone)
            if len(labels_dict)==frame_count:
                print("[Info]  All frames are labeld")

        if len(labels_dict)==frame_count:
            labels_list = self.return_labels_as_list(labels_dict)
            labels_numpy = np.array(labels_list)
            
            df_m = pd.read_csv(multi_dataset_file,index_col=0) 
            df_m['top_labels'] = labels_numpy[:,0].tolist()
            df_m['mid_labels'] = labels_numpy[:,1].tolist()
            df_m['bot_labels'] = labels_numpy[:,2].tolist()
            df_m.to_csv(multi_dataset_file)
            print("[Info]  MultiLabels saved")

            df_s = pd.read_csv(single_dataset_file,index_col=0) 
            df_s['labels'] = labels_numpy.reshape(1,-1)
            df_s.to_csv(single_dataset_file)
            print("[Info]  SingleLabels saved")

        else:
            print("[Info]  Exiting without saving labels")


    def set_frame_label(self,frame_i):
        print(f"Insert labels of frame {frame_i}. In the form of - [top,mid,bot] ")
        label_list = [int(input("Insert top label: ")),int(input("Insert mid label: ")),int(input("Insert bottom label: "))]
        
        return label_list


class LabelParser(Parser):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description ='Bag Iterator')

        parser.add_argument('-b', dest="single_bag", type=is_bag_file,
                            help="Use single bag file only")
        
        parser.add_argument('-mb', dest="multilabel_single_bag", type=is_bag_file,
                            help="Use single bag file only for multilabel")

        parser.add_argument('-v', dest="video_name", type=is_vid_file,
                            help="Insert video name")

        return parser.parse_args()

def is_vid_file(arg_vid_str: str) -> str:
    """"""
    # check file validation
    if arg_vid_str.split('.')[-1]=='mp4':
        return arg_vid_str
    else:
        msg = f"Given video file {arg_vid_str} is not valid! "
        raise argparse.ArgumentTypeError(msg)


def main():

    bag_obj = BagReader()
    
    args = LabelParser.get_args()
    # default
    bag_file = '/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-12-27-18-06-43.bag' 

    if args.single_bag is not None: 
        bag_file = args.single_bag
        label_tool = LabelTool()
    elif args.multilabel_single_bag is not None:
        bag_file = args.multilabel_single_bag
        label_tool = MultiLabelTool()

    bag_obj.bag = bag_file
    
    if args.video_name is not None:
        video_name = args.video_name    
        label_tool.set_bag_labels(bag_obj,video_name=video_name)
    else:
        label_tool.set_bag_labels(bag_obj)

if __name__ == "__main__":
    main()