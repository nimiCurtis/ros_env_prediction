#!/usr/bin/env /home/nimibot/py3.7_ws/py3.7_venv/bin/python

# Libraries
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from omegaconf import DictConfig
import rospy
import rosparam
import subprocess
import os

class RosbagRecord:
    """TBD...
        """    
    def __init__(self, cfg):
        """_summary_

        Args:
            cfg (DictConfig): configuration dictionary for recording process
        """        
        topics_list = []
        
        # record topics depend on the configuration params
        if cfg.recording.rec_imu:                       # record imu
            topics_list.append(cfg.topics.imu)

        if cfg.recording.rec_tf:                        # record tf
            topics_list.append(cfg.topics.tf)

        if cfg.recording.rec_rgb:                       # record rgb
            topics_list.append(cfg.topics.rgb)

        if cfg.recording.rec_depth:                     # record depth
            topics_list.append(cfg.topics.depth)
        
        if cfg.recording.rec_confidence:                # record confidence
            topics_list.append(cfg.topics.confidence)
        
        if cfg.recording.rec_disparity:
            topics_list.append(cfg.topics.disparity)

        self.record_script = cfg.recording.script           # use bash script from path in config
        self.record_folder = cfg.recording.bag_folder       # use folder to store the bag from path in config
        self.recorded_cam_params_folder = cfg.recording.camera_params_fodler

        rospy.on_shutdown(self.stop_recording_handler)      # when ros shuting down execute the handler 
        
        command = "source " + self.record_script +" "+  " ".join(topics_list) # build rosbag command depend on the topic list
        
        # execute bash script for recording using the subprocess.popen module
        self.p = subprocess.Popen(command, 
                                    stdin=subprocess.PIPE,
                                    cwd=self.record_folder,
                                    shell=True,
                                    executable='/bin/bash') 
        
        rospy.spin() # effectively go into an infinite loop until it receives a shutdown signal
        
        

    def terminate_ros_node(self, s):
        """This function terminate the ros node starting with the given argument

            Args:
                s (string): first word of the target node to kill
            """        
        # Adapted from http://answers.ros.org/question/10714/start-and-stop-rosbag-within-a-python-script/
        # get topics to kill from 'rosnode list' using shell command.
        list_cmd = subprocess.Popen("rosnode list", shell=True, stdout=subprocess.PIPE)
        list_output = list_cmd.stdout.read()
        retcode = list_cmd.wait()
        assert retcode == 0, "List command returned %d" % retcode
        for str in list_output.split(b"\n"):            # iterate topics using split(b"\n") | 'b' for byte type
            str_decode = str.decode('utf8')             # decode to string
            if (str_decode.startswith(s)):              # if it starts with string which 's' stored >> kill it
                os.system("rosnode kill " + str_decode) # kill node

    def stop_recording_handler(self):
        """This function execute the terminate function when ros shutdown
            """        
        rospy.loginfo("Ctrl-c detected")
        self.terminate_ros_node("/record")
        rospy.loginfo("Bag saved")

        rospy.loginfo("Saving camera configurations..")
        os.mkdir(self.recorded_cam_params_folder)
        rosparam.dump_params(self.recorded_cam_params_folder+"/zedm.yaml",param="zedm")
        rosparam.dump_params(self.recorded_cam_params_folder+"/common.yaml",param="common")
        
        

# Use hydra for configuration managing
@hydra.main(config_path="../../config", config_name = "record")
def main(cfg):
    rospy.init_node('rosbag_record')                # Init node
    rospy.loginfo(rospy.get_name() + ' start')      

    
    # Go to class functions that do all the heavy lifting. Do error checking.
    try:
        rosbag_record = RosbagRecord(cfg)
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':


    main()
