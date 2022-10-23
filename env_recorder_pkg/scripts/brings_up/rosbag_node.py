#!/usr/bin/env /home/zion/py3.7_ws/py3.7_venv/bin/python
import hydra
from omegaconf import DictConfig
import rospy
import subprocess
import os
import signal
import time


class RosbagRecord:
    def __init__(self, cfg):
        
        topics_list = []
        
        if cfg.recording.rec_imu:
            topics_list.append(cfg.topics.imu)
        
        if cfg.recording.rec_rgb:
            topics_list.append(cfg.topics.rgb)
        
        if cfg.recording.rec_tf:
            topics_list.append(cfg.topics.tf)
        
        if cfg.recording.rec_confidence:
            topics_list.append(cfg.topics.confidence)
        
        if cfg.recording.rec_depth:
            topics_list.append(cfg.topics.depth)

        self.record_script = cfg.recording.script
        self.record_folder = cfg.recording.bag_folder
        
        rospy.on_shutdown(self.stop_recording_handler)

        # Start recording.
        # command1 = "source " + self.launch_script
        # self.p1 = subprocess.Popen(command1, stdin=subprocess.PIPE, shell=True,
        #                             executable='/bin/bash')
        
        command = "source " + self.record_script +" "+  " ".join(topics_list) #" ".join(topics_list)
        self.p = subprocess.Popen(command, stdin=subprocess.PIPE, cwd=self.record_folder,   shell=True,
                                    executable='/bin/bash')
        
        rospy.spin()
        
        

    def terminate_ros_node(self, s):
        # Adapted from http://answers.ros.org/question/10714/start-and-stop-rosbag-within-a-python-script/
        list_cmd = subprocess.Popen("rosnode list", shell=True, stdout=subprocess.PIPE)
        list_output = list_cmd.stdout.read()
        retcode = list_cmd.wait()
        assert retcode == 0, "List command returned %d" % retcode
        for str in list_output.split(b"\n"):
            str_decode = str.decode('utf8')
            if (str_decode.startswith(s)):
                os.system("rosnode kill " + str_decode)

    def stop_recording_handler(self):
        print("ctrl-c detected")
        self.terminate_ros_node("/record")


@hydra.main(config_path="../../config", config_name = "record")
def main(cfg):
    rospy.init_node('rosbag_record')
    rospy.loginfo(rospy.get_name() + ' start')

    # Go to class functions that do all the heavy lifting. Do error checking.
    try:
        rosbag_record = RosbagRecord(cfg)
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()