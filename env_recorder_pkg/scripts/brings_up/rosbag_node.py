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
        tf_topic = "/tf"
        imu_topic = "/zedm/zed_node/imu/data"
        rgb_topic = "/zedm/zed_node/rgb/image_rect_color"
        depth_topic = "/zedm/zed_node/depth/depth_registered"
        confidence_topic = "/zedm/zed_node/confidence/confidence_map"
        
        if cfg.rec_imu:
            topics_list.append(imu_topic)
        
        if cfg.rec_rgb:
            topics_list.append(rgb_topic)
        
        if cfg.rec_tf:
            topics_list.append(tf_topic)
        
        if cfg.rec_confidence:
            topics_list.append(confidence_topic)
        
        if cfg.rec_depth:
            topics_list.append(depth_topic)

        self.record_script = '/home/zion/catkin_ws/src/ros_env_prediction/env_recorder_pkg/scripts/brings_up/record_zed.sh '
        self.record_folder = '/home/zion/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag'
        
        rospy.on_shutdown(self.stop_recording_handler)

        # Start recording.
        # command1 = "source " + self.launch_script
        # self.p1 = subprocess.Popen(command1, stdin=subprocess.PIPE, shell=True,
        #                             executable='/bin/bash')
        
        command = "source " + self.record_script + " ".join(topics_list) #" ".join(topics_list)
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