# ros_env_prediction : package containing the tools and scripts use for Environment Prediction research. 

## Overview

The Environment Prediction research is running by ROB-TAU research team in collaboration with [ReWalk Robotics](https://rewalk.com/)  
as a part of the HRI collection of researches. 

TBD
## Hardware and System architecture

TBD

## Installation
TBD (need to add python 3.7 installation, zed api installation, ros installation and catkin_ws)

Clone the repository:

```bash
git clone git@github.com:nimiCurtis/ros_env_prediction.git
```
NOTE: using a virtual environment is recommended for the installation of the requiered packages 

The requiements are seperated for the Jetson Nano requierments as the data collector and actual tool for implement the trained models, and for the processing requierments which recomended to be install on PC for better computational capabilities such as DL/ML models training (will build in the future).  

You can install them by:

```bash
python3 -m pip install -r requirements/jetson_req.txt #for jetson
python3 -m pip install -r requirements/remote_pc_req.txt #for pc
```

Finally for using the ZED camera SDK of stereolabs you must download the SDK properly.
Therefore you should follow the SDK installation process for PC/nvidia_jetson.

See: [Download and Install the ZED SDK](https://www.stereolabs.com/docs/installation/jetson/).


## Content
TBD

The pacakge contains several folder:

- [config](env_recorder_pkg/config): contains ```record.yaml``` 
- [bag](env_recorder_pkg/bag): contains the saved dataset and bags folders 
- [launch](env_recorder_pkg/launch): ....
- [params](env_recorder_pkg/params): ....
- [scripts](env_recorder_pkg/scripts):....



## Usage 

TBD

## Helpful Links
TBD 
- [ZED git python API](https://github.com/stereolabs/zed-python-api)
- [ZED API Documentation](https://www.stereolabs.com/docs/api/)
- [ZED examples](https://github.com/stereolabs/zed-examples)
- [ZED ROS Documentation](https://www.stereolabs.com/docs/ros/)
- [ZED ROS examples](https://github.com/stereolabs/zed-ros-examples)

