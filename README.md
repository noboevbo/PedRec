# PedRecNet
This repository contains the code for the PedRecNet as well as EHPI3D. It is the successor of our EHPI2D work (https://github.com/noboevbo/ehpi_action_recognition). The PedRecNet is a multi-purpose network that provides the following functions:

- Human BB Detection (via YoloV4).
- Human Tracking
- 2D Human Pose Estimation
- 3D Human Pose Estimation
- Human Body Orientation (currently only Phi) Estimation
- Human Head Orientation (currently only Phi) Estimation
- "Pedestrian recognizes the camera" estimation
- Human Action Recognition (via EHPI3D)

Note: This work is currently unpublished. It is part of my PhD dissertation and we are currently in the process to prepare (a? maybe more) paper. Note also, that, for now, I am no longer active in research, thus this code is provided as is.

[![PedRecNet: Demo01 - Pedestrian crossing the street + Hitchhike](https://img.youtube.com/vi/IPeJK1Bk5qY/0.jpg)](https://www.youtube.com/watch?v=IPeJK1Bk5qY)

[![PedRecNet Demo 02: Multiple Pedestrians](https://img.youtube.com/vi/xUcTKKGHfEs/0.jpg)](https://www.youtube.com/watch?v=xUcTKKGHfEs)


# Installation
## Requirements
- Python 3.9 (venv suggested)
- working CUDA / CUDNN

## Installation steps
- Clone this repository
- cd PedRec
- pip install -r requirements.txt
- Download the pretrained models if you want to run the PedRecNet
- Download the required datasets, dataframes and maybe some of the checkpoints (see Dataset Download section)

## Required Data
### Pretrained models
- [YoloV4 weights](https://dennisnotes.com/files/pedrec/models/yolo_v4/yolov4.pth) (adapted from https://github.com/Tianxiaomo/pytorch-YOLOv4). Place it in *data/models/yolo_v4/yolov4.pth*.
- [PedRecNet weights](https://dennisnotes.com/files/pedrec/models/pedrec/experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0_net.pth) - place it in *data/models/pedrec/experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0_net.pth*.
- [EHPI3D weights](https://dennisnotes.com/files/pedrec/models/ehpi3d/ehpi_3d_sim_c01_actionrec_gt_pred_64frames.pth) - Place it in *models/ehpi3d/ehpi_3d_sim_c01_actionrec_gt_pred_64frames.pth*.

Not required to run the network but for some experiments / trainings:
- [Simple Baselines for Human Pose Estimation Weights](https://dennisnotes.com/files/pedrec/models/human_pose_baseline/pose_resnet_50_256x192.pth.tar) - adapted from https://github.com/microsoft/human-pose-estimation.pytorch - place it in data/models/human_pose_baseline/pose_resnet_50_256x192.pth.tar.

### Datasets
- If you want to train the network(s) yourself, you need the following datasets:
  - [COCO (2017)](https://cocodataset.org/#download)
    - Additionally: [MEBOW body orientation annotations](https://github.com/ChenyanWu/MEBOW) - train_hoe.json and val_hoe.json need to be placed in COCO/annotations
  - Human3.6m
    - Additionally: [train/36m_train_pedrec.pkl](https://dennisnotes.com/files/pedrec/datasets/H36M/h36m_train_pedrec.pkl) in h36m_dir/train/ and [*train/36m_val_pedrec.pkl*](https://dennisnotes.com/files/pedrec/datasets/H36M/h36m_val_pedrec.pkl) in h36m_dir/val/
  - [ROMb (SIM-ROM)](https://dennisnotes.com/files/pedrec/datasets/ROMb.7z)
  - [RT3DValidate (SIM-Circle)](https://dennisnotes.com/files/pedrec/datasets/RT3DValidate.7z)
  - [TUD](https://www.mpi-inf.mpg.de/de/departments/computer-vision-and-machine-learning/research/people-detection-pose-estimation-and-tracking/monocular-3d-pose-estimation-and-tracking-by-detection) - cvpr10_multiview_pedestrians
- For action recognition:
  - SIM-C01 Pose Data (raw image data not published, but you only require the skeleton dataframe for training!)
    - [SIM-C01 Train](https://dennisnotes.com/files/pedrec/datasets/SIM-C01/rt_conti_01_train_FIN.pkl)
    - [SIM-C01 Val](https://dennisnotes.com/files/pedrec/datasets/SIM-C01/rt_conti_01_val.pkl)

Download the datasets and place the additional .pkls in the appropriate folders. Update the paths in experiment_path_helper.py and execute one of the experiments in training/. You might need some intermediate weights if you do not start with experiment_pedrec_2d! You can find them at https://dennisnotes.com/files/pedrec/single_results/filename.pth.

### Demo files
- [Some C01 real examples](https://dennisnotes.com/files/pedrec/demo/05070850_9672.m4v) - place it in *data/demo/* or updated paths in demo_actionrec_dev.
- [Pedestrians crossing a street](https://www.pexels.com/de-de/video/855565/) - place it in *data/demo/* or updated paths in demo_actionrec_dev.

## Installation tips
Currently I would recommend to use a PIP environment instead of Anaconda. I tried the (recommended) Anaconda environment for PyTorch various times, but the performance is hugely inferior to the PIP environment on my system(s). Using Anaconda I get about 9FPS on videos with a single human compared to 25FPS on my PIP environment. One thing I noticed is that the performance difference shrinks the more people are in a video, thus with 7+ people the performance of the Anaconda and the PIP environment are almost equal. If someone has an idea what the problem could be, please notify me. Things tested:

- CUDA / CUDNN are working enabled and recognized by PyTorch on both environments
- Pillow-SIMD installed
- Usage of opencv-contrib-python-headless instead of the Conda version.

# Demo / Run
Check out the *demo_actionrec_dev.py* file. It contains examples on how to run the application on videos, image dirs, images and a webcam via the "input providers".
Example (if you've downloaded the demo videos!):

python pedrec/demo_actionrec_dev.py

# Generate own training data
Check out the panda dataframes (e.g. the rt_conti_01_train_FIN.pkl from SIM-C01 dataset, or the pkls from the H36M dataset). If you provide a dataset of the same structure you can just use the pedrec dataset class.
You can find some scripts I used to generate the dataframes in tools/datasets/... but I have not tested them in a while.
The same applies for EHPI3D action recognition data: Check out the dataframes from the rt_conti_01_train_FIN.pkl file! You might want to checkout the notebook *dataset_rtsim_conti01_ehpi* as well. You can find the result files (e.g. the C01F_train_pred_df_experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0_allframes.pkl) at https://dennisnotes.com/files/pedrec/result_dfs/filename.

# Notebooks
I've just pasted a few of my notebooks in the notebooks folder. They are not cleaned up and may contain absolute paths etc. but maybe they help the one or other to understand some concepts / validation results.

# Appendix
Note: probably outdated information! Need to recheck this part.

## Numpy "Datatypes"
note: not really datatypes, those types are stored in numpy arrays due to performance considerations.
There are helper methods providing more userfriendly access to those values (e.g. joint_helper(_3d), bb_helper).
Those datatypes are the ones used internally in the PedRecNet application, there might be differences in types used in e.g. datasets etc.

| "Datatype" name | Shape |
| -------------------------- |----------- |
| bb_2d | center_x, center_y, width, height, confidence, class_idx |
| joint_2d | x, y, confidence | 
| joint_3d | x, y, z, confidence | 

## Expected shapes of PedRecNet HDF5 dataset files
note: n = dataset length

| dataset name               | Shape      | DType   | Description |
| -------------------------- |----------- | ------- | ----------  |
| img_paths                  | (n)        | str     |img path, relative to the dataset root |
| joints2d                   | (n,17,4)   | float32 |17 = joints, 4 = x, y, confidence, visibility (coordinates in pixels, starting from top left of the image) |
| skeleton_3d_hip_normalized | (n,17,5)   | float32 | 17 = joints, 5 = x, y, z, confidence, visibility (coordinates in mm) |
| env_position               | (n,3)      | float32 | 3 = x, y, z (mm) |
| body_orientation           | (n,4)      | float32 | 4 = theta, phi, confidence, visibility |
| head_orientation           | (n,4)      | float32 | 4 = theta, phi, confidence, visibility |
| bbs                        | (n,6)      | float32 | 5 = center_x, center_y, width, height, confidence, class_idx |
| scene_idx_range            | (n,2)      | uint32  | 2 = scene_idx_start, scene_idx_stop the index range in the hdf5 file containing data from the same scene |
| actions                    | (n)        | uint32  | List = dynamic sized list of action ids, e.g. [[1, 2], [3, 4, 5]] |
| movements                  | (n)        | uint32  | ids, see constants for ID <-> NAME mapping |
| movement_speeds            | (n)        | uint32  | ids, see constants for ID <-> NAME mapping |
| genders                    | (n)        | uint32  | ids, see constants for ID <-> NAME mapping |
| skin_colors                | (n)        | uint32  | ids, see constants for ID <-> NAME mapping |
| sizes                      | (n)        | uint32  | ids, see constants for ID <-> NAME mapping |
| weights                    | (n)        | uint32  | ids, see constants for ID <-> NAME mapping |
| ages                       | (n)        | uint32  | ids, see constants for ID <-> NAME mapping |
| frame_nr_locals            | (n)        | uint32  | frame number of the current scene |
| frame_nr_global            | (n)        | uint32  | frame number of the complete record |


## Original dataset notes
Some notes to original datasets.
Important: Those notes do NOT apply to internal PedRec usage, the original datasets are converted to PedRec Datasets before usage, thus those notes can usually be ignored.

### Human3.6M

#### BB Structure
They use a binary mask containing 1s in the bounding box area.
#### Joint Order
- 0 = 'Hips'
- 1 = 'RightUpLeg'
- 2 = 'RightLeg'
- 3 = 'RightFoot'
- 4 = 'RightToeBase'
- 5 = 'Site'  - ????
- 6 = 'LeftUpLeg'
- 7 = 'LeftLeg'
- 8 = 'LeftFoot'
- 9 = 'LeftToeBase'
- 10 = 'Site'  - ????
- 11 = 'Spine'
- 12 = 'Spine1'
- 13 = 'Neck'
- 14 = 'Head'
- 15 = 'Site'
- 16 = 'LShoulder'
- 17 = 'LeftArm'
- 18 = 'LeftForeArm'
- 19 = 'LeftHand'
- 20 = 'LeftHandThumb'
- 21 = 'Site'
- 22 = 'L_Wrist_End'
- 23 = 'Site'
- 24 = 'RightShoulder'
- 25 = 'RightArm'
- 26 = 'RightForeArm'
- 27 = 'RightHand'
- 28 = 'LeftHandThumb'
- 29 = 'Site'
- 30 = 'L_Wrist_End'
- 31 = 'Site'

# Attributions
- YoloV4 object detection: https://github.com/Tianxiaomo/pytorch-YOLOv4
- Pose-resnet as base network: https://github.com/microsoft/human-pose-estimation.pytorch

## Icons
- Skeleton by Wolf Böse from the Noun Project
- Head by Naveen from the Noun Project
- body by Makarenko Andrey from the Noun Project
- Eye by Simon Sim from the Noun Project
- jogging by Adrien Coquet from the Noun Project
- Walk by Adrien Coquet from the Noun Project
- stand by Gan Khoon Lay from the Noun Project
- sit by Adrien Coquet from the Noun Project

# Contact
- Dennis Burgermeister, Cognitive Systems Research Group, Reutlingen University (no longer active)
- Cristóbal Curio, Cognitive Systems Research Group, Reutlingen University

# Acknowledgment
This project was funded by the Continental AG.
