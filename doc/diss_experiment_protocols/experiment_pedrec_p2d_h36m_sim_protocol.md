# PedRecNet2D
## Trial Name
**experiment_pedrec_p2d_h36m_sim**
## Initialization
| Network Part                   | Initialization                                               |
| ------------------------------ | ------------------------------------------------------------ |
| net.model.feature_extractor    | TODO                                                         |
| net.model.conv_transpose_shared | TODO                                                         |
| net.model.head_pose_2d         | TODO                                                         |
| net.model.head_pose_3d         | TODO                                                         |
| net.model.head_orientation     | TODO                                                         |
| net.model.head_conf            | TODO                                                         |
Notes. Initialized from p2d_h36m
## Datasets
### Training
| Dataset                   | Subsampling | Full set length | Used length |
| ------------------------- | ----------- | --------------- | ----------- |
| COCO (TRAIN)              | 1           | 149813          | 149813      |
| rt_rom_01b.pkl            | 1           | 147729          | 147729      |
| rt_validate_3d.pkl        | 1           | 39484           | 39484       |
| h36m_train_pedrec.pkl     | 10          | 155976          | 155976      |
### Validation
| Dataset                   | Subsampling | Full set length | Used length |
| ------------------------- | ----------- | --------------- | ----------- |
| COCO (VAL)                | 1           | 6352            | 6352        |
| rt_validate_3d.pkl        | 10          | 3949            | 3949        |
| h36m_val_pedrec.pkl       | 64          | 8604            | 8604        |
## Augmentation
### Training Augmentations - COCO
| Augmentation              | Value                     |
| ------------------------- | ------------------------- |
| Scale                     | 0.25                      |
| Flip                      | True                      |
| Rotate                    | 30                        |
### Training Augmentations - SIM
| Augmentation              | Value                     |
| ------------------------- | ------------------------- |
| Scale                     | 0.25                      |
| Flip                      | True                      |
| Rotate                    | 0                         |
### Training Augmentations - H36M
| Augmentation              | Value                     |
| ------------------------- | ------------------------- |
| Scale                     | 0.25                      |
| Flip                      | True                      |
| Rotate                    | 0                         |
## General data preparation
```python
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```
suggested lr: 2.08e-03
## Round 0
- Epochs: **10**
- LR Scheduler: **OneCycleLR**
### Optimizer
| Property                  | Value                     |
| ------------------------- | ------------------------- |
| name                      | AdamW                     |
| lr                        | 0.001                     |
| betas                     | (0.9, 0.999)              |
| eps                       | 1e-08                     |
| weight_decay              | 0.01                      |
| amsgrad                   | False                     |
### LRs
| Network Part              | LRs                  | Frozen? |
| ------------------------- | -------------------- | ------- |
| net.model.feature_extractor | 2.08e-03, 2.08e-03   | True    |
| net.model.conv_transpose_shared | 2.08e-03, 2.08e-03   | False   |
| net.model.head_pose_2d    | 2.08e-03, 2.08e-03   | False   |
| net.model.head_pose_3d    | 0.00e+00, 0.00e+00   | False   |
| net.model.head_orientation | 0.00e+00, 0.00e+00   | False   |
| net.model.head_conf       | 0.00e+00, 0.00e+00   | False   |
### Results
| Epoch | Train Loss | COCO PCK@0.05 | COCO PCK@0.2 | COCO Val Loss | COCO Val Time | SIM PCK@0.05 | SIM PCK@0.2 | SIM Val Loss | SIM Val Time | H36M PCK@0.05 | H36M PCK@0.2 | H36M Val Loss | H36M Val Time | Train Time |
| ----- | ---------- | ------------- | ------------ | ------------- | ------------- | ------------ | ----------- | ------------ | ------------ | ------------- | ------------ | ------------- | ------------- | ---------- |
| 0     | 0.69881    | 53.16         | 92.00        | 0.7036        | 0:00:21       | 63.73        | 99.00       | 0.6974       | 0:00:18      | 56.65         | 92.01        | 0.6995        | 0:00:28       | 0:49:54    |
| 1     | 0.69808    | 52.63         | 91.81        | 0.7038        | 0:00:21       | 72.79        | 99.45       | 0.6967       | 0:00:18      | 56.63         | 91.87        | 0.6996        | 0:00:28       | 0:49:53    |
| 2     | 0.69785    | 52.88         | 91.87        | 0.7038        | 0:00:21       | 77.01        | 99.64       | 0.6964       | 0:00:18      | 55.36         | 91.95        | 0.6996        | 0:00:28       | 0:49:57    |
| 3     | 0.69768    | 52.82         | 91.87        | 0.7038        | 0:00:21       | 79.39        | 99.70       | 0.6962       | 0:00:18      | 54.96         | 91.91        | 0.6996        | 0:00:28       | 0:49:54    |
| 4     | 0.69753    | 53.38         | 91.92        | 0.7037        | 0:00:21       | 82.06        | 99.70       | 0.6960       | 0:00:18      | 57.10         | 91.91        | 0.6995        | 0:00:28       | 0:49:56    |
| 5     | 0.69739    | 53.57         | 92.10        | 0.7037        | 0:00:21       | 83.55        | 99.77       | 0.6959       | 0:00:18      | 57.09         | 92.08        | 0.6995        | 0:00:28       | 0:49:57    |
| 6     | 0.69727    | 53.77         | 92.10        | 0.7036        | 0:00:21       | 85.15        | 99.81       | 0.6958       | 0:00:18      | 58.29         | 92.19        | 0.6994        | 0:00:28       | 0:49:57    |
| 7     | 0.69717    | 53.99         | 92.16        | 0.7036        | 0:00:21       | 86.08        | 99.83       | 0.6957       | 0:00:18      | 57.91         | 92.25        | 0.6994        | 0:00:28       | 0:49:56    |
| 8     | 0.69709    | 53.96         | 92.16        | 0.7036        | 0:00:21       | 86.36        | 99.83       | 0.6957       | 0:00:18      | 57.95         | 92.22        | 0.6994        | 0:00:28       | 0:50:10    |
| 9     | 0.69706    | 53.94         | 92.19        | 0.7036        | 0:00:21       | 86.42        | 99.83       | 0.6957       | 0:00:18      | 57.93         | 92.20        | 0.6994        | 0:00:28       | 0:49:57    |
## Round 1
- Epochs: **5**
- LR Scheduler: **OneCycleLR**
### Optimizer
| Property                  | Value                     |
| ------------------------- | ------------------------- |
| name                      | AdamW                     |
| lr                        | 0.001                     |
| betas                     | (0.9, 0.999)              |
| eps                       | 1e-08                     |
| weight_decay              | 0.01                      |
| amsgrad                   | False                     |
### LRs
| Network Part              | LRs                  | Frozen? |
| ------------------------- | -------------------- | ------- |
| net.model.feature_extractor | 1.04e-04, 1.04e-04   | False   |
| net.model.conv_transpose_shared | 2.08e-04, 2.08e-04   | False   |
| net.model.head_pose_2d    | 2.08e-04, 2.08e-04   | False   |
| net.model.head_pose_3d    | 0.00e+00, 0.00e+00   | False   |
| net.model.head_orientation | 0.00e+00, 0.00e+00   | False   |
| net.model.head_conf       | 0.00e+00, 0.00e+00   | False   |
### Results
| Epoch | Train Loss | COCO PCK@0.05 | COCO PCK@0.2 | COCO Val Loss | COCO Val Time | SIM PCK@0.05 | SIM PCK@0.2 | SIM Val Loss | SIM Val Time | H36M PCK@0.05 | H36M PCK@0.2 | H36M Val Loss | H36M Val Time | Train Time |
| ----- | ---------- | ------------- | ------------ | ------------- | ------------- | ------------ | ----------- | ------------ | ------------ | ------------- | ------------ | ------------- | ------------- | ---------- |
| 0     | 0.69700    | 54.06         | 92.07        | 0.7036        | 0:00:21       | 88.38        | 99.90       | 0.6955       | 0:00:18      | 57.81         | 92.23        | 0.6993        | 0:00:28       | 1:00:56    |
| 1     | 0.69693    | 54.05         | 92.12        | 0.7036        | 0:00:21       | 90.00        | 99.93       | 0.6954       | 0:00:18      | 58.88         | 92.31        | 0.6993        | 0:00:28       | 1:00:55    |
| 2     | 0.69683    | 54.17         | 92.13        | 0.7035        | 0:00:21       | 91.10        | 99.95       | 0.6953       | 0:00:18      | 59.56         | 92.45        | 0.6992        | 0:00:28       | 1:00:55    |
| 3     | 0.69673    | 54.55         | 92.33        | 0.7034        | 0:00:21       | 91.74        | 99.95       | 0.6952       | 0:00:18      | 59.86         | 92.55        | 0.6991        | 0:00:28       | 1:00:54    |
| 4     | 0.69667    | 54.46         | 92.28        | 0.7034        | 0:00:21       | 91.80        | 99.95       | 0.6952       | 0:00:18      | 59.66         | 92.54        | 0.6992        | 0:00:28       | 1:00:56    |
