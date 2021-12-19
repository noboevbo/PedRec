# PedRecNet2D
## Trial Name
**experiment_pedrec_p2d_c**
## Initialization
| Network Part                   | Initialization                                               |
| ------------------------------ | ------------------------------------------------------------ |
| net.model.feature_extractor    | TODO                                                         |
| net.model.conv_transpose_shared | TODO                                                         |
| net.model.head_pose_2d         | TODO                                                         |
| net.model.head_pose_3d         | TODO                                                         |
| net.model.head_orientation     | TODO                                                         |
| net.model.head_conf            | TODO                                                         |
Notes. Initialized from p2d_coco_only
## Datasets
### Training
| Dataset                   | Subsampling | Full set length | Used length |
| ------------------------- | ----------- | --------------- | ----------- |
| COCO (TRAIN)              | 1           | 149813          | 149813      |
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
suggested lr: 2.29e-05
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
| net.model.feature_extractor | 2.29e-05, 2.29e-05   | True    |
| net.model.conv_transpose_shared | 2.29e-05, 2.29e-05   | False   |
| net.model.head_pose_2d    | 2.29e-05, 2.29e-05   | False   |
| net.model.head_pose_3d    | 0.00e+00, 0.00e+00   | False   |
| net.model.head_orientation | 0.00e+00, 0.00e+00   | False   |
| net.model.head_conf       | 2.29e-05, 2.29e-05   | False   |
### Results
| Epoch | Train Loss | COCO PCK@0.05 | COCO PCK@0.2 | COCO JointAcc | COCO Val Loss | COCO Val Time | SIM PCK@0.05 | SIM PCK@0.2 | SIM JointAcc | SIM Val Loss | SIM Val Time | H36M PCK@0.05 | H36M PCK@0.2 | H36M JointAcc | H36M Val Loss | H36M Val Time | Train Time |
| ----- | ---------- | ------------- | ------------ | ------------- | ------------- | ------------- | ------------ | ----------- | ------------ | ------------ | ------------ | ------------- | ------------ | ------------- | ------------- | ------------- | ---------- |
| 0     | 1.48004    | 56.50         | 92.68        | 0.69          | 1.2815        | 0:00:12       | 30.22        | 66.42       | 0.67         | 1.3347       | 0:00:11      | 17.78         | 53.39        | 0.67          | 1.3481        | 0:00:16       | 0:08:19    |
| 1     | 1.23403    | 56.48         | 92.70        | 0.85          | 1.0206        | 0:00:11       | 30.12        | 66.34       | 0.74         | 1.1583       | 0:00:10      | 17.71         | 53.30        | 0.75          | 1.1388        | 0:00:15       | 0:08:10    |
| 2     | 1.02285    | 56.51         | 92.68        | 0.88          | 0.8689        | 0:00:11       | 29.96        | 66.06       | 0.72         | 1.1879       | 0:00:10      | 17.48         | 53.15        | 0.75          | 1.1114        | 0:00:15       | 0:07:59    |
| 3     | 0.88901    | 56.56         | 92.68        | 0.89          | 0.7755        | 0:00:11       | 30.13        | 66.14       | 0.72         | 1.1537       | 0:00:10      | 17.60         | 53.20        | 0.74          | 1.0388        | 0:00:16       | 0:08:16    |
| 4     | 0.80312    | 56.45         | 92.68        | 0.89          | 0.7110        | 0:00:11       | 30.20        | 66.11       | 0.72         | 1.0933       | 0:00:10      | 17.62         | 53.20        | 0.73          | 0.9752        | 0:00:15       | 0:08:05    |
| 5     | 0.73928    | 56.44         | 92.70        | 0.90          | 0.6629        | 0:00:11       | 30.23        | 66.20       | 0.72         | 1.0342       | 0:00:10      | 17.58         | 53.20        | 0.74          | 0.9144        | 0:00:15       | 0:07:59    |
| 6     | 0.69522    | 56.44         | 92.63        | 0.90          | 0.6317        | 0:00:11       | 30.22        | 66.09       | 0.71         | 1.0200       | 0:00:10      | 17.55         | 53.11        | 0.73          | 0.8906        | 0:00:15       | 0:07:59    |
| 7     | 0.66764    | 56.45         | 92.67        | 0.90          | 0.6131        | 0:00:11       | 29.95        | 66.12       | 0.71         | 1.0095       | 0:00:10      | 17.60         | 53.17        | 0.73          | 0.8737        | 0:00:15       | 0:07:58    |
| 8     | 0.65290    | 56.52         | 92.73        | 0.90          | 0.6060        | 0:00:11       | 30.03        | 66.07       | 0.71         | 0.9919       | 0:00:10      | 17.60         | 53.15        | 0.73          | 0.8655        | 0:00:15       | 0:07:59    |
| 9     | 0.64956    | 56.41         | 92.65        | 0.90          | 0.6050        | 0:00:11       | 30.13        | 65.98       | 0.71         | 0.9950       | 0:00:10      | 17.57         | 53.13        | 0.73          | 0.8646        | 0:00:15       | 0:08:00    |
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
| net.model.feature_extractor | 1.15e-06, 1.15e-06   | False   |
| net.model.conv_transpose_shared | 2.29e-06, 2.29e-06   | False   |
| net.model.head_pose_2d    | 2.29e-06, 2.29e-06   | False   |
| net.model.head_pose_3d    | 0.00e+00, 0.00e+00   | False   |
| net.model.head_orientation | 0.00e+00, 0.00e+00   | False   |
| net.model.head_conf       | 2.29e-06, 2.29e-06   | False   |
### Results
| Epoch | Train Loss | COCO PCK@0.05 | COCO PCK@0.2 | COCO JointAcc | COCO Val Loss | COCO Val Time | SIM PCK@0.05 | SIM PCK@0.2 | SIM JointAcc | SIM Val Loss | SIM Val Time | H36M PCK@0.05 | H36M PCK@0.2 | H36M JointAcc | H36M Val Loss | H36M Val Time | Train Time |
| ----- | ---------- | ------------- | ------------ | ------------- | ------------- | ------------- | ------------ | ----------- | ------------ | ------------ | ------------ | ------------- | ------------ | ------------- | ------------- | ------------- | ---------- |
| 0     | 0.64806    | 56.49         | 92.69        | 0.90          | 0.6030        | 0:00:11       | 30.25        | 66.07       | 0.72         | 0.9891       | 0:00:10      | 17.62         | 53.12        | 0.73          | 0.8606        | 0:00:15       | 0:09:35    |
| 1     | 0.64148    | 56.49         | 92.69        | 0.90          | 0.5973        | 0:00:11       | 30.26        | 66.15       | 0.71         | 0.9866       | 0:00:10      | 17.69         | 53.20        | 0.73          | 0.8608        | 0:00:15       | 0:09:35    |
| 2     | 0.63558    | 56.45         | 92.67        | 0.90          | 0.5922        | 0:00:11       | 30.30        | 66.17       | 0.71         | 0.9841       | 0:00:10      | 17.59         | 53.19        | 0.73          | 0.8517        | 0:00:15       | 0:09:35    |
| 3     | 0.63195    | 56.36         | 92.69        | 0.90          | 0.5904        | 0:00:11       | 30.14        | 66.11       | 0.71         | 0.9872       | 0:00:10      | 17.60         | 53.18        | 0.73          | 0.8531        | 0:00:15       | 0:09:57    |
| 4     | 0.63204    | 56.44         | 92.68        | 0.90          | 0.5897        | 0:00:12       | 30.15        | 66.16       | 0.71         | 0.9846       | 0:00:10      | 17.64         | 53.17        | 0.73          | 0.8526        | 0:00:16       | 0:09:47    |
