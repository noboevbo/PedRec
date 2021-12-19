# PedRecNet2D
## Trial Name
**experiment_pedrec_p2d_h36m**
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
| 0     | 0.70192    | 52.61         | 91.76        | 0.7039        | 0:00:12       | 29.06        | 92.96       | 0.7012       | 0:00:11      | 47.19         | 90.21        | 0.7007        | 0:00:17       | 0:17:24    |
| 1     | 0.70039    | 52.75         | 91.59        | 0.7039        | 0:00:12       | 29.94        | 93.65       | 0.7010       | 0:00:11      | 48.79         | 90.64        | 0.7005        | 0:00:16       | 0:17:28    |
| 2     | 0.70007    | 53.01         | 91.73        | 0.7039        | 0:00:12       | 29.46        | 93.98       | 0.7010       | 0:00:11      | 50.00         | 90.87        | 0.7003        | 0:00:16       | 0:17:02    |
| 3     | 0.69984    | 53.10         | 91.64        | 0.7039        | 0:00:11       | 30.01        | 94.22       | 0.7009       | 0:00:10      | 51.78         | 90.98        | 0.7002        | 0:00:16       | 0:17:10    |
| 4     | 0.69962    | 53.43         | 91.67        | 0.7039        | 0:00:11       | 30.29        | 94.32       | 0.7008       | 0:00:10      | 52.06         | 91.05        | 0.7001        | 0:00:15       | 0:16:28    |
| 5     | 0.69944    | 53.76         | 91.73        | 0.7038        | 0:00:11       | 30.95        | 94.22       | 0.7008       | 0:00:10      | 52.72         | 91.24        | 0.7000        | 0:00:16       | 0:16:27    |
| 6     | 0.69927    | 53.92         | 91.82        | 0.7037        | 0:00:11       | 31.46        | 94.56       | 0.7007       | 0:00:10      | 52.87         | 91.36        | 0.7000        | 0:00:16       | 0:16:28    |
| 7     | 0.69912    | 54.26         | 91.92        | 0.7037        | 0:00:12       | 31.62        | 94.72       | 0.7007       | 0:00:10      | 53.58         | 91.42        | 0.6999        | 0:00:16       | 0:16:43    |
| 8     | 0.69902    | 54.22         | 91.89        | 0.7037        | 0:00:12       | 31.68        | 94.65       | 0.7007       | 0:00:11      | 53.86         | 91.46        | 0.6999        | 0:00:16       | 0:17:29    |
| 9     | 0.69897    | 54.25         | 91.94        | 0.7037        | 0:00:12       | 31.50        | 94.66       | 0.7007       | 0:00:11      | 53.81         | 91.48        | 0.6999        | 0:00:17       | 0:17:29    |
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
| 0     | 0.69885    | 54.82         | 92.22        | 0.7035        | 0:00:12       | 32.29        | 94.93       | 0.7006       | 0:00:11      | 55.60         | 91.99        | 0.6996        | 0:00:16       | 0:20:45    |
| 1     | 0.69870    | 54.84         | 92.27        | 0.7034        | 0:00:12       | 33.11        | 95.18       | 0.7004       | 0:00:10      | 57.44         | 92.29        | 0.6994        | 0:00:16       | 0:20:51    |
| 2     | 0.69856    | 55.08         | 92.34        | 0.7034        | 0:00:12       | 33.11        | 95.51       | 0.7004       | 0:00:10      | 57.63         | 92.33        | 0.6993        | 0:00:16       | 0:20:58    |
| 3     | 0.69843    | 55.19         | 92.40        | 0.7034        | 0:00:12       | 33.57        | 95.60       | 0.7003       | 0:00:10      | 58.25         | 92.41        | 0.6993        | 0:00:16       | 0:20:45    |
| 4     | 0.69836    | 55.26         | 92.47        | 0.7033        | 0:00:12       | 33.56        | 95.57       | 0.7003       | 0:00:10      | 58.41         | 92.42        | 0.6993        | 0:00:16       | 0:21:15    |
