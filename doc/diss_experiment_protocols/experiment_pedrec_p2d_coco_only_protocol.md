# PedRecNet2D
## Trial Name
**experiment_pedrec_p2d_coco_only**
## Initialization
| Network Part                   | Initialization                                               |
| ------------------------------ | ------------------------------------------------------------ |
| net.model.feature_extractor    | TODO                                                         |
| net.model.conv_transpose_shared | TODO                                                         |
| net.model.head_pose_2d         | TODO                                                         |
| net.model.head_pose_3d         | TODO                                                         |
| net.model.head_orientation     | TODO                                                         |
| net.model.head_conf            | TODO                                                         |
Notes. Initialized from ResNet
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
suggested lr: 1.59e-03
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
| net.model.feature_extractor | 1.59e-03, 1.59e-03   | True    |
| net.model.conv_transpose_shared | 1.59e-03, 1.59e-03   | False   |
| net.model.head_pose_2d    | 1.59e-03, 1.59e-03   | False   |
| net.model.head_pose_3d    | 0.00e+00, 0.00e+00   | False   |
| net.model.head_orientation | 0.00e+00, 0.00e+00   | False   |
| net.model.head_conf       | 0.00e+00, 0.00e+00   | False   |
### Results
| Epoch | Train Loss | COCO PCK@0.05 | COCO PCK@0.2 | COCO Val Loss | COCO Val Time | SIM PCK@0.05 | SIM PCK@0.2 | SIM Val Loss | SIM Val Time | H36M PCK@0.05 | H36M PCK@0.2 | H36M Val Loss | H36M Val Time | Train Time |
| ----- | ---------- | ------------- | ------------ | ------------- | ------------- | ------------ | ----------- | ------------ | ------------ | ------------- | ------------ | ------------- | ------------- | ---------- |
| 0     | 0.70854    | 51.78         | 90.53        | 0.7048        | 0:00:12       | 28.50        | 64.39       | 0.7239       | 0:00:10      | 16.13         | 51.29        | 0.7333        | 0:00:16       | 0:08:19    |
| 1     | 0.70327    | 54.84         | 92.25        | 0.7036        | 0:00:12       | 29.95        | 64.63       | 0.7234       | 0:00:10      | 16.45         | 51.43        | 0.7315        | 0:00:16       | 0:08:21    |
| 2     | 0.70235    | 54.65         | 92.21        | 0.7036        | 0:00:12       | 28.48        | 65.19       | 0.7246       | 0:00:10      | 16.15         | 52.17        | 0.7301        | 0:00:16       | 0:08:17    |
| 3     | 0.70235    | 54.68         | 92.21        | 0.7036        | 0:00:12       | 29.54        | 65.94       | 0.7261       | 0:00:10      | 16.49         | 52.58        | 0.7309        | 0:00:16       | 0:08:17    |
| 4     | 0.70228    | 54.93         | 92.28        | 0.7035        | 0:00:11       | 29.25        | 65.99       | 0.7256       | 0:00:10      | 16.44         | 52.65        | 0.7303        | 0:00:16       | 0:07:59    |
| 5     | 0.70219    | 55.17         | 92.28        | 0.7035        | 0:00:11       | 29.22        | 66.73       | 0.7246       | 0:00:10      | 16.94         | 52.97        | 0.7301        | 0:00:16       | 0:07:59    |
| 6     | 0.70211    | 55.60         | 92.37        | 0.7034        | 0:00:11       | 30.12        | 66.44       | 0.7280       | 0:00:10      | 17.21         | 53.07        | 0.7312        | 0:00:16       | 0:07:59    |
| 7     | 0.70201    | 55.76         | 92.48        | 0.7034        | 0:00:11       | 29.80        | 66.68       | 0.7260       | 0:00:10      | 17.35         | 53.23        | 0.7301        | 0:00:16       | 0:07:58    |
| 8     | 0.70196    | 55.95         | 92.53        | 0.7033        | 0:00:12       | 29.77        | 66.53       | 0.7257       | 0:00:10      | 17.35         | 53.17        | 0.7304        | 0:00:16       | 0:08:11    |
| 9     | 0.70192    | 55.87         | 92.51        | 0.7033        | 0:00:11       | 29.84        | 66.39       | 0.7267       | 0:00:10      | 17.42         | 53.19        | 0.7304        | 0:00:15       | 0:08:07    |
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
| net.model.feature_extractor | 7.97e-05, 7.97e-05   | False   |
| net.model.conv_transpose_shared | 1.59e-04, 1.59e-04   | False   |
| net.model.head_pose_2d    | 1.59e-04, 1.59e-04   | False   |
| net.model.head_pose_3d    | 0.00e+00, 0.00e+00   | False   |
| net.model.head_orientation | 0.00e+00, 0.00e+00   | False   |
| net.model.head_conf       | 0.00e+00, 0.00e+00   | False   |
### Results
| Epoch | Train Loss | COCO PCK@0.05 | COCO PCK@0.2 | COCO Val Loss | COCO Val Time | SIM PCK@0.05 | SIM PCK@0.2 | SIM Val Loss | SIM Val Time | H36M PCK@0.05 | H36M PCK@0.2 | H36M Val Loss | H36M Val Time | Train Time |
| ----- | ---------- | ------------- | ------------ | ------------- | ------------- | ------------ | ----------- | ------------ | ------------ | ------------- | ------------ | ------------- | ------------- | ---------- |
| 0     | 0.70192    | 56.12         | 92.57        | 0.7033        | 0:00:11       | 30.44        | 66.53       | 0.7266       | 0:00:10      | 17.37         | 53.33        | 0.7306        | 0:00:15       | 0:09:34    |
| 1     | 0.70187    | 56.27         | 92.57        | 0.7032        | 0:00:12       | 30.53        | 66.56       | 0.7256       | 0:00:11      | 17.42         | 53.32        | 0.7299        | 0:00:16       | 0:10:09    |
| 2     | 0.70182    | 56.34         | 92.55        | 0.7032        | 0:00:12       | 30.34        | 66.28       | 0.7260       | 0:00:11      | 17.51         | 53.23        | 0.7297        | 0:00:16       | 0:10:09    |
| 3     | 0.70172    | 56.36         | 92.64        | 0.7032        | 0:00:12       | 30.06        | 66.30       | 0.7259       | 0:00:10      | 17.62         | 53.31        | 0.7298        | 0:00:16       | 0:10:11    |
| 4     | 0.70166    | 56.47         | 92.65        | 0.7032        | 0:00:12       | 30.13        | 66.38       | 0.7253       | 0:00:10      | 17.73         | 53.31        | 0.7297        | 0:00:16       | 0:10:06    |
