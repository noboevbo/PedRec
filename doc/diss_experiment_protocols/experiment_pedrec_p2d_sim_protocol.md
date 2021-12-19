# PedRecNet2D
## Trial Name
**experiment_pedrec_p2d_sim**
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
| rt_rom_01b.pkl            | 1           | 147729          | 147729      |
| rt_validate_3d.pkl        | 1           | 39484           | 39484       |
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
| 0     | 0.70056    | 52.76         | 91.77        | 0.7039        | 0:00:35       | 63.95        | 98.95       | 0.6974       | 0:00:25      | 21.71         | 84.09        | 0.7045        | 0:00:47       | 0:59:39    |
| 1     | 0.69873    | 52.62         | 91.57        | 0.7039        | 0:00:34       | 71.99        | 99.37       | 0.6967       | 0:00:23      | 22.32         | 84.12        | 0.7045        | 0:00:46       | 0:59:38    |
| 2     | 0.69834    | 52.41         | 91.52        | 0.7040        | 0:00:34       | 75.63        | 99.54       | 0.6965       | 0:00:24      | 22.15         | 84.51        | 0.7045        | 0:00:46       | 0:59:38    |
| 3     | 0.69809    | 52.97         | 91.61        | 0.7039        | 0:00:34       | 79.12        | 99.69       | 0.6962       | 0:00:24      | 23.12         | 84.71        | 0.7043        | 0:00:46       | 0:59:37    |
| 4     | 0.69790    | 52.94         | 91.62        | 0.7040        | 0:00:34       | 81.12        | 99.69       | 0.6961       | 0:00:24      | 23.59         | 84.68        | 0.7044        | 0:00:46       | 0:59:38    |
| 5     | 0.69774    | 53.36         | 91.76        | 0.7039        | 0:00:34       | 83.27        | 99.77       | 0.6959       | 0:00:24      | 22.67         | 84.91        | 0.7043        | 0:00:46       | 0:59:38    |
| 6     | 0.69759    | 53.73         | 91.85        | 0.7037        | 0:00:34       | 84.31        | 99.78       | 0.6958       | 0:00:24      | 22.38         | 85.08        | 0.7042        | 0:00:46       | 0:59:38    |
| 7     | 0.69748    | 53.98         | 91.80        | 0.7037        | 0:00:34       | 85.84        | 99.82       | 0.6957       | 0:00:24      | 23.03         | 84.90        | 0.7044        | 0:00:46       | 0:59:38    |
| 8     | 0.69740    | 54.11         | 91.96        | 0.7037        | 0:00:34       | 86.47        | 99.83       | 0.6957       | 0:00:24      | 23.08         | 85.10        | 0.7043        | 0:00:46       | 0:59:39    |
| 9     | 0.69737    | 54.09         | 91.89        | 0.7037        | 0:00:34       | 86.52        | 99.84       | 0.6957       | 0:00:23      | 23.40         | 85.13        | 0.7042        | 0:00:46       | 0:59:38    |
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
| 0     | 0.69724    | 54.52         | 92.18        | 0.7035        | 0:00:34       | 88.70        | 99.92       | 0.6955       | 0:00:23      | 23.45         | 85.50        | 0.7041        | 0:00:46       | 1:11:42    |
| 1     | 0.69712    | 54.46         | 92.15        | 0.7036        | 0:00:34       | 90.29        | 99.94       | 0.6954       | 0:00:23      | 24.37         | 85.70        | 0.7039        | 0:00:46       | 1:11:41    |
| 2     | 0.69701    | 54.62         | 92.21        | 0.7035        | 0:00:34       | 91.52        | 99.96       | 0.6953       | 0:00:23      | 24.59         | 85.87        | 0.7039        | 0:00:46       | 1:11:40    |
| 3     | 0.69691    | 54.93         | 92.37        | 0.7034        | 0:00:34       | 91.95        | 99.96       | 0.6952       | 0:00:23      | 24.80         | 85.97        | 0.7038        | 0:00:46       | 1:11:40    |
| 4     | 0.69685    | 54.94         | 92.34        | 0.7034        | 0:00:34       | 92.14        | 99.97       | 0.6952       | 0:00:24      | 24.57         | 85.85        | 0.7039        | 0:00:46       | 1:11:41    |
