/home/dennis/.virtualenvs/pedrec/bin/python /home/dennis/code/python/pedrec/pedrec/evaluations/eval_ehpi3d.py
2021-09-02 20:32:04,814 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
2021-09-02 20:32:06,298 numexpr.utils INFO     Note: NumExpr detected 24 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
2021-09-02 20:32:06,298 numexpr.utils INFO     NumExpr defaulting to 8 threads.
  0%|          | 0/2539 [00:00<?, ?it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/usr/lib/python3.9/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /build/python-pytorch/src/pytorch-1.9.0-opt-cuda/c10/core/TensorImpl.h:1153.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
  0%|          | 1/2539 [00:01<1:16:14,  1.80s/it]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
100%|██████████| 2539/2539 [00:20<00:00, 122.30it/s]
### GT_RESULT_RATIO: 1 (ehpi_3d_sim_c01_actionrec_pred)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $86.9$ & $94.3$ & $84.6$ & $94.8$ & $76.4$ & 45652 \\
ACTION.IDLE & $70.5$ & $67.5$ & $56.5$ & $88.9$ & $41.4$ & 10168 \\
ACTION.WALK & $90.1$ & $95.2$ & $89.6$ & $88.1$ & $91.3$ & 57699 \\
ACTION.JOG & $84.0$ & $81.3$ & $74.6$ & $78.7$ & $71.0$ & 16207 \\
ACTION.WAVE & $76.8$ & $60.5$ & $53.5$ & $52.1$ & $54.9$ & 2970 \\
ACTION.KICK_BALL & $58.8$ & $47.2$ & $29.5$ & $90.2$ & $17.7$ & 1144 \\
ACTION.THROW & $61.3$ & $26.3$ & $28.3$ & $37.3$ & $22.9$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $82.2$ & $86.3$ & $76.0$ & $90.8$ & $65.4$ & 14819 \\
ACTION.HITCHHIKE & $76.1$ & $79.5$ & $66.4$ & $90.8$ & $52.3$ & 4018 \\
ACTION.TURN_AROUND & $65.2$ & $51.2$ & $44.2$ & $78.6$ & $30.7$ & 3494 \\
ACTION.WORK & $69.1$ & $45.7$ & $46.5$ & $57.0$ & $39.3$ & 4263 \\
ACTION.ARGUE & $78.8$ & $76.6$ & $70.6$ & $91.0$ & $57.6$ & 961 \\
ACTION.STUMBLE & $52.0$ & $17.2$ & $7.7$ & $73.9$ & $4.1$ & 837 \\
ACTION.OPEN_DOOR & $63.8$ & $43.5$ & $40.8$ & $78.1$ & $27.6$ & 1192 \\
ACTION.FALL & $84.4$ & $46.6$ & $58.9$ & $51.3$ & $69.2$ & 591 \\
ACTION.STAND_UP & $86.2$ & $59.0$ & $49.7$ & $37.6$ & $73.3$ & 873 \\
ACTION.FIGHT & $58.4$ & $18.3$ & $22.4$ & $32.5$ & $17.1$ & 990 \\
Total: mBAcc 73.21%, mAP: 58.60%, CF1: 52.93%, CP 71.27%, CR: 47.76%, OF1: 79.16%, OP 86.33%, OR: 73.09%
2021-09-02 20:32:31,941 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
  0%|          | 0/2539 [00:00<?, ?it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  0%|          | 1/2539 [00:00<21:03,  2.01it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  0%|          | 12/2539 [00:00<01:37, 25.87it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
100%|██████████| 2539/2539 [00:19<00:00, 128.91it/s]
### GT_RESULT_RATIO: 1 (ehpi_3d_sim_c01_actionrec_gt)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $91.8$ & $96.5$ & $90.3$ & $94.2$ & $86.8$ & 45652 \\
ACTION.IDLE & $74.4$ & $80.6$ & $61.8$ & $81.6$ & $49.8$ & 10168 \\
ACTION.WALK & $92.7$ & $96.3$ & $92.3$ & $90.4$ & $94.4$ & 57699 \\
ACTION.JOG & $86.9$ & $91.1$ & $82.8$ & $92.9$ & $74.8$ & 16207 \\
ACTION.WAVE & $88.9$ & $88.9$ & $80.5$ & $82.8$ & $78.2$ & 2970 \\
ACTION.KICK_BALL & $74.6$ & $67.6$ & $63.9$ & $90.8$ & $49.3$ & 1144 \\
ACTION.THROW & $73.5$ & $64.5$ & $56.8$ & $71.3$ & $47.2$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $92.7$ & $91.9$ & $85.5$ & $83.3$ & $87.8$ & 14819 \\
ACTION.HITCHHIKE & $91.6$ & $93.2$ & $89.1$ & $95.8$ & $83.3$ & 4018 \\
ACTION.TURN_AROUND & $78.4$ & $66.1$ & $65.9$ & $77.4$ & $57.3$ & 3494 \\
ACTION.WORK & $72.3$ & $78.3$ & $59.7$ & $89.1$ & $44.8$ & 4263 \\
ACTION.ARGUE & $75.9$ & $55.9$ & $57.9$ & $65.3$ & $51.9$ & 961 \\
ACTION.STUMBLE & $61.9$ & $45.2$ & $36.3$ & $75.8$ & $23.9$ & 837 \\
ACTION.OPEN_DOOR & $75.9$ & $79.1$ & $67.1$ & $94.9$ & $51.8$ & 1192 \\
ACTION.FALL & $89.4$ & $61.3$ & $61.6$ & $50.4$ & $79.2$ & 591 \\
ACTION.STAND_UP & $67.9$ & $73.9$ & $50.8$ & $87.2$ & $35.9$ & 873 \\
ACTION.FIGHT & $65.1$ & $38.7$ & $38.4$ & $51.7$ & $30.5$ & 990 \\
Total: mBAcc 79.64%, mAP: 74.65%, CF1: 67.09%, CP 80.87%, CR: 60.40%, OF1: 85.80%, OP 89.68%, OR: 82.25%
2021-09-02 20:32:57,015 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
  0%|          | 0/2539 [00:00<?, ?it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  0%|          | 1/2539 [00:00<20:02,  2.11it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
100%|██████████| 2539/2539 [00:18<00:00, 136.17it/s]
### GT_RESULT_RATIO: 1 (ehpi_3d_sim_c01_actionrec_gt_pred)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $91.4$ & $97.0$ & $90.0$ & $95.2$ & $85.4$ & 45652 \\
ACTION.IDLE & $69.7$ & $73.9$ & $54.0$ & $82.0$ & $40.3$ & 10168 \\
ACTION.WALK & $93.5$ & $97.5$ & $93.2$ & $92.1$ & $94.3$ & 57699 \\
ACTION.JOG & $87.9$ & $92.0$ & $83.0$ & $89.9$ & $77.1$ & 16207 \\
ACTION.WAVE & $89.3$ & $90.0$ & $84.4$ & $90.8$ & $78.8$ & 2970 \\
ACTION.KICK_BALL & $84.4$ & $78.5$ & $75.7$ & $83.9$ & $69.0$ & 1144 \\
ACTION.THROW & $77.4$ & $80.3$ & $70.0$ & $96.9$ & $54.8$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $92.9$ & $93.9$ & $88.0$ & $88.7$ & $87.3$ & 14819 \\
ACTION.HITCHHIKE & $91.1$ & $91.4$ & $86.9$ & $92.0$ & $82.4$ & 4018 \\
ACTION.TURN_AROUND & $76.9$ & $69.6$ & $65.2$ & $81.7$ & $54.2$ & 3494 \\
ACTION.WORK & $75.9$ & $83.9$ & $66.5$ & $92.1$ & $52.0$ & 4263 \\
ACTION.ARGUE & $93.8$ & $84.1$ & $79.5$ & $72.5$ & $87.9$ & 961 \\
ACTION.STUMBLE & $66.0$ & $37.5$ & $39.2$ & $50.2$ & $32.1$ & 837 \\
ACTION.OPEN_DOOR & $62.1$ & $67.6$ & $38.2$ & $89.5$ & $24.2$ & 1192 \\
ACTION.FALL & $91.3$ & $59.7$ & $58.8$ & $45.5$ & $83.1$ & 591 \\
ACTION.STAND_UP & $71.7$ & $66.6$ & $52.4$ & $65.6$ & $43.6$ & 873 \\
ACTION.FIGHT & $65.4$ & $50.9$ & $45.1$ & $83.2$ & $30.9$ & 990 \\
Total: mBAcc 81.22%, mAP: 77.33%, CF1: 68.82%, CP 81.87%, CR: 63.38%, OF1: 86.22%, OP 91.07%, OR: 81.85%
2021-09-02 20:33:21,016 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
  0%|          | 0/2539 [00:00<?, ?it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  0%|          | 1/2539 [00:00<20:41,  2.04it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  1%|          | 14/2539 [00:00<01:22, 30.66it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
100%|██████████| 2539/2539 [00:18<00:00, 136.99it/s]
### GT_RESULT_RATIO: 1 (ehpi_3d_sim_c01_actionrec_gt_pred_ehpi2dvids)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $91.0$ & $96.8$ & $89.6$ & $95.2$ & $84.6$ & 45652 \\
ACTION.IDLE & $71.8$ & $79.7$ & $58.5$ & $87.0$ & $44.1$ & 10168 \\
ACTION.WALK & $93.2$ & $97.0$ & $92.8$ & $91.7$ & $93.9$ & 57699 \\
ACTION.JOG & $88.4$ & $91.9$ & $84.3$ & $91.9$ & $77.8$ & 16207 \\
ACTION.WAVE & $88.0$ & $87.0$ & $80.3$ & $84.8$ & $76.3$ & 2970 \\
ACTION.KICK_BALL & $79.7$ & $75.1$ & $72.8$ & $93.5$ & $59.5$ & 1144 \\
ACTION.THROW & $79.8$ & $79.3$ & $73.8$ & $97.0$ & $59.6$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $92.8$ & $93.2$ & $87.3$ & $87.3$ & $87.4$ & 14819 \\
ACTION.HITCHHIKE & $90.1$ & $88.6$ & $81.0$ & $81.1$ & $80.8$ & 4018 \\
ACTION.TURN_AROUND & $76.6$ & $68.1$ & $63.8$ & $78.6$ & $53.7$ & 3494 \\
ACTION.WORK & $72.0$ & $79.9$ & $59.5$ & $91.0$ & $44.1$ & 4263 \\
ACTION.ARGUE & $92.0$ & $87.3$ & $79.4$ & $75.1$ & $84.3$ & 961 \\
ACTION.STUMBLE & $68.5$ & $56.1$ & $51.0$ & $81.2$ & $37.2$ & 837 \\
ACTION.OPEN_DOOR & $58.3$ & $57.8$ & $28.0$ & $90.8$ & $16.5$ & 1192 \\
ACTION.FALL & $89.7$ & $63.6$ & $60.4$ & $48.5$ & $79.9$ & 591 \\
ACTION.STAND_UP & $73.2$ & $69.3$ & $55.5$ & $68.5$ & $46.6$ & 873 \\
ACTION.FIGHT & $64.5$ & $47.1$ & $40.4$ & $66.4$ & $29.1$ & 990 \\
Total: mBAcc 80.56%, mAP: 77.52%, CF1: 68.14%, CP 82.92%, CR: 62.08%, OF1: 85.90%, OP 90.90%, OR: 81.43%
2021-09-02 20:33:44,919 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
100%|██████████| 2539/2539 [00:18<00:00, 140.47it/s]
### GT_RESULT_RATIO: 1 (ehpi_3d_sim_c01_actionrec_gt_pred_no_unit_skeleton)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $91.0$ & $96.7$ & $89.5$ & $94.9$ & $84.7$ & 45652 \\
ACTION.IDLE & $71.5$ & $77.5$ & $57.0$ & $81.7$ & $43.8$ & 10168 \\
ACTION.WALK & $93.1$ & $97.5$ & $92.7$ & $91.4$ & $94.1$ & 57699 \\
ACTION.JOG & $90.0$ & $93.7$ & $85.8$ & $90.9$ & $81.2$ & 16207 \\
ACTION.WAVE & $91.6$ & $93.6$ & $86.5$ & $89.9$ & $83.4$ & 2970 \\
ACTION.KICK_BALL & $69.6$ & $67.8$ & $55.0$ & $92.6$ & $39.2$ & 1144 \\
ACTION.THROW & $63.9$ & $68.2$ & $43.4$ & $98.3$ & $27.8$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $92.7$ & $94.6$ & $87.8$ & $88.7$ & $87.0$ & 14819 \\
ACTION.HITCHHIKE & $93.0$ & $84.5$ & $80.1$ & $74.2$ & $87.0$ & 4018 \\
ACTION.TURN_AROUND & $79.6$ & $75.2$ & $69.6$ & $83.8$ & $59.6$ & 3494 \\
ACTION.WORK & $66.8$ & $66.8$ & $48.2$ & $83.2$ & $33.9$ & 4263 \\
ACTION.ARGUE & $93.9$ & $92.3$ & $84.7$ & $81.7$ & $87.9$ & 961 \\
ACTION.STUMBLE & $61.7$ & $38.5$ & $34.5$ & $65.6$ & $23.4$ & 837 \\
ACTION.OPEN_DOOR & $70.8$ & $70.2$ & $57.6$ & $94.1$ & $41.5$ & 1192 \\
ACTION.FALL & $94.2$ & $63.7$ & $62.3$ & $48.0$ & $88.8$ & 591 \\
ACTION.STAND_UP & $71.0$ & $59.8$ & $51.5$ & $66.2$ & $42.2$ & 873 \\
ACTION.FIGHT & $66.0$ & $60.4$ & $48.4$ & $100.0$ & $31.9$ & 990 \\
Total: mBAcc 80.01%, mAP: 76.53%, CF1: 66.75%, CP 83.82%, CR: 61.02%, OF1: 85.86%, OP 90.41%, OR: 81.74%
2021-09-02 20:34:08,377 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
  0%|          | 0/2539 [00:00<?, ?it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  0%|          | 1/2539 [00:00<20:17,  2.08it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  0%|          | 12/2539 [00:00<01:34, 26.77it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
100%|██████████| 2539/2539 [00:17<00:00, 141.62it/s]
### GT_RESULT_RATIO: 1 (ehpi_3d_sim_c01_actionrec_gt_pred_zero_by_score)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $91.3$ & $96.9$ & $90.0$ & $95.4$ & $85.1$ & 45652 \\
ACTION.IDLE & $71.7$ & $78.0$ & $58.9$ & $89.3$ & $43.9$ & 10168 \\
ACTION.WALK & $93.6$ & $97.5$ & $93.3$ & $91.8$ & $94.8$ & 57699 \\
ACTION.JOG & $89.1$ & $92.4$ & $84.9$ & $91.1$ & $79.4$ & 16207 \\
ACTION.WAVE & $90.2$ & $90.1$ & $82.8$ & $84.8$ & $80.8$ & 2970 \\
ACTION.KICK_BALL & $82.6$ & $76.2$ & $73.9$ & $85.0$ & $65.4$ & 1144 \\
ACTION.THROW & $80.2$ & $82.2$ & $73.1$ & $92.7$ & $60.4$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $92.3$ & $93.5$ & $86.8$ & $87.3$ & $86.3$ & 14819 \\
ACTION.HITCHHIKE & $89.8$ & $89.8$ & $84.2$ & $88.8$ & $80.0$ & 4018 \\
ACTION.TURN_AROUND & $75.7$ & $65.6$ & $62.2$ & $77.5$ & $51.9$ & 3494 \\
ACTION.WORK & $71.1$ & $78.0$ & $58.4$ & $93.9$ & $42.4$ & 4263 \\
ACTION.ARGUE & $95.0$ & $90.9$ & $86.5$ & $83.0$ & $90.2$ & 961 \\
ACTION.STUMBLE & $70.6$ & $51.8$ & $52.3$ & $71.3$ & $41.3$ & 837 \\
ACTION.OPEN_DOOR & $69.6$ & $68.3$ & $54.4$ & $88.5$ & $39.3$ & 1192 \\
ACTION.FALL & $88.4$ & $65.9$ & $57.3$ & $45.6$ & $77.3$ & 591 \\
ACTION.STAND_UP & $66.4$ & $76.5$ & $48.4$ & $92.3$ & $32.8$ & 873 \\
ACTION.FIGHT & $69.0$ & $52.4$ & $52.8$ & $86.4$ & $38.0$ & 990 \\
Total: mBAcc 81.57%, mAP: 79.18%, CF1: 70.59%, CP 84.99%, CR: 64.07%, OF1: 86.48%, OP 91.32%, OR: 82.12%
2021-09-02 20:34:31,486 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
  0%|          | 0/2539 [00:00<?, ?it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  0%|          | 1/2539 [00:00<20:00,  2.11it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
100%|██████████| 2539/2539 [00:18<00:00, 141.01it/s]
### GT_RESULT_RATIO: 1 (ehpi_3d_sim_c01_actionrec_gt_pred_15fps)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $90.9$ & $97.1$ & $89.5$ & $95.3$ & $84.3$ & 45652 \\
ACTION.IDLE & $74.5$ & $86.4$ & $64.5$ & $93.1$ & $49.3$ & 10168 \\
ACTION.WALK & $93.3$ & $97.0$ & $93.0$ & $91.5$ & $94.6$ & 57699 \\
ACTION.JOG & $90.8$ & $93.2$ & $86.9$ & $91.4$ & $82.8$ & 16207 \\
ACTION.WAVE & $93.2$ & $93.4$ & $83.2$ & $79.9$ & $86.9$ & 2970 \\
ACTION.KICK_BALL & $78.5$ & $67.0$ & $69.6$ & $89.3$ & $57.0$ & 1144 \\
ACTION.THROW & $80.6$ & $87.7$ & $74.3$ & $94.8$ & $61.1$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $92.5$ & $94.3$ & $87.6$ & $88.8$ & $86.4$ & 14819 \\
ACTION.HITCHHIKE & $91.8$ & $90.8$ & $84.2$ & $84.2$ & $84.1$ & 4018 \\
ACTION.TURN_AROUND & $74.7$ & $67.3$ & $60.8$ & $77.9$ & $49.9$ & 3494 \\
ACTION.WORK & $75.3$ & $89.5$ & $67.0$ & $99.0$ & $50.6$ & 4263 \\
ACTION.ARGUE & $84.0$ & $82.1$ & $73.2$ & $79.0$ & $68.2$ & 961 \\
ACTION.STUMBLE & $61.6$ & $50.6$ & $37.6$ & $100.0$ & $23.2$ & 837 \\
ACTION.OPEN_DOOR & $74.5$ & $80.1$ & $65.0$ & $96.4$ & $49.0$ & 1192 \\
ACTION.FALL & $95.8$ & $54.6$ & $50.2$ & $34.4$ & $92.4$ & 591 \\
ACTION.STAND_UP & $73.4$ & $81.0$ & $62.6$ & $94.5$ & $46.8$ & 873 \\
ACTION.FIGHT & $67.3$ & $55.0$ & $47.1$ & $73.8$ & $34.6$ & 990 \\
Total: mBAcc 81.91%, mAP: 80.41%, CF1: 70.37%, CP 86.08%, CR: 64.78%, OF1: 86.74%, OP 91.08%, OR: 82.79%
2021-09-02 20:34:54,543 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
  0%|          | 0/2539 [00:00<?, ?it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  0%|          | 1/2539 [00:00<19:52,  2.13it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
100%|██████████| 2539/2539 [00:18<00:00, 141.05it/s]
### GT_RESULT_RATIO: 1 (ehpi_3d_sim_c01_actionrec_gt_pred_ehpi2dvids_15fps)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $90.4$ & $96.6$ & $88.8$ & $94.8$ & $83.5$ & 45652 \\
ACTION.IDLE & $71.8$ & $79.7$ & $59.2$ & $89.8$ & $44.1$ & 10168 \\
ACTION.WALK & $93.3$ & $96.9$ & $92.9$ & $91.1$ & $94.8$ & 57699 \\
ACTION.JOG & $90.7$ & $93.1$ & $87.2$ & $92.4$ & $82.5$ & 16207 \\
ACTION.WAVE & $91.3$ & $83.1$ & $77.2$ & $71.7$ & $83.5$ & 2970 \\
ACTION.KICK_BALL & $77.9$ & $68.3$ & $68.7$ & $89.1$ & $55.9$ & 1144 \\
ACTION.THROW & $81.2$ & $87.6$ & $74.0$ & $90.7$ & $62.5$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $91.6$ & $93.4$ & $86.7$ & $88.9$ & $84.7$ & 14819 \\
ACTION.HITCHHIKE & $91.9$ & $92.9$ & $87.2$ & $90.7$ & $84.0$ & 4018 \\
ACTION.TURN_AROUND & $73.7$ & $65.5$ & $59.0$ & $76.8$ & $47.9$ & 3494 \\
ACTION.WORK & $75.2$ & $88.0$ & $66.8$ & $98.4$ & $50.5$ & 4263 \\
ACTION.ARGUE & $90.0$ & $87.8$ & $74.1$ & $68.8$ & $80.2$ & 961 \\
ACTION.STUMBLE & $68.0$ & $59.6$ & $52.6$ & $97.1$ & $36.1$ & 837 \\
ACTION.OPEN_DOOR & $55.3$ & $44.1$ & $18.6$ & $78.3$ & $10.6$ & 1192 \\
ACTION.FALL & $95.9$ & $67.6$ & $59.0$ & $43.3$ & $92.4$ & 591 \\
ACTION.STAND_UP & $80.5$ & $88.1$ & $74.6$ & $96.0$ & $60.9$ & 873 \\
ACTION.FIGHT & $72.5$ & $70.1$ & $60.2$ & $90.7$ & $45.1$ & 990 \\
Total: mBAcc 81.84%, mAP: 80.14%, CF1: 69.81%, CP 85.21%, CR: 64.66%, OF1: 86.23%, OP 90.86%, OR: 82.05%
2021-09-02 20:35:17,761 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
  0%|          | 0/2539 [00:00<?, ?it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
100%|██████████| 2539/2539 [00:19<00:00, 130.61it/s]
### GT_RESULT_RATIO: 1 (ehpi_3d_sim_c01_actionrec_gt_pred_64frames)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $90.8$ & $97.0$ & $89.3$ & $95.3$ & $84.1$ & 45652 \\
ACTION.IDLE & $71.9$ & $82.6$ & $59.5$ & $90.7$ & $44.2$ & 10168 \\
ACTION.WALK & $93.7$ & $97.5$ & $93.4$ & $92.1$ & $94.8$ & 57699 \\
ACTION.JOG & $92.7$ & $92.9$ & $88.6$ & $90.3$ & $86.9$ & 16207 \\
ACTION.WAVE & $93.0$ & $92.7$ & $83.0$ & $79.7$ & $86.5$ & 2970 \\
ACTION.KICK_BALL & $83.6$ & $79.3$ & $77.0$ & $90.1$ & $67.3$ & 1144 \\
ACTION.THROW & $94.1$ & $98.3$ & $93.2$ & $98.7$ & $88.3$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $92.8$ & $94.5$ & $88.4$ & $89.8$ & $87.0$ & 14819 \\
ACTION.HITCHHIKE & $90.8$ & $91.6$ & $86.5$ & $91.6$ & $81.9$ & 4018 \\
ACTION.TURN_AROUND & $70.2$ & $57.4$ & $52.9$ & $75.3$ & $40.7$ & 3494 \\
ACTION.WORK & $78.8$ & $85.8$ & $70.9$ & $91.8$ & $57.7$ & 4263 \\
ACTION.ARGUE & $92.9$ & $89.6$ & $79.1$ & $73.1$ & $86.1$ & 961 \\
ACTION.STUMBLE & $68.5$ & $45.7$ & $49.6$ & $74.8$ & $37.2$ & 837 \\
ACTION.OPEN_DOOR & $75.2$ & $88.9$ & $65.7$ & $94.3$ & $50.4$ & 1192 \\
ACTION.FALL & $89.7$ & $56.5$ & $54.9$ & $41.8$ & $79.9$ & 591 \\
ACTION.STAND_UP & $78.5$ & $80.0$ & $70.6$ & $92.6$ & $57.0$ & 873 \\
ACTION.FIGHT & $55.4$ & $35.9$ & $18.7$ & $65.5$ & $10.9$ & 990 \\
Total: mBAcc 83.10%, mAP: 80.37%, CF1: 71.83%, CP 83.97%, CR: 67.11%, OF1: 87.10%, OP 91.46%, OR: 83.14%
2021-09-02 20:35:42,443 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
  0%|          | 0/2539 [00:00<?, ?it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  0%|          | 1/2539 [00:00<19:37,  2.16it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  0%|          | 12/2539 [00:00<01:33, 27.13it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
100%|██████████| 2539/2539 [00:17<00:00, 141.51it/s]
### GT_RESULT_RATIO: 0 (ehpi_3d_sim_c01_actionrec_pred)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $89.0$ & $95.1$ & $87.0$ & $93.4$ & $81.4$ & 45652 \\
ACTION.IDLE & $62.9$ & $63.3$ & $39.9$ & $83.5$ & $26.2$ & 10168 \\
ACTION.WALK & $90.1$ & $95.0$ & $89.7$ & $88.5$ & $90.9$ & 57699 \\
ACTION.JOG & $87.4$ & $87.3$ & $79.9$ & $82.8$ & $77.3$ & 16207 \\
ACTION.WAVE & $86.1$ & $77.1$ & $69.0$ & $65.2$ & $73.2$ & 2970 \\
ACTION.KICK_BALL & $75.5$ & $68.3$ & $63.3$ & $83.3$ & $51.0$ & 1144 \\
ACTION.THROW & $70.9$ & $53.0$ & $54.3$ & $77.0$ & $41.9$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $87.7$ & $88.5$ & $81.0$ & $85.3$ & $77.2$ & 14819 \\
ACTION.HITCHHIKE & $89.6$ & $87.4$ & $79.1$ & $78.3$ & $80.0$ & 4018 \\
ACTION.TURN_AROUND & $74.7$ & $62.8$ & $60.8$ & $77.8$ & $49.9$ & 3494 \\
ACTION.WORK & $65.8$ & $66.6$ & $46.6$ & $87.1$ & $31.8$ & 4263 \\
ACTION.ARGUE & $69.3$ & $41.9$ & $47.1$ & $60.0$ & $38.7$ & 961 \\
ACTION.STUMBLE & $63.8$ & $34.4$ & $38.9$ & $65.4$ & $27.7$ & 837 \\
ACTION.OPEN_DOOR & $65.8$ & $61.2$ & $47.2$ & $92.6$ & $31.7$ & 1192 \\
ACTION.FALL & $79.9$ & $56.3$ & $49.4$ & $41.8$ & $60.2$ & 591 \\
ACTION.STAND_UP & $59.6$ & $39.9$ & $29.6$ & $63.1$ & $19.4$ & 873 \\
ACTION.FIGHT & $65.1$ & $44.4$ & $42.4$ & $70.0$ & $30.4$ & 990 \\
Total: mBAcc 75.49%, mAP: 66.01%, CF1: 59.14%, CP 76.18%, CR: 52.30%, OF1: 81.48%, OP 87.23%, OR: 76.44%
2021-09-02 20:36:05,644 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
  0%|          | 0/2539 [00:00<?, ?it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  0%|          | 1/2539 [00:00<20:01,  2.11it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  1%|          | 13/2539 [00:00<01:25, 29.42it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
100%|██████████| 2539/2539 [00:17<00:00, 142.71it/s]
### GT_RESULT_RATIO: 0 (ehpi_3d_sim_c01_actionrec_gt)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $78.7$ & $76.8$ & $73.5$ & $64.8$ & $84.9$ & 45652 \\
ACTION.IDLE & $63.2$ & $46.7$ & $39.3$ & $69.7$ & $27.4$ & 10168 \\
ACTION.WALK & $73.7$ & $80.1$ & $70.1$ & $77.6$ & $63.9$ & 57699 \\
ACTION.JOG & $65.9$ & $60.7$ & $47.0$ & $83.5$ & $32.7$ & 16207 \\
ACTION.WAVE & $74.0$ & $47.6$ & $49.6$ & $49.9$ & $49.2$ & 2970 \\
ACTION.KICK_BALL & $75.1$ & $14.0$ & $14.8$ & $8.6$ & $55.8$ & 1144 \\
ACTION.THROW & $56.9$ & $10.2$ & $15.8$ & $17.5$ & $14.5$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $78.2$ & $76.2$ & $67.9$ & $81.3$ & $58.3$ & 14819 \\
ACTION.HITCHHIKE & $76.0$ & $80.4$ & $67.0$ & $93.5$ & $52.2$ & 4018 \\
ACTION.TURN_AROUND & $73.3$ & $50.3$ & $52.4$ & $58.0$ & $47.7$ & 3494 \\
ACTION.WORK & $60.2$ & $29.9$ & $29.0$ & $44.8$ & $21.4$ & 4263 \\
ACTION.ARGUE & $73.5$ & $51.2$ & $55.6$ & $67.8$ & $47.1$ & 961 \\
ACTION.STUMBLE & $53.4$ & $21.3$ & $12.3$ & $54.7$ & $6.9$ & 837 \\
ACTION.OPEN_DOOR & $74.3$ & $46.2$ & $50.0$ & $50.8$ & $49.2$ & 1192 \\
ACTION.FALL & $71.4$ & $57.7$ & $53.6$ & $71.7$ & $42.8$ & 591 \\
ACTION.STAND_UP & $79.9$ & $91.8$ & $74.8$ & $100.0$ & $59.8$ & 873 \\
ACTION.FIGHT & $57.3$ & $13.2$ & $20.5$ & $33.3$ & $14.8$ & 990 \\
Total: mBAcc 69.71%, mAP: 50.26%, CF1: 46.67%, CP 60.45%, CR: 42.86%, OF1: 63.97%, OP 67.60%, OR: 60.71%
2021-09-02 20:36:28,677 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
  0%|          | 0/2539 [00:00<?, ?it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  0%|          | 1/2539 [00:00<19:59,  2.12it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  0%|          | 12/2539 [00:00<01:33, 27.05it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
100%|██████████| 2539/2539 [00:17<00:00, 142.57it/s]
### GT_RESULT_RATIO: 0 (ehpi_3d_sim_c01_actionrec_gt_pred)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $90.0$ & $95.3$ & $88.2$ & $93.3$ & $83.6$ & 45652 \\
ACTION.IDLE & $64.2$ & $59.7$ & $42.7$ & $81.2$ & $29.0$ & 10168 \\
ACTION.WALK & $90.2$ & $95.4$ & $89.6$ & $90.5$ & $88.8$ & 57699 \\
ACTION.JOG & $86.0$ & $87.4$ & $79.1$ & $85.1$ & $73.9$ & 16207 \\
ACTION.WAVE & $87.5$ & $85.8$ & $77.7$ & $80.0$ & $75.6$ & 2970 \\
ACTION.KICK_BALL & $81.5$ & $77.4$ & $71.9$ & $83.6$ & $63.0$ & 1144 \\
ACTION.THROW & $69.5$ & $59.2$ & $55.0$ & $93.0$ & $39.1$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $88.7$ & $89.2$ & $82.3$ & $85.7$ & $79.1$ & 14819 \\
ACTION.HITCHHIKE & $91.2$ & $88.8$ & $82.1$ & $81.0$ & $83.1$ & 4018 \\
ACTION.TURN_AROUND & $75.1$ & $66.1$ & $60.9$ & $76.2$ & $50.7$ & 3494 \\
ACTION.WORK & $75.1$ & $81.7$ & $65.0$ & $91.3$ & $50.4$ & 4263 \\
ACTION.ARGUE & $89.4$ & $78.3$ & $76.7$ & $74.4$ & $79.1$ & 961 \\
ACTION.STUMBLE & $65.3$ & $34.8$ & $41.5$ & $64.1$ & $30.7$ & 837 \\
ACTION.OPEN_DOOR & $65.6$ & $66.1$ & $46.8$ & $92.6$ & $31.3$ & 1192 \\
ACTION.FALL & $87.7$ & $58.7$ & $55.6$ & $43.9$ & $76.0$ & 591 \\
ACTION.STAND_UP & $58.7$ & $50.4$ & $27.8$ & $69.4$ & $17.4$ & 873 \\
ACTION.FIGHT & $63.5$ & $40.4$ & $39.6$ & $72.7$ & $27.2$ & 990 \\
Total: mBAcc 78.19%, mAP: 71.45%, CF1: 63.67%, CP 79.89%, CR: 57.52%, OF1: 82.60%, OP 88.77%, OR: 77.23%
2021-09-02 20:36:51,773 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
  0%|          | 0/2539 [00:00<?, ?it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  0%|          | 1/2539 [00:00<20:39,  2.05it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  1%|          | 17/2539 [00:00<01:07, 37.54it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
100%|██████████| 2539/2539 [00:18<00:00, 137.47it/s]
### GT_RESULT_RATIO: 0 (ehpi_3d_sim_c01_actionrec_gt_pred_ehpi2dvids)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $88.6$ & $94.7$ & $86.6$ & $93.2$ & $80.8$ & 45652 \\
ACTION.IDLE & $66.9$ & $67.5$ & $49.1$ & $85.8$ & $34.4$ & 10168 \\
ACTION.WALK & $90.2$ & $94.9$ & $89.7$ & $89.8$ & $89.6$ & 57699 \\
ACTION.JOG & $86.2$ & $87.3$ & $79.5$ & $85.3$ & $74.4$ & 16207 \\
ACTION.WAVE & $86.9$ & $80.7$ & $76.1$ & $78.1$ & $74.2$ & 2970 \\
ACTION.KICK_BALL & $77.1$ & $72.9$ & $66.4$ & $85.3$ & $54.4$ & 1144 \\
ACTION.THROW & $68.8$ & $56.9$ & $53.7$ & $93.9$ & $37.6$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $88.9$ & $89.2$ & $82.5$ & $85.5$ & $79.7$ & 14819 \\
ACTION.HITCHHIKE & $88.5$ & $87.4$ & $79.0$ & $80.4$ & $77.7$ & 4018 \\
ACTION.TURN_AROUND & $73.0$ & $63.1$ & $57.5$ & $75.6$ & $46.4$ & 3494 \\
ACTION.WORK & $70.0$ & $75.6$ & $55.7$ & $91.0$ & $40.1$ & 4263 \\
ACTION.ARGUE & $89.2$ & $81.5$ & $77.9$ & $77.1$ & $78.7$ & 961 \\
ACTION.STUMBLE & $65.4$ & $47.5$ & $44.1$ & $77.7$ & $30.8$ & 837 \\
ACTION.OPEN_DOOR & $67.6$ & $66.2$ & $51.6$ & $97.0$ & $35.2$ & 1192 \\
ACTION.FALL & $87.2$ & $65.0$ & $55.8$ & $44.5$ & $74.8$ & 591 \\
ACTION.STAND_UP & $60.5$ & $57.3$ & $34.0$ & $90.6$ & $21.0$ & 873 \\
ACTION.FIGHT & $65.7$ & $41.3$ & $44.1$ & $74.2$ & $31.4$ & 990 \\
Total: mBAcc 77.68%, mAP: 72.29%, CF1: 63.72%, CP 82.65%, CR: 56.52%, OF1: 82.22%, OP 88.64%, OR: 76.66%
2021-09-02 20:37:15,622 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
100%|██████████| 2539/2539 [00:18<00:00, 135.35it/s]
### GT_RESULT_RATIO: 0 (ehpi_3d_sim_c01_actionrec_gt_pred_no_unit_skeleton)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $88.1$ & $94.3$ & $85.9$ & $92.6$ & $80.1$ & 45652 \\
ACTION.IDLE & $67.9$ & $62.9$ & $51.6$ & $89.3$ & $36.2$ & 10168 \\
ACTION.WALK & $89.4$ & $94.4$ & $88.8$ & $89.0$ & $88.7$ & 57699 \\
ACTION.JOG & $86.5$ & $87.2$ & $79.9$ & $85.4$ & $75.1$ & 16207 \\
ACTION.WAVE & $82.0$ & $76.8$ & $71.6$ & $80.5$ & $64.4$ & 2970 \\
ACTION.KICK_BALL & $69.1$ & $55.0$ & $50.8$ & $75.3$ & $38.4$ & 1144 \\
ACTION.THROW & $62.8$ & $42.4$ & $39.6$ & $87.3$ & $25.6$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $89.0$ & $89.5$ & $82.7$ & $85.7$ & $79.8$ & 14819 \\
ACTION.HITCHHIKE & $92.4$ & $91.2$ & $82.5$ & $79.6$ & $85.5$ & 4018 \\
ACTION.TURN_AROUND & $74.7$ & $65.8$ & $60.5$ & $76.8$ & $49.9$ & 3494 \\
ACTION.WORK & $68.3$ & $65.3$ & $51.3$ & $84.2$ & $36.9$ & 4263 \\
ACTION.ARGUE & $85.0$ & $79.1$ & $74.3$ & $78.9$ & $70.2$ & 961 \\
ACTION.STUMBLE & $58.1$ & $28.3$ & $25.8$ & $62.1$ & $16.2$ & 837 \\
ACTION.OPEN_DOOR & $67.6$ & $68.9$ & $51.3$ & $95.2$ & $35.2$ & 1192 \\
ACTION.FALL & $89.3$ & $52.6$ & $56.4$ & $43.8$ & $79.2$ & 591 \\
ACTION.STAND_UP & $63.2$ & $57.1$ & $40.2$ & $84.6$ & $26.3$ & 873 \\
ACTION.FIGHT & $60.3$ & $34.1$ & $32.2$ & $71.9$ & $20.7$ & 990 \\
Total: mBAcc 76.11%, mAP: 67.33%, CF1: 60.31%, CP 80.13%, CR: 53.44%, OF1: 81.64%, OP 88.12%, OR: 76.06%
2021-09-02 20:37:39,608 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
  0%|          | 0/2539 [00:00<?, ?it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
100%|██████████| 2539/2539 [00:18<00:00, 133.67it/s]
### GT_RESULT_RATIO: 0 (ehpi_3d_sim_c01_actionrec_gt_pred_zero_by_score)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $89.0$ & $94.7$ & $86.9$ & $93.1$ & $81.5$ & 45652 \\
ACTION.IDLE & $64.9$ & $64.1$ & $44.8$ & $87.4$ & $30.1$ & 10168 \\
ACTION.WALK & $90.0$ & $95.0$ & $89.4$ & $89.7$ & $89.1$ & 57699 \\
ACTION.JOG & $86.5$ & $87.8$ & $79.8$ & $85.3$ & $75.0$ & 16207 \\
ACTION.WAVE & $85.1$ & $76.0$ & $70.7$ & $70.5$ & $71.0$ & 2970 \\
ACTION.KICK_BALL & $80.5$ & $76.0$ & $69.4$ & $80.3$ & $61.2$ & 1144 \\
ACTION.THROW & $70.8$ & $56.0$ & $55.6$ & $83.2$ & $41.7$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $87.2$ & $87.4$ & $80.4$ & $84.9$ & $76.4$ & 14819 \\
ACTION.HITCHHIKE & $90.7$ & $89.2$ & $79.9$ & $77.6$ & $82.3$ & 4018 \\
ACTION.TURN_AROUND & $74.7$ & $65.3$ & $59.7$ & $74.1$ & $49.9$ & 3494 \\
ACTION.WORK & $71.6$ & $74.1$ & $58.9$ & $91.8$ & $43.3$ & 4263 \\
ACTION.ARGUE & $88.0$ & $78.9$ & $74.8$ & $73.4$ & $76.3$ & 961 \\
ACTION.STUMBLE & $65.2$ & $42.0$ & $44.5$ & $83.6$ & $30.3$ & 837 \\
ACTION.OPEN_DOOR & $74.4$ & $76.2$ & $64.6$ & $95.9$ & $48.7$ & 1192 \\
ACTION.FALL & $90.5$ & $71.4$ & $59.2$ & $46.5$ & $81.6$ & 591 \\
ACTION.STAND_UP & $61.2$ & $73.8$ & $36.4$ & $96.6$ & $22.5$ & 873 \\
ACTION.FIGHT & $62.0$ & $36.4$ & $36.2$ & $73.2$ & $24.0$ & 990 \\
Total: mBAcc 78.37%, mAP: 73.20%, CF1: 64.20%, CP 81.59%, CR: 57.94%, OF1: 81.98%, OP 88.20%, OR: 76.58%
2021-09-02 20:38:03,982 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
  0%|          | 0/2539 [00:00<?, ?it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
100%|██████████| 2539/2539 [00:19<00:00, 133.44it/s]
### GT_RESULT_RATIO: 0 (ehpi_3d_sim_c01_actionrec_gt_pred_15fps)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $89.2$ & $95.0$ & $87.3$ & $93.0$ & $82.1$ & 45652 \\
ACTION.IDLE & $65.7$ & $62.7$ & $46.0$ & $81.9$ & $32.0$ & 10168 \\
ACTION.WALK & $89.5$ & $94.6$ & $89.0$ & $88.8$ & $89.1$ & 57699 \\
ACTION.JOG & $85.1$ & $85.8$ & $77.9$ & $84.5$ & $72.2$ & 16207 \\
ACTION.WAVE & $92.5$ & $90.9$ & $82.2$ & $79.2$ & $85.5$ & 2970 \\
ACTION.KICK_BALL & $76.2$ & $70.1$ & $66.2$ & $89.8$ & $52.4$ & 1144 \\
ACTION.THROW & $72.9$ & $66.7$ & $62.1$ & $96.1$ & $45.9$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $87.7$ & $89.2$ & $81.5$ & $86.4$ & $77.2$ & 14819 \\
ACTION.HITCHHIKE & $92.0$ & $89.3$ & $82.8$ & $80.9$ & $84.7$ & 4018 \\
ACTION.TURN_AROUND & $75.0$ & $66.0$ & $61.5$ & $78.9$ & $50.3$ & 3494 \\
ACTION.WORK & $74.4$ & $85.7$ & $64.7$ & $95.6$ & $48.9$ & 4263 \\
ACTION.ARGUE & $74.5$ & $64.9$ & $60.1$ & $77.3$ & $49.1$ & 961 \\
ACTION.STUMBLE & $55.9$ & $40.1$ & $20.7$ & $83.9$ & $11.8$ & 837 \\
ACTION.OPEN_DOOR & $82.5$ & $85.1$ & $76.5$ & $93.0$ & $65.0$ & 1192 \\
ACTION.FALL & $92.6$ & $58.6$ & $50.6$ & $35.9$ & $86.0$ & 591 \\
ACTION.STAND_UP & $59.8$ & $73.7$ & $32.4$ & $94.5$ & $19.6$ & 873 \\
ACTION.FIGHT & $67.0$ & $37.9$ & $41.4$ & $52.5$ & $34.2$ & 990 \\
Total: mBAcc 78.38%, mAP: 73.90%, CF1: 63.70%, CP 81.89%, CR: 58.01%, OF1: 82.14%, OP 88.00%, OR: 77.02%
2021-09-02 20:38:28,410 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
  0%|          | 0/2539 [00:00<?, ?it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  0%|          | 1/2539 [00:00<20:20,  2.08it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
100%|██████████| 2539/2539 [00:19<00:00, 131.48it/s]
### GT_RESULT_RATIO: 0 (ehpi_3d_sim_c01_actionrec_gt_pred_ehpi2dvids_15fps)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $88.4$ & $94.6$ & $86.2$ & $92.5$ & $80.6$ & 45652 \\
ACTION.IDLE & $68.3$ & $66.0$ & $51.3$ & $82.3$ & $37.3$ & 10168 \\
ACTION.WALK & $89.3$ & $94.7$ & $88.8$ & $88.8$ & $88.8$ & 57699 \\
ACTION.JOG & $86.2$ & $86.6$ & $79.2$ & $84.6$ & $74.5$ & 16207 \\
ACTION.WAVE & $89.0$ & $87.5$ & $80.1$ & $81.9$ & $78.4$ & 2970 \\
ACTION.KICK_BALL & $72.8$ & $62.9$ & $60.0$ & $87.2$ & $45.7$ & 1144 \\
ACTION.THROW & $72.5$ & $65.7$ & $60.6$ & $92.8$ & $45.0$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $88.3$ & $89.1$ & $81.9$ & $85.6$ & $78.5$ & 14819 \\
ACTION.HITCHHIKE & $92.3$ & $90.2$ & $83.5$ & $81.8$ & $85.2$ & 4018 \\
ACTION.TURN_AROUND & $72.2$ & $60.6$ & $56.2$ & $75.5$ & $44.8$ & 3494 \\
ACTION.WORK & $74.6$ & $85.9$ & $65.0$ & $95.5$ & $49.3$ & 4263 \\
ACTION.ARGUE & $76.8$ & $61.1$ & $55.5$ & $57.3$ & $53.9$ & 961 \\
ACTION.STUMBLE & $58.4$ & $41.5$ & $27.8$ & $79.7$ & $16.8$ & 837 \\
ACTION.OPEN_DOOR & $83.1$ & $84.1$ & $76.9$ & $91.6$ & $66.4$ & 1192 \\
ACTION.FALL & $93.3$ & $55.5$ & $55.5$ & $40.7$ & $87.1$ & 591 \\
ACTION.STAND_UP & $66.6$ & $84.7$ & $49.7$ & $98.3$ & $33.2$ & 873 \\
ACTION.FIGHT & $61.1$ & $36.8$ & $33.0$ & $63.5$ & $22.3$ & 990 \\
Total: mBAcc 78.42%, mAP: 73.38%, CF1: 64.20%, CP 81.15%, CR: 58.11%, OF1: 82.02%, OP 87.83%, OR: 76.93%
2021-09-02 20:38:52,948 pedrec.utils.torch_utils.torch_helper INFO     Working on GPU: NVIDIA GeForce RTX 3080!
  0%|          | 0/2539 [00:00<?, ?it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
  0%|          | 1/2539 [00:00<20:14,  2.09it/s]/home/dennis/code/python/pedrec/pedrec/utils/skeleton_helper_3d.py:41: RuntimeWarning: invalid value encountered in true_divide
  normalized_direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
100%|██████████| 2539/2539 [00:19<00:00, 130.32it/s]
### GT_RESULT_RATIO: 0 (ehpi_3d_sim_c01_actionrec_gt_pred_64frames)

Action & BAcc & AP & F1 & Precision & Recall & NPS \\\midrule
ACTION.STAND & $89.5$ & $95.0$ & $87.5$ & $93.1$ & $82.6$ & 45652 \\
ACTION.IDLE & $68.0$ & $65.3$ & $51.5$ & $87.2$ & $36.6$ & 10168 \\
ACTION.WALK & $90.8$ & $95.3$ & $90.3$ & $90.0$ & $90.7$ & 57699 \\
ACTION.JOG & $87.8$ & $87.3$ & $81.6$ & $86.0$ & $77.5$ & 16207 \\
ACTION.WAVE & $92.2$ & $92.6$ & $85.1$ & $85.4$ & $84.8$ & 2970 \\
ACTION.KICK_BALL & $85.6$ & $84.8$ & $79.6$ & $90.3$ & $71.2$ & 1144 \\
ACTION.THROW & $79.0$ & $82.7$ & $73.2$ & $99.3$ & $58.0$ & 1024 \\
ACTION.LOOK_FOR_TRAFFIC & $89.3$ & $91.1$ & $83.3$ & $86.4$ & $80.4$ & 14819 \\
ACTION.HITCHHIKE & $92.1$ & $92.1$ & $83.7$ & $82.6$ & $84.8$ & 4018 \\
ACTION.TURN_AROUND & $70.8$ & $56.1$ & $54.3$ & $76.8$ & $42.0$ & 3494 \\
ACTION.WORK & $75.1$ & $87.8$ & $66.1$ & $96.7$ & $50.2$ & 4263 \\
ACTION.ARGUE & $85.6$ & $80.1$ & $75.2$ & $79.5$ & $71.3$ & 961 \\
ACTION.STUMBLE & $65.3$ & $48.7$ & $43.9$ & $76.7$ & $30.7$ & 837 \\
ACTION.OPEN_DOOR & $87.9$ & $94.1$ & $85.4$ & $97.6$ & $75.8$ & 1192 \\
ACTION.FALL & $88.7$ & $60.5$ & $57.6$ & $45.7$ & $77.8$ & 591 \\
ACTION.STAND_UP & $69.5$ & $83.5$ & $55.9$ & $98.8$ & $38.9$ & 873 \\
ACTION.FIGHT & $59.4$ & $43.1$ & $29.8$ & $71.8$ & $18.8$ & 990 \\
Total: mBAcc 80.98%, mAP: 78.83%, CF1: 69.65%, CP 84.94%, CR: 63.08%, OF1: 83.91%, OP 89.35%, OR: 79.09%
| Model | GT/Result Ratio | Balanced Acc | mAP | OF1 | OP | OR | CF1 | CP | CR
| P | 1 | 73.21 | 58.60 | 79.16  | 86.33  | 73.09 | 52.93  | 71.27  | 47.76
| G | 1 | 79.64 | 74.65 | 85.80  | 89.68  | 82.25 | 67.09  | 80.87  | 60.40
| G+P | 1 | 81.22 | 77.33 | 86.22  | 91.07  | 81.85 | 68.82  | 81.87  | 63.38
| G+P+E | 1 | 80.56 | 77.52 | 85.90  | 90.90  | 81.43 | 68.14  | 82.92  | 62.08
| G+P+Neton | 1 | 80.01 | 76.53 | 85.86  | 90.41  | 81.74 | 66.75  | 83.82  | 61.02
| G+P+zero+by+score | 1 | 81.57 | 79.18 | 86.48  | 91.32  | 82.12 | 70.59  | 84.99  | 64.07
| G+P+15fps | 1 | 81.91 | 80.41 | 86.74  | 91.08  | 82.79 | 70.37  | 86.08  | 64.78
| G+P+E+15fps | 1 | 81.84 | 80.14 | 86.23  | 90.86  | 82.05 | 69.81  | 85.21  | 64.66
| G+P+64frames | 1 | 83.10 | 80.37 | 87.10  | 91.46  | 83.14 | 71.83  | 83.97  | 67.11
| P | 0 | 75.49 | 66.01 | 81.48  | 87.23  | 76.44 | 59.14  | 76.18  | 52.30
| G | 0 | 69.71 | 50.26 | 63.97  | 67.60  | 60.71 | 46.67  | 60.45  | 42.86
| G+P | 0 | 78.19 | 71.45 | 82.60  | 88.77  | 77.23 | 63.67  | 79.89  | 57.52
| G+P+E | 0 | 77.68 | 72.29 | 82.22  | 88.64  | 76.66 | 63.72  | 82.65  | 56.52
| G+P+Neton | 0 | 76.11 | 67.33 | 81.64  | 88.12  | 76.06 | 60.31  | 80.13  | 53.44
| G+P+zero+by+score | 0 | 78.37 | 73.20 | 81.98  | 88.20  | 76.58 | 64.20  | 81.59  | 57.94
| G+P+15fps | 0 | 78.38 | 73.90 | 82.14  | 88.00  | 77.02 | 63.70  | 81.89  | 58.01
| G+P+E+15fps | 0 | 78.42 | 73.38 | 82.02  | 87.83  | 76.93 | 64.20  | 81.15  | 58.11
| G+P+64frames | 0 | 80.98 | 78.83 | 83.91  | 89.35  | 79.09 | 69.65  | 84.94  | 63.08
Model & G & mBAcc & mAP & OF1 & OP & OR & CF1 & CP & CR \\\midrule
P & $1$ &  $73.2$ & $58.6$ & $79.2$ & $86.3$ & $73.1$ & $52.9$ & $71.3$ & $47.8$ \\
G & $1$ &  $79.6$ & $74.6$ & $85.8$ & $89.7$ & $82.2$ & $67.1$ & $80.9$ & $60.4$ \\
G+P & $1$ &  $81.2$ & $77.3$ & $86.2$ & $91.1$ & $81.9$ & $68.8$ & $81.9$ & $63.4$ \\
G+P+E & $1$ &  $80.6$ & $77.5$ & $85.9$ & $90.9$ & $81.4$ & $68.1$ & $82.9$ & $62.1$ \\
G+P+Neton & $1$ &  $80.0$ & $76.5$ & $85.9$ & $90.4$ & $81.7$ & $66.7$ & $83.8$ & $61.0$ \\
G+P+zero+by+score & $1$ &  $81.6$ & $79.2$ & $86.5$ & $91.3$ & $82.1$ & $70.6$ & $85.0$ & $64.1$ \\
G+P+15fps & $1$ &  $81.9$ & $80.4$ & $86.7$ & $91.1$ & $82.8$ & $70.4$ & $86.1$ & $64.8$ \\
G+P+E+15fps & $1$ &  $81.8$ & $80.1$ & $86.2$ & $90.9$ & $82.0$ & $69.8$ & $85.2$ & $64.7$ \\
G+P+64frames & $1$ &  $83.1$ & $80.4$ & $87.1$ & $91.5$ & $83.1$ & $71.8$ & $84.0$ & $67.1$ \\
P & $0$ &  $75.5$ & $66.0$ & $81.5$ & $87.2$ & $76.4$ & $59.1$ & $76.2$ & $52.3$ \\
G & $0$ &  $69.7$ & $50.3$ & $64.0$ & $67.6$ & $60.7$ & $46.7$ & $60.5$ & $42.9$ \\
G+P & $0$ &  $78.2$ & $71.4$ & $82.6$ & $88.8$ & $77.2$ & $63.7$ & $79.9$ & $57.5$ \\
G+P+E & $0$ &  $77.7$ & $72.3$ & $82.2$ & $88.6$ & $76.7$ & $63.7$ & $82.7$ & $56.5$ \\
G+P+Neton & $0$ &  $76.1$ & $67.3$ & $81.6$ & $88.1$ & $76.1$ & $60.3$ & $80.1$ & $53.4$ \\
G+P+zero+by+score & $0$ &  $78.4$ & $73.2$ & $82.0$ & $88.2$ & $76.6$ & $64.2$ & $81.6$ & $57.9$ \\
G+P+15fps & $0$ &  $78.4$ & $73.9$ & $82.1$ & $88.0$ & $77.0$ & $63.7$ & $81.9$ & $58.0$ \\
G+P+E+15fps & $0$ &  $78.4$ & $73.4$ & $82.0$ & $87.8$ & $76.9$ & $64.2$ & $81.2$ & $58.1$ \\
G+P+64frames & $0$ &  $81.0$ & $78.8$ & $83.9$ & $89.4$ & $79.1$ & $69.6$ & $84.9$ & $63.1$ \\

Process finished with exit code 0
