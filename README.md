# RetinaFace
Portable RetinaFace in Pytorch. Easy to use as loss function for bounding boxes and facial landmark.

**Prerequisites:**

- Python 3.x with argparse, torch, and PIL (Pillow)
- RetinaFaceDetector class (in module 'detector')
- RetinaFaceLoss class (in module 'loss')
- Trained RetinaFace model (default path: './weights/Resnet50_Final.pth')

**Instructions for running run_loss.py:**

**Steps:**

1. Open a terminal.
2. Navigate to the directory containing `run_loss.py`.
3. Run:

```bash
   python run_loss.py [options]
```

```bash
python run_loss.py --help
usage: run_loss.py [-h] [-p_A IMAGE_PATH_A] [-p_B IMAGE_PATH_B] [-m TRAINED_MODEL] [--network NETWORK] [--gpu] [--confidence_threshold CONFIDENCE_THRESHOLD] [--top_k TOP_K] [--nms_threshold NMS_THRESHOLD]
                   [--keep_top_k KEEP_TOP_K]


optional arguments:
  -h, --help            show this help message and exit
  -p_A IMAGE_PATH_A, --image_path_A IMAGE_PATH_A
                        Path to image A
  -p_B IMAGE_PATH_B, --image_path_B IMAGE_PATH_B
                        Path to image B
  -m TRAINED_MODEL, --trained_model TRAINED_MODEL
                        Path to trained model
  --network NETWORK     Backbone network mobile0.25 or resnet50
  --gpu                 Use gpu inference
  --confidence_threshold CONFIDENCE_THRESHOLD
                        confidence_threshold
  --top_k TOP_K         top_k
  --nms_threshold NMS_THRESHOLD
                        nms_threshold
  --keep_top_k KEEP_TOP_K
                        keep_top_k

```

**Instructions for running run_detector.py:**

**Steps:**

1. Open a terminal.
2. Navigate to the directory containing `run_detector.py`.
3. Run:

```bash
   python run_detector.py [options]

```

```
python run_detector.py --help
usage: run_detector.py [-h] [-p IMAGE_PATH] [-m TRAINED_MODEL] [--network NETWORK] [--gpu] [--confidence_threshold CONFIDENCE_THRESHOLD] [--top_k TOP_K] [--nms_threshold NMS_THRESHOLD] [--keep_top_k KEEP_TOP_K] [-s]
                       [--vis_thres VIS_THRES]

Retinaface

optional arguments:
  -h, --help            show this help message and exit
  -p IMAGE_PATH, --image_path IMAGE_PATH
                        Trained state_dict file path to open
  -m TRAINED_MODEL, --trained_model TRAINED_MODEL
                        Trained state_dict file path to open
  --network NETWORK     Backbone network mobile0.25 or resnet50
  --gpu                 Use gpu inference
  --confidence_threshold CONFIDENCE_THRESHOLD
                        confidence_threshold
  --top_k TOP_K         top_k
  --nms_threshold NMS_THRESHOLD
                        nms_threshold
  --keep_top_k KEEP_TOP_K
                        keep_top_k
  -s, --save_image      show detection results
  --vis_thres VIS_THRES
                        visualization_threshold
```