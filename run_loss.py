import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from detector import (
    RetinaFaceDetector,
)  # Assuming the class is in a module named 'retinaface'
from loss import RetinaFaceLoss

parser = argparse.ArgumentParser(description="Retinaface")
parser.add_argument(
    "-p_A",
    "--image_path_A",
    default="./data/A.jpg",
    type=str,
    help="Path to image A",
)

parser.add_argument(
    "-p_B",
    "--image_path_B",
    default="./data/B.jpg",
    type=str,
    help="Path to image B",
)

parser.add_argument(
    "-m",
    "--trained_model",
    default="./weights/Resnet50_Final.pth",
    type=str,
    help="Path to trained model",
)
parser.add_argument(
    "--network", default="resnet50", help="Backbone network mobile0.25 or resnet50"
)
parser.add_argument(
    "--gpu", action="store_true", default=False, help="Use gpu inference"
)
parser.add_argument(
    "--confidence_threshold", default=0.02, type=float, help="confidence_threshold"
)
parser.add_argument("--top_k", default=5000, type=int, help="top_k")
parser.add_argument("--nms_threshold", default=0.4, type=float, help="nms_threshold")
parser.add_argument("--keep_top_k", default=750, type=int, help="keep_top_k")

args = parser.parse_args()

detector = RetinaFaceDetector(
    trained_model=args.trained_model,
    network=args.network,
    gpu=args.gpu,
    confidence_threshold=args.confidence_threshold,
    top_k=args.top_k,
    nms_threshold=args.nms_threshold,
    keep_top_k=args.keep_top_k,
)
# Preprocess image
img_A, _, _, _ = detector.preprocess_image("./data/A.jpg")
img_B, _, _, _ = detector.preprocess_image("./data/B.jpg")

# Bounding box loss
loss = RetinaFaceLoss()
bbox_loss = loss.bounding_box_loss(img_A, img_B)
print("Bounding box loss: ", bbox_loss)

# Landmark loss
landmark_loss = loss.landmark_loss(img_A, img_B)
print("Landmark loss: ", landmark_loss)
