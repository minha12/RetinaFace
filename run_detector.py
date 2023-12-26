import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from detector import (
    RetinaFaceDetector,
)  # Assuming the class is in a module named 'retinaface'

parser = argparse.ArgumentParser(description="Retinaface")
parser.add_argument(
    "-p",
    "--image_path",
    default="./test_imgs/00044.jpg",
    type=str,
    help="Trained state_dict file path to open",
)
parser.add_argument(
    "-m",
    "--trained_model",
    default="./weights/Resnet50_Final.pth",
    type=str,
    help="Trained state_dict file path to open",
)
parser.add_argument(
    "--network", default="resnet50", help="Backbone network mobile0.25 or resnet50"
)
parser.add_argument(
    "--gpu", action="store_true", default=True, help="Use gpu inference"
)
parser.add_argument(
    "--confidence_threshold", default=0.02, type=float, help="confidence_threshold"
)
parser.add_argument("--top_k", default=5000, type=int, help="top_k")
parser.add_argument("--nms_threshold", default=0.4, type=float, help="nms_threshold")
parser.add_argument("--keep_top_k", default=750, type=int, help="keep_top_k")
parser.add_argument(
    "-s",
    "--save_image",
    action="store_true",
    default=True,
    help="show detection results",
)
parser.add_argument(
    "--vis_thres", default=0.6, type=float, help="visualization_threshold"
)
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
image, im_height, im_width, scale = detector.preprocess_image("./test_imgs/00044.jpg")

# Detect faces
dets = detector.detect_faces(image)

# print("dets: ", dets)

# show image
if args.save_image:
    img_raw = Image.open(args.image_path)
    draw = ImageDraw.Draw(img_raw)
    # font = ImageFont.truetype("arial.ttf", 12)  # Adjust font path and size as needed

    font = ImageFont.load_default()

    for b in dets:
        if b[4] < args.vis_thres:
            continue

        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        draw.rectangle((b[0], b[1], b[2], b[3]), fill=None, outline="red", width=2)
        draw.text((b[0], b[1] + 12), text, font=font, fill="white")

        # Landmarks
        draw.ellipse((b[5] - 2, b[6] - 2, b[5] + 2, b[6] + 2), fill="red")  # Right eye
        draw.ellipse(
            (b[7] - 2, b[8] - 2, b[7] + 2, b[8] + 2), fill="yellow"
        )  # Left eye
        draw.ellipse((b[9] - 2, b[10] - 2, b[9] + 2, b[10] + 2), fill="magenta")  # Nose
        draw.ellipse(
            (b[11] - 2, b[12] - 2, b[11] + 2, b[12] + 2), fill="green"
        )  # Left mouth
        draw.ellipse(
            (b[13] - 2, b[14] - 2, b[13] + 2, b[14] + 2), fill="blue"
        )  # Right mouth

    img_raw.save("test.jpg")
