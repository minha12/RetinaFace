import torch
from detector import RetinaFaceDetector


class RetinaFaceLoss:
    def __init__(self):
        self.detector = RetinaFaceDetector()
        self.smooth_l1_loss = torch.nn.SmoothL1Loss()
        self.mse_loss = torch.nn.MSELoss()

    def bounding_box_loss(self, image1, image2):
        # Get bounding boxes for both images
        dets1 = self.detector.detect_faces(image1)
        dets2 = self.detector.detect_faces(image2)

        # Extract bounding box coordinates
        boxes1 = dets1[:, :4]
        boxes2 = dets2[:, :4]
        # Calculate Smooth L1 loss between corresponding boxes
        loss = self.smooth_l1_loss(boxes1, boxes2)
        return loss

    def landmark_loss(self, image1, image2):
        # Get facial landmarks for both images
        dets1 = self.detector.detect_faces(image1)
        dets2 = self.detector.detect_faces(image2)

        # Extract landmark coordinates
        landmarks1 = dets1[:, 5:]
        landmarks2 = dets2[:, 5:]
        # Calculate MSE loss between corresponding landmarks
        loss = self.mse_loss(landmarks1, landmarks2)
        return loss
