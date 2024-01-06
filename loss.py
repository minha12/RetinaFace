import os
import sys
import torch
from detector import RetinaFaceDetector

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(project_root)


class RetinaFaceLoss:
    def __init__(self):
        self.detector = RetinaFaceDetector()
        self.smooth_l1_loss = torch.nn.SmoothL1Loss()
        self.mse_loss = torch.nn.MSELoss(reduction="mean")

    def bounding_box_loss(self, image1, image2):
        # Get bounding boxes for both images
        dets1 = self.detector.detect_faces(image1, need_preprocess=True)
        dets2 = self.detector.detect_faces(image2, need_preprocess=True)

        # Extract bounding box coordinates
        boxes1 = dets1[:, :4]
        boxes2 = dets2[:, :4]
        # Calculate Smooth L1 loss between corresponding boxes
        loss = self.mse_loss(boxes1, boxes2)
        return loss

    def shift_landmarks(self, landmarks, x_shift, y_shift):
        """
        Shifts the landmarks by the specified x and y values.

        Parameters:
        landmarks (Tensor): A tensor of shape [1, 10] representing the facial landmarks.
        x_shift (int or float): The value by which to shift the x-coordinates.
        y_shift (int or float): The value by which to shift the y-coordinates.

        Returns:
        Tensor: The shifted landmarks tensor.
        """
        for i in range(0, landmarks.size(1), 2):
            landmarks[0][i] += x_shift  # Shift x-coordinate
            landmarks[0][i + 1] += y_shift  # Shift y-coordinate

        return landmarks

    def shift_loss(self, x, y, shift_x=0, shift_y=0):
        """
        Calculates the landmark loss between x and y, where y is shifted by the specified amounts.
        """
        x_feats = self.detector.detect_faces(x, need_preprocess=True)[:, 5:]
        y_feats = self.detector.detect_faces(y, need_preprocess=True)[:, 5:]
        y_shifted = self.shift_landmarks(y_feats, shift_x, shift_y)
        loss = self.mse_loss(x_feats, y_shifted)
        return loss

    def landmark_loss(self, image1, image2):
        # Get facial landmarks for both images
        dets1 = self.detector.detect_faces(image1, need_preprocess=True)
        dets2 = self.detector.detect_faces(image2, need_preprocess=True)
        # print("image1 shape ", image1.shape)
        # Extract landmark coordinates
        landmarks1 = dets1[:, 5:]
        # landmarks1 = dets1
        # print("landmarks1", landmarks1)
        landmarks2 = dets2[:, 5:]
        # landmarks2 = dets2
        # print("landmarks2", landmarks2)
        # Calculate MSE loss between corresponding landmarks
        loss = self.smooth_l1_loss(landmarks1, landmarks2)
        return loss
