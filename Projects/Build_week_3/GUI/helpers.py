import mediapipe as mp
import cv2
import numpy as np


def hand_center(h_landmarks, img):

    x = [landmark.x for landmark in h_landmarks.landmark]
    y = [landmark.y for landmark in h_landmarks.landmark]

    center = np.array([np.mean(x) * img.shape[1], np.mean(y) * img.shape[0]]).astype('int32')

    cv2.circle(img, tuple(center), 10, (255, 0, 0), 2)  # for checking the center

    return center