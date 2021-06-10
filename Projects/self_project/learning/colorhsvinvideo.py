import cv2
import numpy as np


video = cv2.VideoCapture(0)


while True:
    _, frame = video.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV frame', hsv)
