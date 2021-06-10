import cv2
import numpy as np


video = cv2.VideoCapture(0)


while True:
    _, frame = video.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV frame', frame)

    k = cv2.waitKey(0) & 0xFF == ord('q')
    if k == 27:
        break

cv2.destroyAllWindows()
video.release()
