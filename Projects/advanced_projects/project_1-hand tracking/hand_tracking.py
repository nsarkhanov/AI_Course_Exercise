import cv2
import mediapipe as mp
import time


video = cv2.VideoCapture(0)


while True:
    _, frame = video.read()

    cv2.imshow('video', frame)

    if 0xFF == ord('q'):
        break
    cv2.waitKey(0)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
