
from imp import lock_held
import cv2
import numpy as np


def nothing(x):
    pass


cv2.namedWindow('Trackbars')
cv2.createTrackbar('L-H', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('L-S', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('L-V', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('U-H', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('U-S', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('U-V', 'Trackbars', 0, 255, nothing)


video = cv2.VideoCapture(0)
while True:
    _, frame = video.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos('L-H', 'Trackbars')
    l_s = cv2.getTrackbarPos('L-S', 'Trackbars')
    l_v = cv2.getTrackbarPos('L-V', 'Trackbars')
    u_h = cv2.getTrackbarPos('U-H', 'Trackbars')
    u_s = cv2.getTrackbarPos('U-S', 'Trackbars')
    u_v = cv2.getTrackbarPos('U-V', 'Trackbars')

    lower_red = np.array([l_h, l_s, l_v])

    upper_red = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    median = cv2.medianBlur(result, 15)

    cv2.imshow('Frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    cv2.imshow('median', median)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
video.release()
