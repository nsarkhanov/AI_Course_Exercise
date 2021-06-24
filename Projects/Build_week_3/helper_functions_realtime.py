import cv2
import numpy as np
from PIL import Image


def add_dim(image):

    kernel = (3, 3)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    inverted = 255 - gray

    blurred = cv2.GaussianBlur(inverted, kernel, 0)

    canny = cv2.Canny(blurred, 100, 200)

    closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    dilated = cv2.dilate(closing, kernel, iterations=4)

    dilated = dilated.reshape(255, 255, 1)

    stacked = np.dstack((image, dilated))

    return stacked
