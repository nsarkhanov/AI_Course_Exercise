import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def imshow(img):
    plt.figure(figsize=(20, 15))
    plt.imshow(img)


# blue image  for 0
hist_blue = cv.calcHist([img], [0], None, [255], [0, 255])
plt.plot(hist_blue, color='b')

# green image

hist_blue = cv.calcHist([img], [1], None, [255], [0, 255])
plt.plot(hist_blue, color='b')

# red  image  for 0
hist_blue = cv.calcHist([img], [2], None, [255], [0, 255])
plt.plot(hist_blue, color='b')


img = cv2.warpAffine(img, translatein_matrix, (w, b))
cv.resize()
