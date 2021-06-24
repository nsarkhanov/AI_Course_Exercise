import cv2
import numpy as np
import mediapipe as mp


blank_image = np.zeros(shape=[512, 512, 3], dtype=np.uint8)

cv2.imshow("White Blank", blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
