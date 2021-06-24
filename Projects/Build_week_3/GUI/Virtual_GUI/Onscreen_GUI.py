
import mediapipe as mp
import cv2
import numpy as np
from menu_buttons import buttons
import time


# Media_pipe requirements
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Access to webcam
vc = cv2.VideoCapture(0)
if vc.isOpened():
    response, frame = vc.read()

else:
    response = False

Start = buttons(frame.shape, 'Start')
How_to_play = buttons(frame.shape, 'How to play')
Exit = buttons(frame.shape, 'Exit')

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while response:
        response, frame = vc.read()
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        list_landmarks = []
        if results.multi_hand_landmarks:
            image_height, image_width, maybe_depth = image.shape
            for hand_landmarks in results.multi_hand_landmarks:
                #print(f'landmark {hand_landmarks}')
                list_land = hand_landmarks.landmark
                # print(
                #     f'Index finger tip coordinates: (',
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height},'
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * maybe_depth})'
                # )
                index_fing_x, index_fing_y = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                print(index_fing_x, index_fing_y)

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # # draw small circles on each corner
        # img = cv2.circle(image, (0, 0), 20, (0, 255, 0), -1)
        # img = cv2.circle(image, (0, image.shape[0]), 20, (0, 0, 255), -1)
        # img = cv2.circle(image, (image.shape[1], 0), 20, (255, 0, 0), -1)
        # img = cv2.circle(image, (image.shape[1], image.shape[0]), 20, (255, 255, 255), -1)

                Start.draw_button(image)
                How_to_play.draw_button(image)
                Exit.draw_button(image)


                # if 270>=index_fing_x<=365 and 95<=index_fing_y>=145:
                #     print('its working')
                #     time.sleep(2)

        cv2.imshow('test', image)
        key = cv2.waitKey(1)
        if key == 27:
            break

vc.release()
cv2.destroyAllWindows()