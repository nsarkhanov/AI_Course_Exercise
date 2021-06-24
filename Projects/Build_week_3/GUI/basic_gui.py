from pet_food import Pet_food
import mediapipe as mp
import cv2
import numpy as np
import helpers as hp

# Media_pipe requirements
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Access to webcam
vc = cv2.VideoCapture(0)
if vc.isOpened():
    response, frame = vc.read()

else:
    response = False

bone = Pet_food((1, 100), 'bone', 0)
carrot = Pet_food((1, 200), 'carrot', 1)
concrete = Pet_food((0, 400), 'concrete', 2)
worm = Pet_food((0, 300), 'worm', 3)



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
            for hand_landmarks in results.multi_hand_landmarks:
                # print(f'landmark {hand_landmarks}')
                list_land = hand_landmarks.landmark
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hc = hp.hand_center(hand_landmarks, image)
                # print(hc)
        # img = cv2.circle(image, (100, 100), 20, (0, 255, 0), -1)

        img = cv2.circle(image, (0, 0), 20, (0, 255, 0), -1)
        img = cv2.circle(image, (0, image.shape[0]), 20, (0, 0, 255), -1)
        img = cv2.circle(image, (image.shape[1], 0), 20, (255, 0, 0), -1)
        img = cv2.circle(image, (image.shape[1], image.shape[0]), 20, (255, 255, 255), -1)


        bone.move_food()
        carrot.move_food()
        concrete.move_food()
        worm.move_food()

        img = bone.draw_food(image)
        img = carrot.draw_food(image)
        img = concrete.draw_food(image)
        img = worm.draw_food(image)

        bone.detect_edge(img)
        carrot.detect_edge(img)
        concrete.detect_edge(img)
        worm.detect_edge(img)

        

        bone.hand_collision(hc)
        carrot.hand_collision(hc)
        concrete.hand_collision(hc)
        worm.hand_collision(hc)



        
        cv2.imshow('test', img)
        key = cv2.waitKey(1)
        if key == 27:
            break

vc.release()
cv2.destroyAllWindows()

# video = cv2.VideoCapture(0)

# while(video.isOpened()):
#     check, frame = video.read()
#     if frame is not None:
#         img = cv2.circle(frame, (100, 100), 20, (0, 255, 0), -1)
#         #img = frame
#         cv2.imshow('frame',img)        
#         if cv2.waitKey(30) & 0xFF == ord('q'):
#             break
#     else:
#         break


# video.release()
# cv2.destroyAllWindows()
# cv2.waitKey(1)


