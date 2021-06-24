import cv2
import mediapipe as mp
from helper_functions_realtime import add_dim

import torch
import torch.nn.functional as F
from cnn_model import CNN
import data_handler_CNN as dhc
from processing_df_image import bbox_landmarks

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

examples = dhc.train_data.class_to_idx
inv_map = {v: k for k, v in examples.items()}

model = torch.load('saved_models/best_acc_model.pth', map_location='cpu')
model = model.eval()

vc = cv2.VideoCapture(0)
if vc.isOpened():
    response, frame = vc.read()

else:
    response = False
if response:
    image_height, image_width, _ = frame.shape

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while response:
        response, frame = vc.read()
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                list_land = hand_landmarks.landmark
                #mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                crop = bbox_landmarks(hand_landmarks, image)
                bbox_landmarks(hand_landmarks, image)
                try:
                    crop = cv2.resize(crop, (255, 255), interpolation=cv2.INTER_AREA)
                    crop = add_dim(crop)
                    crop = F.normalize(torch.from_numpy(crop).float())
                    outputs = model(crop.view(-1, 4, 255, 255))
                    probs = F.softmax(outputs)
                    pred = torch.argmax(probs)
                    print(inv_map[pred.item()], probs)
                    cv2.putText(image, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                except:
                    continue

            cv2.imshow('test', image)

        key = cv2.waitKey(1)
        if key == 27:
            break

vc.release()
cv2.destroyAllWindows()


