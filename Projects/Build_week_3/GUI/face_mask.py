import cv2
import  dlib


detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor()
cap=cv2.VideoCapture(0)
while True:
    success,frame=cap.read()
    frame=cv2.flip(frame,4)
    # if len(list)!=0:
    #     #img=af.draw_dog(list,img)
    #     # img=af.draw_angry_dog(list,img)
    #     # img=af.draw_rabbit(list,img)
    #     img=af.draw_fish(list,img)

    cv2.imshow("Frame",frame)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
