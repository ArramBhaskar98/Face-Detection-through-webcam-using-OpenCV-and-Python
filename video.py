import os
import cv2
import numpy as np
import faceDetection as fr

# This module captures images via webcam and performs face recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/dell/PycharmProjects/Detection/trainingData3.yml')

name = {0:'Bhaskar', 1:'Dhoni'}

cap = cv2.VideoCapture(0)

while True:
    ret,test_img = cap.read()
    faces_detected, gray_img = fr.faceDetection(test_img)

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=1)

    resized_img = cv2.resize(test_img,(1000,700))
    cv2.imshow("face detection:",resized_img)
    cv2.waitKey(10)

    for face in faces_detected:
        (x,y,w,h) = face
        roi_gray = gray_img[y:y+w,x:x+h]
        label,confidence = face_recognizer.predict(roi_gray)
        print("confidence:",confidence)
        print("label:",label)
        fr.draw_rect(test_img,face)
        predicted_name = name[label]
        if confidence < 39:
            fr.put_text(test_img,predicted_name,x,y)

    resized_img = cv2.resize(test_img,(1000,700))
    cv2.imshow("face recoginition:",resized_img)
    if cv2.waitKey(10) == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
