import cv2
import numpy as np
import os
import faceDetection as fr

test_img = cv2.imread('C:/Users/dell/PycharmProjects/Detection/test_images/bachi.jpg')
faces_detected, gray_img = fr.faceDetection(test_img)
print("faces_detected:",faces_detected)

# faces,faceID = fr.labels_for_training_data('C:/Users/dell/PycharmProjects/Detection/Images')
# face_recognizer = fr.train_classifier(faces,faceID)
# face_recognizer.save('trainingData3.yml')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/dell/PycharmProjects/Detection/trainingData3.yml')
name = {0:'Bhaskar',1:'Dhoni'}

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+h,x:x+h]
    label, confidence = face_recognizer.predict(roi_gray)
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name = name[label]
    fr.put_text(test_img,predicted_name,x,y)


resized_img = cv2.resize(test_img,(1000,700))
cv2.imshow("face detected:",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


