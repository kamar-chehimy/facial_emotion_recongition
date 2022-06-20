import tensorflow as tf
from tensorflow.keras.models import load_model

import cv2
import os
import random
import numpy as np


model_path='fac_emot_rec_pretr.h5'
model = load_model(model_path)

face_xml='haarcascade_frontalface_default.xml'

classes=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier(face_xml)
test =face_cascade.load(face_xml)
#print(test) # it should print true to know function will work properly


font = cv2.FONT_HERSHEY_SIMPLEX
video_capture = cv2.VideoCapture(0) # try 1,2,3,etc if 0 caused error
#print(video_capture)


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    arr_min=[]
    arr_max=[]
    faces_roi=[]
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        arr_min.append((x,y))
        arr_max.append((x+w, y+h))

        facess=face_cascade.detectMultiScale(roi_gray)
        if len(facess)==0:
          print("face not identified")

        else:
          for (ex, ey, ew, eh) in facess:
            face_roi= roi_color[ey:ey+eh, ex:ex+ew]
            #print("face Identified")
            faces_roi.append(face_roi)

    for i in range(len(faces_roi)):
      image=cv2.rectangle(frame, arr_min[i], arr_max[i] , (255,0,0),2)
      pro_img=cv2.resize(faces_roi[i], (224,224))

      pro_img=np.expand_dims(pro_img, axis=0) # addd 4rth dim (1, 224,224,3)

      pro_img=pro_img/255.0

      pred=model.predict(pro_img)
      pred_ind=np.argmax(pred)


      cv2.putText(image,str(classes[pred_ind]),( arr_min[i][0],  arr_min[i][1]), font, 1,(255,0,0),2)

    # Display the resulting frame
    cv2.imshow('Video', frame) # # Video(Window name) in which image frame is displayed 

    # waits for user to press q
    # (this is necessary to avoid Python kernel form crashing)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
      break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


