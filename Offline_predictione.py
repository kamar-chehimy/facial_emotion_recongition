import tensorflow as tf
from tensorflow.keras.models import load_model

import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt


model_path='strip_facial_emot_rec.h5'
model = load_model(model_path)

# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_xml='haarcascade_frontalface_default.xml'


classes=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

#frame=cv2.imread('happy family.png')
frame=cv2.imread('different.jpg')


gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

font = cv2.FONT_HERSHEY_SIMPLEX


face_cascade = cv2.CascadeClassifier(face_xml)
test =face_cascade.load(face_xml)
print(test) # it should print true to know function will work properly

# Detect faces
faces = face_cascade.detectMultiScale(
                                        gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        flags=cv2.CASCADE_SCALE_IMAGE
                                        )# For each face
arr_min=[]
arr_max=[]
faces_roi=[]
for (x, y, w, h) in faces:
  # Draw rectangle around the face

  roi_gray=gray[y:y+h, x:x+w]
  roi_color=frame[y:y+h, x:x+w]
  arr_min.append((x,y))
  arr_max.append((x+w, y+h))
  
  facess=face_cascade.detectMultiScale(roi_gray)
  if len(facess)==0:
    print("face not identified")

  else:
    for (ex, ey, ew, eh) in facess:
      face_roi= roi_color[ey:ey+eh, ex:ex+ew]
      print("face Identified")

      faces_roi.append(face_roi)


for i in range(len(faces_roi)):

  image=cv2.rectangle(frame, arr_min[i], arr_max[i] , (255,0,0),2)
  pro_img=cv2.resize(faces_roi[i], (224,224))

  pro_img=np.expand_dims(pro_img, axis=0) # addd 4rth dim (1, 224,224,3)

  pro_img=pro_img/255.0

  pred=model.predict(pro_img)
  pred_ind=np.argmax(pred)


  cv2.putText(image,str(classes[pred_ind]),( arr_min[i][0],  arr_min[i][1]), font, 1,(255,0,0),2)

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

plt.axis('off')
plt.show()


