import cv2 as cv
import joblib
import numpy as np
from tensorflow.keras import datasets, layers, models
import tensorflow as tf

filename = 'trained_mask_model.sav'
loaded_model = joblib.load(filename)

class_names = ["with_mask","without_mask"]

img_height = 128
img_width = 128
haar_cascade = cv.CascadeClassifier('haar_face.xml')

def predict(frame, model):
    frame= cv.resize(frame, (img_height,img_width), interpolation = cv.INTER_AREA)
    img_array = tf.keras.preprocessing.image.img_to_array(frame)
    img_batch = np.expand_dims(img_array, axis=0)
    tf.image.rgb_to_grayscale(img_batch)
    prediction = model.predict(img_batch)
    return class_names[np.argmax(prediction)]

def processFrame(frame):
    gray = cv.cvtColor(frame,cv.COLOR_BGR2RGB )
    faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors = 3)
    text = predict(frame,loaded_model)
    for(x,y,w,h) in faces_rect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness = 2)
    if(len(faces_rect)>=0):
        cv.putText(img=frame, text= text, org=(img_height,img_width), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=3)

    return frame 

capture = cv.VideoCapture(0)
while True:
    ret, frame = capture.read()
    frame = cv.flip(frame,1)
    frame = processFrame(frame)
    cv.imshow('Frames',frame)
    if cv.waitKey(1) == ord('q'):
        break


capture.release()
cv.destroyAllWindows()

