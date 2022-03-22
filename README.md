# Face_Mask_Model

<img src=data/screenshots/opencv_test.jpg width="750" height="500">

## Project description

This is a personal project which is my introduction into Machine Learning, as well as Image Processing. The project uses [20,000 images](https://www.kaggle.com/datasets/pranavsingaraju/facemask-detection-dataset-20000-images) and [TensorFlow](https://www.tensorflow.org/resources/learn-ml?gclid=EAIaIQobChMIj57ll57a9gIVPYBQBh2a4wkDEAAYASAAEgLVS_D_BwE) to train a Convolutional Neural Network to distinguish between people who wear a face mask and those who don't. After trainig the model, it is then imported into `mask-model.py` where I am using `haar_cascades` and [OpenCV](https://opencv.org/) to open the webcam and detect the face in the frame, and output on the image whether the person wears a face mask by having the model predict that.

## Training Walkthrough

The first thing done was acquiring the dataset images. Then, they 70% of them are used for training, and the rest of them is used for validating : 
<img src=data/screenshots/data_split.png width="1500" height="500">

Next, the CNN was designed. The architecture is as follows : 

```
model = models.Sequential(tf.keras.layers.Rescaling(1./255))
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2,activation = 'softmax'))
```

For the compiler, Adam optimizer is being used. Since the last layer uses Softmax, logits is set to `False` . Once the model was compiled, the model was trained, using the number of batches specified in the beginning. 

<img src=data/screenshots/93%25.png width="1500" height="500">

A helper function has been designed to help test the model. It takes an imagepath as input, and then using the trained model it predicts whether the person in the picture wears a facemask. Some predictions can be observed below : 

<img src=data/screenshots/mask.png width="1500" height="500">
<img src=data/screenshots/wmask.png width="1500" height="500">

Lastly, the model is save to the specified filename.

<img src=data/screenshots/saving.png width="1500" height="400">

## OpenCV

Next, the saved model is loaded into `mask-model.py`. Then using OpenCV the webcam is opened : 
```Python
capture = cv.VideoCapture(0)
while True:
    ret, frame = capture.read()
    frame = cv.flip(frame,1)
    frame = processFrame(frame)
    cv.imshow('Frames',frame)
    if cv.waitKey(1) == ord('q'):
        break

```

`processFrame` then takes each frame and uses the `haar_cascade` and loaded model to draw a rectangle around the person's face, and ouput on the image whether the person wears a facemask.

## Conclusions

While the end product is not perfect(The model is trained on only 14,000 images of people from the same frontal profile "wearing" the same type of mask, and haar_cascade struggles sometimes to find the face with a mask on), it has been a great tool of teaching myself some things about Image Processing, as well as an introduction into Machine Learning. 
