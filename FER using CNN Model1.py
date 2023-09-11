#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES..

# In[14]:


import os
import glob as gb
import pandas as pd
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt


# In[15]:


class_names = ['Anger','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']


# In[16]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(TRAIN_DIR,
                                                 target_size = (128, 128),
                                                 batch_size = BATCH_SIZE,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(TEST_DIR,
                                            target_size = (128, 128),
                                            batch_size = BATCH_SIZE,
                                            class_mode = 'categorical')


# # RESULT..

# In[17]:


model_path = "model1.h5"
loaded_model = keras.models.load_model(model_path)

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

image = cv2.imread("C:/Users/Stewart/OneDrive/Desktop/Emotion-Detection/test/Happiness/IMG-20221121-WA0008.jpg")

image_fromarray = Image.fromarray(image, 'RGB')
resize_image = image_fromarray.resize((128, 128))
expand_input = np.expand_dims(resize_image,axis=0)
input_data = np.array(expand_input)
input_data = input_data/255

pred = loaded_model.predict(input_data)
result = pred.argmax()
result


# In[18]:


training_set.class_indices


# # TRYING TO DRAW A RECTANGLE ACROSS THE FACE..

# In[19]:


#Object Detection Algorithm used to identify faces in an image or a real time video.
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#Haar-Cascade Face Detection Algorithm.


# In[20]:


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, 1.1, 4)
for(x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


# In[21]:


plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# In[ ]:




