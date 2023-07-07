#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


directory= 'C:/Users/Nikhitha Reddy/OneDrive/Documents/VI_sem/mini_project/pcos_detection/data/train'
#C:\Users\Nikhitha Reddy\OneDrive\Documents\VI_sem\mini_project\pcos_detection\data


# In[3]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten 
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2
import numpy as np
import os
import pandas as pd


# In[4]:


train_ds=tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="binary",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(224, 224),
    shuffle=True,
    seed=24,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=True,
    crop_to_aspect_ratio=False,
)


# In[6]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
class_names = train_ds.class_names
for images, labels in train_ds.take(2):
    for i in range(32):
        ax = plt.subplot(6, 6, i + 1) # since a perfect square of 6 and above is > 32 , it works for  any combination which produces the space for >= 32 images 
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")


# In[7]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
# create generator
datagen = ImageDataGenerator(rescale = 1./255,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             horizontal_flip = True,
                             vertical_flip=True,
                             rotation_range=30,
                             validation_split=0.3,
                             fill_mode='nearest')
# prepare an iterators for each dataset
train_it = datagen.flow_from_directory( directory, 
                                       class_mode='categorical',
                                       classes=['infected', 'notinfected'],
                                       target_size=(224, 224),
                                       batch_size=100,
                                       subset='training',
                                       seed=24)
val_it = datagen.flow_from_directory( directory, 
                                       class_mode='categorical',
                                       classes=['infected', 'notinfected'],
                                       target_size=(224, 224),
                                       batch_size=100,
                                       subset='validation',
                                       seed=24)


# In[8]:


batchX, batchy = train_it.next()
#batch size we took is 200 so its shape will be 100 here
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchy.shape, batchy.min(), batchy.max()))
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))


# In[9]:


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import Sequential


# # Model1

# In[10]:


model1 = Sequential()
#create a sequential model, which is a linear stack of layers 
#using this we can create model by adding layers one at a time
#Conv2D(filters, kernel_size, activation, input_shape,etc); relu -> f(x) = max(0, x) This can help with the problem of vanishing gradients,
model1.add(Conv2D(10, (5,5),padding='valid',activation='relu',input_shape=(224,224,3)))
#maxpooling2d -> reduce the spatial size (height and width) of the input tensor, while keeping the number of channels the same
#Max pooling is used in CNNs to reduce the dimensionality of the input data and to prevent overfitting
model1.add(MaxPooling2D(pool_size=(4,4)))
model1.add(Conv2D(12, (5,5),padding='valid',activation='relu'))
model1.add(MaxPooling2D(pool_size=(4,4)))
model1.add(Conv2D(5, (3,3),padding='valid',activation='relu'))
# model3.add(Conv2D(256, (5,5),padding='valid',activation='relu'))
model1.add(MaxPooling2D(pool_size=(3,3)))
model1.add(Flatten())
#model3.add(Dense(128,activation='relu'))
#model3.add(Dense(64,activation='relu'))
model1.add(Dense(2,activation='softmax'))


# In[11]:


from tensorflow.keras.losses import CategoricalCrossentropy
model1.compile(
  optimizer='adam',
  loss=CategoricalCrossentropy(),
    #CategoricalCrossentropy loss function is used when the target variable is one-hot encoded
  metrics=['accuracy'])


# In[12]:


history = model1.fit( 
  train_it,
  validation_data=val_it,
  epochs=6)


# # model2

# In[15]:


model2 = Sequential()
model2.add(Conv2D(15, (5,5),padding='valid',activation='relu',input_shape=(224,224,3)))
model2.add(MaxPooling2D(pool_size=(5,5)))
model2.add(Conv2D(12, (4,4),padding='valid',activation='relu'))
model2.add(MaxPooling2D(pool_size=(4,4)))
model2.add(Conv2D(8, (3,3),padding='valid',activation='relu'))
# model5.add(Conv2D(256, (5,5),padding='valid',activation='relu'))
model2.add(MaxPooling2D(pool_size=(3,3)))
model2.add(Flatten())
#model5.add(Dense(128,activation='relu'))
#model5.add(Dense(64,activation='relu'))
model2.add(Dense(2,activation='softmax'))


# In[16]:


from tensorflow.keras.losses import CategoricalCrossentropy
model2.compile(
  optimizer='adam',
  loss=CategoricalCrossentropy(),
  metrics=['accuracy'])


# In[18]:


history = model2.fit( 
  train_it,
  validation_data=val_it,
  epochs=7)


# # Model 3

# In[20]:


model3 = Sequential()
model3.add(Conv2D(10, (5,5),padding='valid',activation='relu',input_shape=(224,224,3)))
model3.add(MaxPooling2D(pool_size=(4,4)))
model3.add(Conv2D(12, (5,5),padding='valid',activation='relu'))
model3.add(MaxPooling2D(pool_size=(4,4)))
model3.add(Conv2D(5, (3,3),padding='valid',activation='relu'))
# model3.add(Conv2D(256, (5,5),padding='valid',activation='relu'))
model3.add(MaxPooling2D(pool_size=(3,3)))
model3.add(Flatten())
#model3.add(Dense(128,activation='relu'))
#model3.add(Dense(64,activation='relu'))
model3.add(Dense(2,activation='softmax'))


# In[21]:


from tensorflow.keras.losses import CategoricalCrossentropy
model3.compile(
  optimizer='adam',
  loss=CategoricalCrossentropy(),
  metrics=['accuracy'])


# In[22]:


history = model3.fit( 
  train_it, 
  validation_data=val_it,
  epochs=6)


# # Model 4

# In[23]:


model4 = Sequential()
model4.add(Conv2D(12, (5,5),padding='valid',activation='relu',input_shape=(224,224,3)))
model4.add(MaxPooling2D(pool_size=(4,4)))
model4.add(Conv2D(10, (5,5),padding='valid',activation='relu'))
model4.add(MaxPooling2D(pool_size=(4,4)))
model4.add(Conv2D(8, (3,3),padding='valid',activation='relu'))
# model4.add(Conv2D(256, (5,5),padding='valid',activation='relu'))
model4.add(MaxPooling2D(pool_size=(3,3)))
model4.add(Flatten())
#model4.add(Dense(128,activation='relu'))
#model4.add(Dense(64,activation='relu'))
model4.add(Dense(2,activation='softmax'))


# In[24]:


from tensorflow.keras.losses import CategoricalCrossentropy
model4.compile(
  optimizer='adam',
  loss=CategoricalCrossentropy(),
  metrics=['accuracy'])


# In[25]:


tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)

history = model4.fit( 
  train_it,
  validation_data=val_it,
  epochs=10,callbacks=[tb_callback])


# # Model 5

# In[27]:


model5 = Sequential()
model5.add(Conv2D(15, (5,5),padding='valid',activation='relu',input_shape=(224,224,3)))
model5.add(MaxPooling2D(pool_size=(5,5)))
model5.add(Conv2D(12, (4,4),padding='valid',activation='relu'))
model5.add(MaxPooling2D(pool_size=(4,4)))
model5.add(Conv2D(8, (3,3),padding='valid',activation='relu'))
# model5.add(Conv2D(256, (5,5),padding='valid',activation='relu'))
model5.add(MaxPooling2D(pool_size=(3,3)))
model5.add(Flatten())
#model5.add(Dense(128,activation='relu'))
#model5.add(Dense(64,activation='relu'))
model5.add(Dense(2,activation='softmax'))


# In[28]:


from tensorflow.keras.losses import CategoricalCrossentropy
model5.compile(
  optimizer='adam',
  loss=CategoricalCrossentropy(),
  metrics=['accuracy'])


# In[29]:


history = model5.fit( 
  train_it,
  validation_data=val_it,
  epochs=7)


# # saving model

# In[30]:


model2.save('model.h2')
#saved to a file model.h2


# In[31]:


from tensorflow import keras
model = keras.models.load_model('model.h2')
#saved model is being loaded here


# In[32]:


from tensorflow.keras.preprocessing.image import load_img
image = load_img('C:/Users/Nikhitha Reddy/OneDrive/Documents/VI_sem/mini_project/pcos_detection/data/test/infected/img_0_360.jpg', target_size=(224, 224))
img = np.array(image)
img = img / 255.0
img = img.reshape(1,224,224,3)
prediction = model.predict(img)


# In[33]:


type(prediction)


# In[34]:


print(prediction)


# In[35]:


l={"infected":prediction[0][0],"notinfected":prediction[0][1]}
def get_key(val):
    for key, value in l.items():
         if val == value:
             return key
 
    return "key doesn't exist"


# In[36]:


j=prediction.max()
get_key(j)


# In[37]:


# final accuracy we got is 99.48%

