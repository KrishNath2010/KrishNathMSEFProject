# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 22:24:47 2022

@author: Krish Nath
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#data_dir = r"C:\Users\Krish Nath\Downloads\flower_photos\flower_photos"
#folder_dir = r"C:\Users\Krish Nath\MSEF Project\Boston_Buildings_Dataset_test"
#for images in os.listdir(folder_dir):
    #if (images.endswith(".jpg")):
        #im1 = Image.open(images)
        #im1 = im1.save(im1.resize((500,500)))
#for filename in os.listdir(folder_dir):
   # if (filename.endswith(".jpg")):
        #img = Image.open(os.path.join(folder_dir, filename)) # images are color images
        #img = img.resize((500,500), Image.ANTIALIAS)
        #img.save(folder_dir+filename+'.jpeg') 
#data_dir = (r"C:\Users\Krish Nath\Downloads\flower_photos")
data_dir = (r"C:\Users\Krish Nath\MSEF Project\Boston_Buildings_Dataset_orig")
#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
#roses = list(data_dir.glob('roses/*'))
#PIL.Image.open(str(roses[0]))
batch_size = 32
img_height = 500
img_width = 500
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
#import matplotlib.pyplot as plt
class_names = train_ds.class_names
print(class_names)
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
num_classes = len(class_names)
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.experimental.preprocessing.RandomRotation(0),
    layers.experimental.preprocessing.RandomZoom(0.1)])
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255, 
  input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.1505),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
# model = Sequential([
#   layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
epochs=24
#history = model.fit(train_ds, validation_data=train_ds, epochs=epochs)
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
path=(r"C:\Users\Krish Nath\MSEF Project" +"\ mnh").replace(" mnh","")
model.save(path)