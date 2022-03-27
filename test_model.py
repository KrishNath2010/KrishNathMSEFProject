# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:17:23 2022

@author: Krish Nath
"""
import PIL
import sys
#print(sys.path)
from PIL import Image
import numpy as np
import tensorflow as tf
#from tensorflow import keras
test_img=(Image.open(r"C:\Users\Krish Nath\MSEF Project\test_image_2.jpg")).resize((500,500))
model = tf.keras.models.load_model((r"C:\Users\Krish Nath\MSEF Project" +"\ mnh").replace(" mnh",""))
class_names=['100 Federal Street', '111 Huntington Ave', 'BNY Mellon Center at One Boston Place',
'Bulfinch Crossing', 'Federal Reserve Bank', 'Four Seasons Hotel & Private Residence',
'Jhon Hancok', 'Millennium Tower', 'One International Place', 'Prudential']
img_array = tf.keras.utils.img_to_array(test_img)
img_array = tf.expand_dims(img_array, 0)
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
#score=0
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)