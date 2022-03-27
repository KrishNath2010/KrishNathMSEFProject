# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 21:52:55 2022

@author: Krish Nath
"""

import PIL
import tensorflow as tf
from PIL import Image
import glob
import os
import pathlib

#print(Path.cwd())
#img_paths = glob.glob(os.path.join(<path_to_dataset>,'*/*.*') # assuming you point to the directory containing the label folders.
img_paths = r"C:\Users\Krish Nath\MSEF Project\Boston_Buildings_Dataset_test\Prudential"
#data_dir = pathlib.Path(img_paths)
bad_paths = []
i=0
#folder_dir = (r"C:\Users\Krish Nath\MSEF Project\Boston_Buildings_Dataset"+ "\ mnh").replace(" mnh","") + "111 Huntington Ave"
#for images in os.listdir(folder_dir):
 #   if (images.endswith(".jpg") and i!=5):
  #      im = Image.open(((r"C:\Users\Krish Nath\MSEF Project\Boston_Buildings_Dataset"+ "\ mnh").replace(" mnh","") + "100 Federal Street" +"\ mnh").replace(" mnh","") +str(images))
        #im.show()
        #i+=1
   #     im1 = tf.read_file()
    #    decode = tf.io.decode_image(im)
for image_path in os.listdir(img_paths):
    print(image_path)
    if (image_path.endswith(".jpg")):
        try:
            full_file_path = os.path.join(img_paths,image_path)
            img_bytes = tf.io.read_file(full_file_path)
            decoded_img = tf.io.decode_image(img_bytes)
        except tf.errors.InvalidArgumentError as e:
            print(f"Found bad path {image_path}...")
            bad_paths.append(image_path)

        print(f"{image_path}: OK")

print("BAD PATHS:")
for bad_path in bad_paths:
    print(f"{bad_path}")
