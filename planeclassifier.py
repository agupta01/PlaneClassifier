import os
import random

import numpy as np

import keras.preprocessing.image as ip
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import backend as K
from tensorflow import keras

K.tensorflow_backend._get_available_gpus()

test_split = 0.30 #proportion of data split away for use in test set

className = ["Airbus", "Boeing"]

# Import images from library
airbus_dir = "./AirbusPics"
boeing_dir = "./BoeingPics"
imgs_airbus = ['./AirbusPics/{}'.format(i) for i in os.listdir(airbus_dir)]
imgs_boeing = ['./BoeingPics/{}'.format(i) for i in os.listdir(boeing_dir)]
random.shuffle(imgs_airbus)
random.shuffle(imgs_boeing)
print("Total imgs: " + str(len(imgs_airbus) + len(imgs_boeing)))

# Split into training and test sets
train_imgs = imgs_airbus[0:round(len(imgs_airbus)*(1.00-test_split))] + imgs_boeing[0:round(len(imgs_boeing)*(1.00-test_split))]
random.shuffle(train_imgs)
test_imgs = imgs_airbus[round(len(imgs_airbus)*(1.00-test_split)): len(imgs_airbus)] + imgs_boeing[round(len(imgs_boeing)*(1.00-test_split)): len(imgs_boeing)]
random.shuffle(test_imgs)

# Check to make sure images imported correctly
img = ip.load_img(train_imgs[0])
imgt = ip.load_img(test_imgs[0])
print(str(img.size) + " \t " + str(imgt.size))
img.show()
print("Training pics: " + str(len(train_imgs)))
print("Testing pics: " + str(len(test_imgs)))
print("Full set: " + str(len(train_imgs) + len(test_imgs)))
imgarray = ip.img_to_array(img)
print(imgarray.shape)

# Augment dataset by altering each image and saving the copy
datagen = ip.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
