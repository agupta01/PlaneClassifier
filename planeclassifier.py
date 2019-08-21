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

airbus_dir = "./AirbusPics"
boeing_dir = "./BoeingPics"
imgs_airbus = ['./AirbusPics/{}'.format(i) for i in os.listdir(airbus_dir)]
imgs_boeing = ['./BoeingPics/{}'.format(i) for i in os.listdir(boeing_dir)]
print("Boeing pic count: " + str(len(imgs_boeing)))
random.shuffle(imgs_airbus)
random.shuffle(imgs_boeing)

train_imgs = imgs_airbus[0:round(len(imgs_airbus)*(1.00-test_split))]
train_imgs.append(imgs_boeing[0:round(len(imgs_boeing)*(1.00-test_split))])
random.shuffle(train_imgs)

test_imgs = imgs_airbus[]


img = ip.load_img(train_imgs[0])
print(img.size)
img.show()
imgarray = ip.img_to_array(img)
print(imgarray)
print(imgarray.shape)

datagen = ip.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
