import os
import random

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from keras import backend as K
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPooling2D)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import to_categorical

K.tensorflow_backend._get_available_gpus()

IMAGE_WIDTH = 399
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_CHANNELS = 3

# Load Data
filenames = os.listdir("./data")
random.shuffle(filenames)
imgs = []
fileCount = 0
print("Loading data...")
while (fileCount < 10):
    filename = filenames[fileCount]
    category = filename[0:6]
    # print(category)
    if category == 'airbus':
        fileCount += 1
        imgs.append(filename)
    elif category == 'boeing':
        fileCount += 1
        imgs.append(filename)

df = pd.DataFrame({
    'filename': imgs,
})
nb_samples = df.shape[0]

# Create testing generator
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    df,
    "./data/",
    x_col = 'filename',
    y_col = None,
    class_mode = None,
    target_size = IMAGE_SIZE,
    batch_size = 15,
    shuffle = False
)

# Redefine model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Load previously saved weights into model
model.load_weights("./model.h5")

# Run prediction using model and generate probabilities array
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/15))
img = load_img('./data/' + df.iloc[0]['filename'])
img.show()
print("There is a {}% chance this aircraft is an Airbus.".format(round((predict[0][0]*100), 3)))
print("There is a {}% chance this aircraft is a Boeing.".format(round((predict[0][1]*100), 3)))