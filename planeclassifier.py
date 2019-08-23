import os
import random

import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
categories = []
fileCount = 0
print("Loading data...")
for filename in tqdm(filenames):
    category = filename[0:6]
    # print(category)
    if category == 'airbus':
        fileCount += 1
        imgs.append(filename)
        categories.append(0)
    elif category == 'boeing':
        fileCount += 1
        imgs.append(filename)
        categories.append(1)

df = pd.DataFrame({
    'filename': imgs,
    'category': categories
})

print(df.tail())
print(df.shape)
print(fileCount)

# Initialize model
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

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

# Preprocess data
df['category'] = df['category'].replace({0: 'airbus', 1: 'boeing'})

train_df, validate_df = train_test_split(df, test_size=0.10, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 12

train_datagen = ImageDataGenerator(
    rotation_range = 25,
    rescale = 1./255,
    shear_range = 0.5,
    horizontal_flip = True,
    vertical_flip = True,
    channel_shift_range = 0.5
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    "./data/",
    x_col = 'filename',
    y_col = 'category',
    target_size = IMAGE_SIZE,
    class_mode = 'categorical',
    batch_size = batch_size
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    './data/',
    x_col = 'filename',
    y_col = 'category',
    target_size = IMAGE_SIZE,
    class_mode = 'categorical',
    batch_size = batch_size
)

# Train model
history = model.fit_generator(
    train_generator,
    epochs = 30,
    validation_data = validation_generator,
    validation_steps = total_validate//batch_size,
    steps_per_epoch = total_train//batch_size,
    callbacks = callbacks
)

# Save fully trained model
model.save_weights("model.h5")