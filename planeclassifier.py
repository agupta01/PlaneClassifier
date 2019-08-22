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

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print(model.summary())

earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

# Preprocess data
df['category'] = df['category'].replace({0: 'airbus', 1: 'boeing'})

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 15

train_datagen = ImageDataGenerator(
    rotation_range = 25,
    rescale = 1./255,
    shear_range = 0.5,
    horizontal_flip = True,
    vertical_flip = True,
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

# import os
# import random

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow import keras

# import keras
# import keras.preprocessing.image as ip
# import tqdm
# from keras import backend as K
# from keras import layers

# K.tensorflow_backend._get_available_gpus()

# test_split = 0.30 #proportion of data split away for use in test set

# className = ["Airbus", "Boeing"]

# # Import images from library
# airbus_dir = "./AirbusPics/airbus"
# boeing_dir = "./BoeingPics/boeing"
# imgs_airbus = []
# imgs_boeing = []
# for i in range(0,2548):
#     imgs_airbus.append(airbus_dir + str(i) + '.jpg')
#     imgs_boeing.append(boeing_dir + str(i) + '.jpg')
# random.shuffle(imgs_airbus)
# random.shuffle(imgs_boeing)
# print("Total imgs: " + str(len(imgs_airbus) + len(imgs_boeing)))

# # Split into training and test sets
# train_imgs = imgs_airbus[0:round(len(imgs_airbus)*(1.00-test_split))] + imgs_boeing[0:round(len(imgs_boeing)*(1.00-test_split))]
# random.shuffle(train_imgs)
# test_imgs = imgs_airbus[round(len(imgs_airbus)*(1.00-test_split)): len(imgs_airbus)] + imgs_boeing[round(len(imgs_boeing)*(1.00-test_split)): len(imgs_boeing)]
# random.shuffle(test_imgs)

# # Check to make sure images imported correctly
# # img = ip.load_img(train_imgs[0])
# # imgt = ip.load_img(test_imgs[0])
# # print(str(img.size) + " \t " + str(imgt.size))
# # img.show()
# # print("Training pics: " + str(len(train_imgs)))
# # print("Testing pics: " + str(len(test_imgs)))
# # print("Full set: " + str(len(train_imgs) + len(test_imgs)))
# # imgarray = ip.img_to_array(img)
# # print(imgarray.shape)

# # Preprocess data by flattening image
# train_data = np.zeros((len(train_imgs),268128))
# train_labels = np.empty(len(train_imgs), dtype=object)
# print("Loading data...")
# for i in range(0,len(train_imgs)):
#     img = ip.load_img(train_imgs[i])
#     img_data = ip.img_to_array(img)
#     # if (img_data.shape != (225,400,3)):
#     #     print("Inconsistency at {}. Shape is {}".format(i, img_data.shape))
#     train_data[i] = img_data[:224,:399,:3].flatten()
#     if 'boeing' in train_imgs[i]:
#         train_labels[i] = "Boeing"
#     elif 'airbus' in train_imgs[i]:
#         train_labels[i] = "Airbus"
# img = ip.load_img(train_imgs[0])
# img.show()
# print("Aircraft type: " + train_labels[0])
# print(train_data[0].shape)

# # Initialize model: 18 layers, binary classification
# model = keras.Sequential([
# 	keras.layers.Dense(134064, input_shape=(268128,)),
# 	keras.layers.Dense(67032, activation=tf.nn.relu),
#     keras.layers.Dense(16758, activation=tf.nn.relu),
#     keras.layers.Dense(4190, activation=tf.nn.relu),
#     keras.layers.Dense(1048, activation=tf.nn.relu),
#     keras.layers.Dense(131, activation=tf.nn.relu),
#     keras.layers.Dense(16, activation=tf.nn.relu),
#     keras.layers.Dense(2, activation=tf.nn.relu)
# ])

# # Compile model
# print("Compiling...")
# model.compile(optimizer=tf.train.AdamOptimizer(), #favorite optimizer b/c it works a bit faster than stochastic gradient descent
# 			  loss='sparse_categorical_crossentropy', #this is a special loss function for probability outputs, a.k.a log-loss
# 			  metrics=['accuracy']) #evaulates model on how many images are correctly classified

# # Train model
# print('Fitting begun...')
# model.fit(train_data, train_labels, epochs=10)
# print('Fitting finished!')

# # NOT USED YET: Augment dataset by altering each image and saving the copy
# # datagen = ip.ImageDataGenerator(
# #     featurewise_center=True,
# #     featurewise_std_normalization=True,
# #     rotation_range=20,
# #     width_shift_range=0.2,
# #     height_shift_range=0.2,
# #     horizontal_flip=True)
