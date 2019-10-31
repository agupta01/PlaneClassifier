import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
# %matplotlib inline

# Load Data
data = np.array([])
filenames = os.listdir("./data")
for file in tqdm(filenames):
    if ("boeing" in file) or ("airbus" in file):
        data = np.append(data,  img_to_array(load_img("./data/" + file)))

x_train, x_split = 
