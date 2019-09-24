import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
from keras import backend as K
from keras.datasets import cifar10
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array


# Load Data
filenames = os.listdir("./data")
random.shuffle(filenames)
imgs = []
fileCount = 0
print("Loading data...")
for filename in tqdm(filenames):
    if (filename[0:6] == 'airbus' or filename[0:6] == 'boeing'):
        fileCount += 1
        imgArray = img_to_array(load_img("./data/" + filename))
        imgs.append(imgArray.flatten())
print(str(fileCount) + " images loaded.")

# PCA data
df = pd.DataFrame(imgs)

print(df.head()) 

pcagen = PCA(n_components=3)
pca_converted = pcagen.fit_transform(df.iloc[:,:-1])
pca_df = pd.DataFrame(pca_converted)
print(pca_df.head())

gui = plt.figure()
plot = gui.add_subplot(111, projection='3d')
x = pca_converted[:,0]
y = pca_converted[:,1]
z = pca_converted[:,2]

plot.scatter(x, y, z)
plt.show()