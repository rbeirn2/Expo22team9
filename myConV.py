import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #from internet
import keras.losses
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import time
from IPython.display import clear_output
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
import json

from tensorflow import keras
from keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, plot_confusion_matrix, plot_roc_curve
from keras.utils import np_utils



# PREPARE DATA

sample_size = 3000
height = 90
width = 160


files = ['damaged', 'intact'] #calling the category where the dataset of images is in
adress = '/Users/"name"/Desktop/pythonProject/data_top_only/{}' #directory of the dataset
data_surface = {}
for f in files:
    data_surface[f]=[]
for col in files:
    os.chdir(adress.format(col))
    for i in os.listdir(os.getcwd()):
        if i.endswith('.png'):
            data_surface[col].append(i)



# STANDARDIZATION = SCALING IMAGES


start = time.time()
image_data = []
image_target = []

for title in files:
    os.chdir('/Users/"name"/Desktop/pythonProject/data_top_only/{}'.format(title))
    counter = 0
    for i in data_surface[title]:
        img = cv2.imread(i,0)
        image_data.append(cv2.resize(img,(width, height)))
        image_target.append(title)
        counter += 1
        if counter == sample_size:
            break
    clear_output(wait=True)
    print("Compiled Class",title)
calculate_time = time.time() - start
print("Load Img Time",round(calculate_time,3))




#NORMALIZATION = ENCODE TARGET VALUES/converting the data from integer to floating point values (float32= 0 to 1 range)


# image_data_array= np.array(image_data)
# print(image_data_array.shape)
labels = LabelEncoder() #encode target values
labels.fit(image_target)
y_labels = labels.transform(image_target)




# SPLITTING DATA


# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing

image_data_train, X_test, y_train_total, y_test = train_test_split(image_data, y_labels,
                                                                            test_size=0.2, random_state=42,
                                                                            shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(image_data_train, y_train_total,
                                                  test_size=0.2, random_state=42,
                                                  shuffle=True)




# DATA AUGMENTATION


# expend the 3D array to 4D array = (90,160,3) -> (90,160,3,1)

image_data_train= np.expand_dims(image_data_train, axis=-1)
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test= np.expand_dims(X_test, axis=-1)


# construct the training image generator for data augmentation
#Data augmentation--no rescaling rescale = none
image_gen = ImageDataGenerator(rescale=None)
train_set = image_gen.flow(X_train,y_train)
val_set = image_gen.flow(X_val,y_val)
test_set = image_gen.flow(X_test,y_test)
image_data_train_set = image_gen.flow(image_data_train,y_train_total)




# BUILD NN MODEL WITH CONVOLUTIONAL LAYERS


# ConV Network
image_shape = (height,width,1)
batch_size = 32
backend.clear_session()
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(4,4), strides=2, input_shape=image_shape, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=16, kernel_size=(2,2), strides=1, input_shape=image_shape, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(Conv2D(filters=32, kernel_size=(2,2), strides=1, input_shape=image_shape, activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(units=64,kernel_regularizer=regularizers.l2(0.001), activation='relu'))
# model.add(Dense(112, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=1, activation='sigmoid'))




#TRAINING PROCESS

#initialize the optimizer and model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

#train the network
from time import time
n_epochs = 10
start = time()
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
model_history = model.fit_generator(train_set, epochs=n_epochs,
                                    shuffle=True, validation_data=val_set,
                                    callbacks=[es_callback])
print('Time per epoch',(time()-start)/n_epochs)





#MODEL EVAULATION

pred_train= model.predict(image_data_train_set)
scores = model.evaluate(image_data_train_set, verbose=0)
print('Accuracy on training data: {}'.format(scores[1]))

pred_val= model.predict(val_set)
scores2 = model.evaluate(val_set, verbose=0)
print('Accuracy on validation data: {}'.format(scores2[1]))

pred_test= model.predict(test_set)
scores2 = model.evaluate(test_set, verbose=0)
print('Accuracy on test data: {}'.format(scores2[1]))

pred_total= model.predict(image_data_train_set)
scores = model.evaluate(image_data_train_set, verbose=0)
print('Accuracy on whole data: {}'.format(scores[1]))





#PLOTS

# # plot the training loss and accuracy
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
