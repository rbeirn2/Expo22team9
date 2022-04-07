
import pandas as pd
import numpy as np
import tensorflow as tf   # fast numerical computing used to create Deep Learning models 
import matplotlib.pyplot as plt
import time   # allows us to handle various operations regarding time,
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #from internet
from IPython.display import clear_output # in a loop clear all old data and print just the last one
from sklearn.decomposition import PCA  #Linear dimensionality reduction using Singular Value Decomposition to a lower dimensional space.
import pickle as pk # Pickling is a quick and easy way to serialize objects.; binary, bytes
import cv2

from sklearn.model_selection import train_test_split  # splitting data arrays into two subsets:training data and for testing data,automatically
from sklearn.neighbors import KNeighborsClassifier   #finding the nearest neighbors between two sets of data,
from sklearn.svm import SVC  #SVC (Support Vector Classifier) is to fit to the data you provide, returning a "best fit" hyperplane that divides, or categorizes, your data.
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, plot_confusion_matrix, plot_roc_curve
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, GaussianNoise, Conv1D
from keras import regularizers
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# PREPARE DATA


sample_size = 3000
width = 160
height = 90


files = ['damaged', 'intact'] #calling the category of images from the dataset
adress = '/Users/"name"/Desktop/pythonProject/data_side_only/{}' #directory of the dataset 


# Use chdir() to change the directory ; format columns
# Use of os.getcwd() To know the current working directory of the file,
# finding an item which ends with png
# you can use to add items to the end of a given list

data_surface = {}
for f in files:
    data_surface[f]=[]
for col in files:
    os.chdir(adress.format(col))         
    for i in os.listdir(os.getcwd()):     
        if i.endswith('.png'):           
            data_surface[col].append(i)  


# STANDARDIZATION = SCALING IMAGES


# assigning
start = time.time() #returns the time as a floating point number expressed in seconds
image_data = []
image_target = []


# resizing of the images 
# make a list with all the titles
# clear the all output
# calculate the time of resizing

for title in files:
    os.chdir('/Users/"name"/Desktop/pythonProject/data_side_only/{}'.format(title))
    counter = 0
    for i in data_surface[title]:
        img = cv2.imread(i,0)
        image_data.append(cv2.resize(img,(width, height)).flatten())
        image_target.append(title)
        counter += 1
        if counter == sample_size:
            break
    clear_output(wait=True)
    print("Compiled Class",title)
calculate_time = time.time() - start
print("Load Img Time",round(calculate_time,3))




#NORMALIZATION = ENCODE TARGET VALUES


# takes 3 dim arrays of image RGB: example(456,342,123,3)
# LabelEncoder = Encode target labels with value between 0 and n_classes-1. = label each class

image_data_array= np.array(image_data)
labels = LabelEncoder()
labels.fit(image_target)

# Normalization
image_data_norm =image_data_array




#PCA PROCESS = DIMENSIONALITY REDUCTION (preprocessing, extracting features, matching feature, classification)


# time of finding
# PCA process

start = time.time()
pca = PCA()
pca.fit(image_data_norm)
# print('Time of PCA1: ',(time()-start))
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.99) + 1
print('PCA dimension: ',d)
# start = time()
pca = PCA(n_components=d)
image_data_PCA=pca.fit_transform(image_data_norm)
print('Time of PCA: ',round((time.time()-start),3))




# SPLITTING DATA


y_labels = labels.transform(image_target)
# y = to_categorical(y_labels)
image_data_train, image_data_test, y_train, y_test = train_test_split(image_data_PCA, y_labels,
                                                                      test_size=0.2,
                                                                      random_state=42,shuffle=True)


# BUILD NN MODEL WITH DENSITY LAYERS AND DROPOUT LAYERS


# KNeighborsClassifier ; for multiclass classification.
# model = KNeighborsClassifier(2)
# model.fit(image_data_train, y_train)
# y_pred = model.predict(image_data_test)
# print("Acc:",round(accuracy_score(y_test,y_pred),2))
# plot_confusion_matrix(model,image_data_test, y_test, cmap='Greens')
# plt.show()

# SVC model ; fitting our model
# model = SVC()
# model.C = 100
# model.fit(image_data_train, y_train)
# y_pred = model.predict(image_data_test)
# print("Acc:",round(accuracy_score(y_test,y_pred),3))
# plot_confusion_matrix(model,image_data_test, y_test, cmap='Greens')
# plt.show()


# NN model
X_train = image_data_train
X_test = image_data_test
print('Train size: ', X_train.shape)
print('Test size: ', X_test.shape)

# NN model
model = Sequential()
model.add(Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_dim=d))
model.add(Dropout(rate=0.5))
# model.add(Dense(8, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
# model.add(Dropout(rate=0.4))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(rate=0.3))
model.add(Dense(units=1, activation='sigmoid'))





#TRAINING PROCESS

#initialize the optimizer and model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#train the network
start = time.time()
n_epochs=200
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
model_history = model.fit(X_train, y_train, batch_size=32, epochs=n_epochs, shuffle=True,
                          validation_split = 0.2, callbacks=[es_callback])
print('Time per epoch: ',(time.time()-start))





#MODEL EVAULATION

pred_train= model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}'.format(scores[1]))

pred_test= model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}'.format(scores2[1]))

pred_total= model.predict(image_data_PCA)
scores = model.evaluate(image_data_PCA, y_labels, verbose=0)
print('Accuracy on whole data: {}'.format(scores[1]))





#PLOTS

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
