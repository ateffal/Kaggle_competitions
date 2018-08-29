# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 11:09:58 2018

@author: a.teffal
"""

#importing packages
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

#reading and splitting data
labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

#visualizing an image
i=1
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])

plt.hist(train_images.iloc[100])

#First model
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)

#Transforming images to black and white
test_images[test_images>0]=1
train_images[train_images>0]=1

#Visualizing an image
img=train_images.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])

plt.hist(train_images.iloc[i])

#Retraining the model 
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)


#Predicting on test data
test_data=pd.read_csv('test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:5000])

#Saving predictions to a file
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)

i=199
img=test_data.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(test_data.index[i])

#function to transform an image into a line of dataframe having the structure
#of test_data
def numeriser(file):
    pixels=[]
    im = Image.open(file)
    pix=im.load()
    ligne=im.size[0]
    colonne=im.size[1]
    for j in range(0,ligne):
        for i in range(0,colonne):
            pixels=pixels+[sum(pix[i,j])/3]
    zz=[tuple(pixels)]
    df = pd.DataFrame.from_records(zz, columns=list(test_data.columns))
    return df
    
#function to predict an image structured as a dataframe
def predict_image(df_):
    df_[df_>0]=1
    return clf.predict(df_)
    
    
#function to visualize an image stored as dataframe
def visualize_image(df):
    img=df.iloc[0].as_matrix().reshape((28,28))
    plt.imshow(img,cmap='binary')
    plt.title(train_labels.iloc[i])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
