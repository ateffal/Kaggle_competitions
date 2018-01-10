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
from sklearn.neural_network import MLPClassifier
    


#Setting the maximum number of lines to use in original train data
n_max=42000

#Setting the ration train/test
ratio=0.8

#reading and splitting data
labeled_images = pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

#Selecting data to use
images = labeled_images.iloc[0:n_max,1:]
labels = labeled_images.iloc[0:n_max,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=ratio, random_state=0)

#==============================================================================
# #visualizing an image
# i=11
# img=train_images.iloc[i].as_matrix()
# img=img.reshape((28,28))
# plt.imshow(img,cmap='gray')
# plt.title(train_labels.iloc[i,0])
# plt.hist(train_images.iloc[100])
#==============================================================================

#Transforming images to black and white
test_images[test_images>0]=1
train_images[train_images>0]=1
test_data[test_data>0]=1
#==============================================================================
# i=32
# #Visualizing an image
# img=train_images.iloc[i].as_matrix().reshape((28,28))
# plt.imshow(img,cmap='binary')
# plt.title(train_labels.iloc[i])
# plt.hist(train_images.iloc[i])
#==============================================================================

#Training the model
X = train_images.values
y = train_labels.values.ravel(n_max*ratio,)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(310, ), random_state=1,activation='logistic')
clf.fit(X, y)

#Scoring the model
print(clf.score(test_images,test_labels))

#Predicting on test data
results=clf.predict(test_data.values)

#Saving predictions to a file
temp=list(range(len(results)))
temp = [i+1 for i in temp]
df1 = pd.DataFrame(temp)
df1.columns=['ImageId']
df2 = pd.DataFrame(results)
df2.columns=['Label']
df=pd.concat([df1,df2],axis=1)
df.to_csv('results_nnet_6_310.csv', header=True,index=False)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    