# -*- coding: utf-8 -*-
"""
Created on Mercredi 29/08/2018

@author: a.teffal
"""

#importing packages
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from scipy import ndimage
from scipy import misc
import os
import numpy as np
import functools
import evaluation

#reading labels
labels = pd.read_csv('stage1_train_labels.csv',sep=',')

def removeRed(f,trashold=128):
    f2=f.copy()
    d1=f2.shape[0]    
    d2=f2.shape[1] 
    for i in range(0,d1):
        for j in range(0,d2):
            if f2[i,j,0]>trashold :
                f2[i,j,0]=0
                f2[i,j,1]=0
                f2[i,j,2]=0
            else:
                f2[i,j,0]=255
                f2[i,j,1]=255
                f2[i,j,2]=255
    return f2

def letRed(f,trashold=128):
    f2=f.copy()
    d1=f2.shape[0]    
    d2=f2.shape[1] 
    for i in range(0,d1):
        for j in range(0,d2):
            if f2[i,j,0]<=trashold:
                f2[i,j,0]=255
                f2[i,j,1]=0
                f2[i,j,2]=0
            else:
                f2[i,j,0]=255
                f2[i,j,1]=255
                f2[i,j,2]=255
    return f2


    
    
def getMask(x, width):
    h=int(len(x)/width)
    y=np.reshape(x,(h,width),order='F')
    return y

def run_length_encoded_old (x,height):
    temp=[]
    i=0
    added=False
    while i< len(x):
        indice = i+1
        count =0
        while i <len(x) and  x[i]==1:
            count=count+1
            i=i+1
            if ((i)%height)==0:
                break
        if count>0:
            temp.append([(indice-1)%height,int((indice-1)/height),str(indice)+' '+str(count)])
            added=True
        else:
            i=i+1
    #Sorting by line number
    if added==False:
        return ['']
    temp.sort(key = lambda x:(x[0],x[1]))
    
    #temp.sort(key=lambda x:x[1])
    temp2=[temp[0][2]]
    c=0
    for i in range(1,len(temp)):
        if (temp[i][1]-temp[i-1][1])==1:
            temp2[c]=temp2[c]+' ' + temp[i][2]
        else:
            temp2.append(temp[i][2])
            c=c+1
        
    #removing those with one packet (therefore thos containing one space)
    nn=len(temp2)
    i=0
    while i <nn:
        if temp2[i].count(' ')==1:
            temp2.pop(i)
            nn=nn-1
        else:
            i=i+1
    return temp2
    
    
def unrun_encode(x0,imageSize):
    y=np.zeros((imageSize,),dtype=bool)
    for w in x0:
        x=w.split(' ')
        x=[int(x[i]) for i in range(0,len(x)) ]
        for i in range(0,len(x),2):
            x[i]=x[i]-1
            y[x[i]:min((x[i]+x[i+1]),len(y)-1)] = [True]*(min((x[i]+x[i+1]),len(y)-1)-x[i])
        
    return y


def run_length_encoded (x,height):
    temp=[]
    i=0
    added=False
    width=int(len(x)/height)
    while i< len(x):
        indice = i+1
        count =0
        while i <len(x) and  x[i]==1:
            count=count+1
            i=i+1
            if ((i)%height)==0:
                break
        if count>0:
            temp.append([(indice-1)%height,int((indice-1)/height),str(indice)+' '+str(count),0])
            added=True
        else:
            i=i+1
    
    if added==False:
        return ['']
    
    
    
    n=len(temp)
    for i in range(0,n):
        w1=temp[i]
#        print(i, w1)
        if w1[2]!='': 
            n1 = int(w1[2][w1[2].find(' ')+1:])
            next_col=w1[1]+1
            while next_col <width:
                found=False
                for j in range(0,n):
                    w2=temp[j]
#                    print(i,j, w1,w2)
                    if w2[2]!='' and w2[3]==0:
                        n2 = int(w2[2][w2[2].find(' ')+1:])
                        if w2[1]==next_col and w2[3]==0 :
                            if (w1[0]>= w2[0] and w1[0]< (w2[0]+n2)) or (w2[0]>= w1[0] and w2[0]< (w1[0]+n1)):
                                w1[2]=w1[2]+' '+w2[2]
                                w2[2]=''
                                w2[3]=1
                                w1[3]=1
                                next_col=next_col+1
                                found=True
                    if found == True:
                        break
                if found == False:
                    break
    
    
    i=0
    while i<n:
        if temp[i][2]=='':
            temp.pop(i)
            n=n-1
        else:
            i=i+1
    
    temp2=[ss[2] for ss in temp]
    
    
    #removing those with one packet (therefore thos containing one space)
    nn=len(temp2)
    i=0
    while i <nn:
        if temp2[i].count(' ')==1:
            temp2.pop(i)
            nn=nn-1
        else:
            i=i+1
    return temp2
            
    
    
#def unrun_encode(x0,imageSize):
#    y=np.zeros((imageSize,),dtype=bool)
#    for w in x0:
#        x=w.split()
#        x=[int(x[i]) for i in range(0,len(x)) ]
#        for i in range(0,len(x),2):
#            y[x[i]:min((x[i]+x[i+1]),len(y)-1)] = [True]*(min((x[i]+x[i+1]),len(y)-1)-x[i])
#        
#    return y
    
    
def maximumDiff_0(x):
    x2=x.copy()
    x2.sort()
    maximum = 0
    for i in range(1,len(x2)):
        if abs(x2[i]-x2[i-1])>maximum:
            maximum = abs(x2[i]-x2[i-1])
    return maximum

def maximumDiff(x):
    x2=x.copy()
    #x2.sort()
    maximum = 0
    for i in range(1,len(x2)):
        if abs(x2[i]-x2[i-1])>maximum:
            maximum = abs(x2[i]-x2[i-1])
    return maximum


def binarize(x, trashold=128, h=0,Type='gray'):
    x2=x.copy()
    for i in range(0,len(x)):
        if x[i]>trashold :
            x2[i]=1
        else:
            x2[i]=0
    showDigitImage(x2,h,Type)
    
    return x2


def splitImage(x,diff,h=0,Type='gray'):
    n= len(x)
    x2=np.zeros((n,),dtype=int)
    if x[0]>150:
        x2[0]=1
    for i in range(1,n):
        if abs((x[i]-x[i-1]))/max(x[i-1],1)>diff:
            x2[i] = 1
        else:
            x2[i]=0
    showDigitImage(x2,h,Type)
    return x2 


       

def showImage(imageId, imageType='images'):
    zz=os.listdir('stage1_train/'+imageId+'/'+imageType)
    for w in zz:
        f = misc.imread('stage1_train/'+imageId+'/'+imageType +'/'+w,flatten=False)
        plt.imshow(f)
        plt.show()
        

def showDigitImage(x, h=0,Type='gray', order_='F'):
    if h<=0:
        dimension_1 = int(np.sqrt(len(x)))
        dimension_2=dimension_1
    else:
        dimension_1=h
        dimension_2=int(len(x)/h)
        
    print('(',dimension_1,dimension_2,')',sep=' ')
    plt.imshow(x.reshape((dimension_1,dimension_2),order=order_),cmap='gray')
    plt.show()
        
def featureImage(ImageId,folder='stage1_train/'):
    f = misc.imread(folder+ImageId+'/images/'+ImageId+'.png',flatten=False)
    if np.mean(f[:,:,0:1])>128:
        f= removeRed(f)
    #f=ski.exposure.equalize_hist(f)
    l=f.shape[0]
    array_image=f.flatten()
    array_image_mean=[]
    for i in range(0,len(array_image),f.shape[2]):
        #array_image_mean.append((0.0+array_image[i]+array_image[i+1]+array_image[i+2]+array_image[i+3])/4)
        array_image_mean.append((0.0+array_image[i]+array_image[i+1]+array_image[i+2]+0)/3)
    n=len(array_image_mean)
    image_features = []
    for i in range(0,len(array_image_mean)):
        #left pixel
        if i>=l:
            left=array_image_mean[i-l]
        else:
            left=0
        
        #right pixel
        if i<n-l:
            right=array_image_mean[i+l]
        else:
            right=0
            
        #top pixel
        if i%l !=0:
            top=array_image_mean[i-1]
        else:
            top=0
            
        #down pixel
        if (i+1)%l !=0:
            down=array_image_mean[i+1]
        else:
            down=0
            
        #left top pixel
        if top!=0 and left!=0:
            left_top=array_image_mean[i-l-1]
        else:
            left_top=0
            
        #right top pixel
        if top!=0 and right!=0:
            right_top=array_image_mean[i+l-1]
        else:
            right_top=0
            
        #left down pixel
        if down!=0 and left!=0:
            left_down=array_image_mean[i-l+1]
        else:
            left_down=0
            
        #right down pixel
        if down!=0 and right!=0:
            right_top=array_image_mean[i+l+1]
        else:
            right_top=0
        
            
#        image_features.append([array_image_mean[i],left,right,top,down,left_top,right_top,left_down,right_top])
#        image_features.append([left-array_image_mean[i],right-array_image_mean[i],top-array_image_mean[i], down-array_image_mean[i]])
        tre=10
        feature_1 = left-array_image_mean[i]
        feature_2 = right-array_image_mean[i]
        feature_3 = top-array_image_mean[i]
        feature_4 = down-array_image_mean[i]
         
        if abs(feature_1) <tre:
            feature_1=0
         
        if abs(feature_2) <tre:
            feature_2=0
             
        if abs(feature_3) <tre:
            feature_3=0
             
        if abs(feature_4) <tre:
            feature_4=0
             
         
        image_features.append([feature_1,feature_2,feature_3,feature_4])

    return np.array(image_features),f.shape[0],f.shape[1]


def featureImage_bis(ImageId,folder='stage1_train/'):
    f = misc.imread(folder+ImageId+'/images/'+ImageId+'.png',flatten=False)
    
    l=f.shape[0]
    m=f.shape[1]
    array_image=f.flatten()
    array_image_mean=[]
    for i in range(0,len(array_image),f.shape[2]):
        array_image_mean.append((0.0+array_image[i]+array_image[i+1]+array_image[i+2]+0)/3)
    n=len(array_image_mean)
    image_features = []
    for i in range(0,len(array_image_mean)):
        #left pixel
        if i>=l:
            left=array_image_mean[i-l]
        else:
            left=0
        
        #right pixel
        if i<n-l:
            right=array_image_mean[i+l]
        else:
            right=0
            
        #top pixel
        if i%l !=0:
            top=array_image_mean[i-1]
        else:
            top=0
            
        #down pixel
        if (i+1)%l !=0:
            down=array_image_mean[i+1]
        else:
            down=0
            
        #left top pixel
        if top!=0 and left!=0:
            left_top=array_image_mean[i-l-1]
        else:
            left_top=0
            
        #right top pixel
        if top!=0 and right!=0:
            right_top=array_image_mean[i+l-1]
        else:
            right_top=0
            
        #left down pixel
        if down!=0 and left!=0:
            left_down=array_image_mean[i-l+1]
        else:
            left_down=0
            
        #right down pixel
        if down!=0 and right!=0:
            right_top=array_image_mean[i+l+1]
        else:
            right_top=0
            
        #line number
        x1=i%l
        
        #column number
        x2=int(i/l)
        
            
        image_features.append([x1,x2,array_image_mean[i],left,right,top,down,left_top,right_top,left_down,right_top])
    
#    print('Feature image')
#    plt.imshow(f)
#    plt.show()
    return np.array(image_features),f.shape[0],f.shape[1]

def meanImage(ImageId, folder='stage1_train/'):
    f = misc.imread(folder+ImageId+'/images/'+ImageId+'.png',flatten=False)
    l=f.shape[0]
    array_image=f.flatten()
    array_image_mean=[]
    for i in range(0,len(array_image),f.shape[2]):
        #array_image_mean.append((0.0+array_image[i]+array_image[i+1]+array_image[i+2]+array_image[i+3])/4)
        array_image_mean.append((0.0+array_image[i]+array_image[i+1]+array_image[i+2]+0)/3)
    n=len(array_image_mean)
    image_features = []
    for i in range(0,len(array_image_mean)):
        #left pixel
        if i>=l:
            left=array_image_mean[i-l]
        else:
            left=0
        
        #right pixel
        if i<n-l:
            right=array_image_mean[i+l]
        else:
            right=0
            
        #top pixel
        if i%l !=0:
            top=array_image_mean[i-1]
        else:
            top=0
            
        #down pixel
        if (i+1)%l !=0:
            down=array_image_mean[i+1]
        else:
            down=0
            
        image_features.append([(array_image_mean[i]+left+right+top+down)/5])
    
    print('Mean image')
    plt.imshow(f)
    plt.show()
    return np.array(image_features),f.shape[0],f.shape[1]

def grayScaleImage(ImageId,folder='stage1_train/'):
    f = misc.imread(folder+ImageId+'/images/'+ImageId+'.png',flatten=False)
    if np.mean(f[:,:,0:1])>150:
        f= removeRed(f)
    
    l=f.shape[0]
    array_image=f.flatten()
    array_image_mean=[]
    for i in range(0,len(array_image),f.shape[2]):
        #array_image_mean.append((0.0+array_image[i]+array_image[i+1]+array_image[i+2]+array_image[i+3])/4)
        array_image_mean.append((0.0+array_image[i]+array_image[i+1]+array_image[i+2]+0)/3)
    n=len(array_image_mean)
    image_features = []
    for i in range(0,len(array_image_mean)):
        image_features.append([array_image_mean[i]])
    
#    print('Gray scale image : ',ImageId)
#    plt.imshow(f)
#    plt.show()
    return np.array(image_features),f.shape[0],f.shape[1]


def labelImage(ImageId,imageSize):
    y=np.zeros((imageSize,),dtype=int)
    temp=list(labels[labels['ImageId']==ImageId]['EncodedPixels'])
    for w in temp:
        x=w.split()
        x=[int(x[i]) for i in range(0,len(x)) ]
        for i in range(0,len(x),2):
            #y[x[i]:(x[i]+x[i+1])] = [1]*x[i+1]
            y[x[i]:min((x[i]+x[i+1]),len(y)-1)] = [1]*(min((x[i]+x[i+1]),len(y)-1)-x[i])
        
    return y

def showPredictedImage(ImageId,heigh,width,df_):
    imageSize=heigh*width
    y=np.zeros((imageSize,),dtype=bool)
    temp=list(df_[df_['ImageId']==ImageId]['EncodedPixels'])
    for w in temp:
        x=w.split()
        x=[int(x[i]) for i in range(0,len(x)) ]
        for i in range(0,len(x),2):
            y[x[i]:min((x[i]+x[i+1]),len(y)-1)] = [True]*(min((x[i]+x[i+1]),len(y)-1)-x[i])
        
    showDigitImage(y,heigh,order_='F')
    return y

def showMask(s,heigh,width):
    y=unrun_encode([s],heigh*width)
    showDigitImage(y,heigh,order_='F')
    
def supperpose_images(true_masks,predicted_masks,heigh,width):
    n=heigh*width
    y=np.zeros((n,),dtype=int)
    for i in range(0,n):
        if predicted_masks[i]==1:
            y[i]=128
        if true_masks[i]==1:
            y[i]=255
        
    showDigitImage(y,heigh,order_='F')
    
    
def train_one_image(w, type_model='feature',folder='stage1_train/'):
    if type_model=='feature':
        int_centers=np.array([[0,0,0,0,0,0,0,0,0 ],[255,255,255,255,255,255,255,255,255 ]])
        X, heigh, width = featureImage(w,folder)
    else:
        int_centers=np.array([[0],[255]])
        X, heigh, width = grayScaleImage(w,folder)
    
    
    kmeans = KMeans(n_clusters=2,init=int_centers)
    
    predictions=[]

    n=int(width/4)
    
    n1=n*heigh
    
    n2=2*n*heigh
    
    n3=3*n*heigh
    
    # Getting the cluster labels
    kmeans.fit(X[:n1,])
    
    X_predictions_1 = kmeans.predict(X[:n1,])
    
    kmeans.fit(X[n1:n2,])
    
    X_predictions_2 = kmeans.predict(X[n1:n2,])
    
    kmeans.fit(X[n2:n3,])
    
    X_predictions_3 = kmeans.predict(X[n2:n3,])
    
    kmeans.fit(X[n3:,])
    
    X_predictions_4 = kmeans.predict(X[n3:,])
    
    
    X_predictions = np.hstack((X_predictions_1,X_predictions_2,X_predictions_3,X_predictions_4))
    
    temp=run_length_encoded(X_predictions.reshape((heigh,width)).flatten(order='F'),heigh)
    for z in temp:
        predictions.append([w,z])
    #Export results
    df=pd.DataFrame(predictions,columns=['ImageId','EncodedPixels'])
    df.to_csv('bowl_2018_km_'+w+'.csv', header=True,index=False)
    y_=labelPredictedImage(w,heigh*width,df)
    showDigitImage(y_,heigh)
    return df, heigh, width


    


test_data=pd.read_csv('stage1_sample_submission.csv')

#Liste of folders in stage1_train
zz=os.listdir('stage1_train')


#K-means
from sklearn.cluster import KMeans

#Logistic regression
from sklearn import linear_model, datasets

#Knn
from sklearn.neighbors import KNeighborsClassifier



knn=KNeighborsClassifier(n_neighbors=1)

#zz=['01d44a26f6680c42ba94c9bc6339228579a95d0e2695b149b7cc0c9592b21baf']
# Number of clusters
int_centers=np.array([[0,0,0,0,0,0,0,0,0 ],[255,255,255,255,255,255,255,255,255 ]])

#int_centers=np.array([[0],[255]])
kmeans = KMeans(n_clusters=2,init=int_centers)
logreg = linear_model.LogisticRegression(C=100,fit_intercept=False)
predictions=[]

folder = 'stage1_train/'

X, heigh, width = featureImage(zz[0],folder)

if folder=='stage1_train/':
    y=labelImage(zz[0],heigh*width)

for w in zz[1:10]:
    X_temp, heigh, width = featureImage(w,folder)
    X=np.vstack((X,X_temp))
    if folder=='stage1_train/':
        y_temp=labelImage(w,heigh*width)
        y=np.hstack((y,y_temp))
        
    
logreg.fit(X, y)


zz=['01d44a26f6680c42ba94c9bc6339228579a95d0e2695b149b7cc0c9592b21baf']

for w in zz:
    X, heigh, width = featureImage(w,folder)
    
    if folder=='stage1_train/':
        y=labelImage(w,heigh*width)
    

#    kmeans.fit(X)
    
#    logreg.fit(X, y)
    
#    X_predictions = kmeans.predict(X)
     
    X_predictions= logreg.predict(X)
    
    if folder=='stage1_train/':
        print('predicted vs true masks ')
        supperpose_images(y,X_predictions.reshape((heigh,width)).flatten(order='F'),heigh,width)
        print('predicted masks ')
        showDigitImage(X_predictions.reshape((heigh,width)).flatten(order='F'),heigh)
    print('true masks')
    showDigitImage(y,heigh)
    
#    X_predictions = np.hstack((X_predictions_1,X_predictions_2,X_predictions_3,X_predictions_4))
    
    temp=run_length_encoded(X_predictions.reshape((heigh,width)).flatten(order='F'),heigh)
    for z in temp:
        predictions.append([w,z])
    #predictions.append(run_length_encoded(X_predictions))
    
#Export results
df=pd.DataFrame(predictions,columns=['ImageId','EncodedPixels'])


tresh=(0.8,)
moyennes = {}



for w in zz:
    pred_masks=list(df[df['ImageId']==w]['EncodedPixels'])
    true_masks=list(labels[labels['ImageId']==w]['EncodedPixels'])
    av, tp,fp,fn=average_precision(pred_masks,true_masks,tresh)
    moyennes[w]=[av,fp,fn]
    
    
    

print('Moyenne sur toutes les images : ',np.mean([ii[0] for ii in moyennes.values()]))


df.to_csv('bowl_2018_km_20180326_test.csv', header=True,index=False)
