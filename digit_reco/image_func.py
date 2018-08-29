# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 09:08:45 2018

@author: a.teffal
"""

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
