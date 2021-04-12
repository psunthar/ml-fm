import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df=pd.read_csv('data.csv')
Y=df['Target_variable'].values
X=df.drop(['Re','Rs','Pd','PS','Target_variable'], axis=1)
#print(X)
X=X.values
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=.2,random_state=101)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

pca=PCA()
pca.fit(X_train)
X_train=pca.transform(X_train)
print(pca.explained_variance_ratio_)

X_test=pca.transform(X_test)

~                                
