mport os
#import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from numpy import load
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import matplotlib.pyplot as plt
from numpy import save
from keras.layers import BatchNormalization

############################################
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

##############################################

#import os
#import tensorflow as tf
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"




Xdata = load('rotate_data.npy')
df=pd.read_csv("4_outputs.csv")
df=df.values

Y=np.dstack(df)

X_train, X_test, y_train, y_test =train_test_split(Xdata.T,Y.T,test_size=0.3)
print(np.shape(X_train))
X_train = X_train.reshape(-1, 1000, 999, 1)
X_test = X_test.reshape(-1, 1000, 999, 1)
model =Sequential()
model.add(Conv2D(4,(9,9),activation= 'relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(8,(7,7),activation= 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(16,(5,5), activation= 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(16,(5,5), activation= 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32,(3,3), activation= 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(1,1), activation= 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(4, activation='linear'))
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss='mse')
model.fit(x=X_train,y=y_train,batch_size=64,validation_data=(X_test,y_test),epochs=100)
sol=model.predict(X_test)
print(sol)
save('prediction',sol)        # saving predicted results
save('testData',y_test)
#plt.scatter(y_test.reshape(30),sol.reshape(30))
#plt.xlabel('original')
#plt.ylabel('predicted')
#plt.show()

