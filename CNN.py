
import os
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
from keras import regularizers
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
df=pd.read_csv("3_outputs.csv")
df=df.values

Y=np.dstack(df)

X_train, X_test, y_train, y_test =train_test_split(Xdata.T,Y.T,test_size=0.1)
print(np.shape(X_train))
X_train = X_train.reshape(-1, 600, 600, 1)
X_test = X_test.reshape(-1, 600, 600, 1)
model =Sequential()
model.add(Conv2D(4,(5,5),activation= 'relu'))
#model.add(Dropout(0.5))
#model.add(BatchNormalization())
#model.add(MaxPooling2D((2,2)))
model.add(Conv2D(8,(5,5),activation= 'relu'))
#model.add(BatchNormalization())

#model.add(Dropout(0.2))
#model.add(MaxPooling2D((2,2)))
model.add(Conv2D(16,(3,3), activation= 'relu'))
model.add(MaxPooling2D((2,2)))
#model.add(BatchNormalization())

model.add(Conv2D(32,(3,3), activation= 'relu'))
model.add(MaxPooling2D((2,2)))
#model.add(BatchNormalization())

model.add(Conv2D(64,(3,3), activation= 'relu'))
model.add(MaxPooling2D((2,2)))
#model.add(BatchNormalization())

model.add(Conv2D(128,(1,1), activation= 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dense(20,activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(10,activation='relu'))
model.add(Dense(3, activation='linear'))
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss='mse')
model.fit(x=X_train,y=y_train,batch_size=16,validation_data=(X_test,y_test),epochs=90)
sol=model.predict(X_test) 
print(sol)
save('prediction',sol)        # saving predicted results
save('testData',y_test)




