import numpy as np	
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
data = load('data.npy')

Y=np.ones(99)
Y=np.dstack(Y)
#print(np.shape(data.T))
#print(np.shape(Y.T))
X_train, X_test, y_train, y_test =train_test_split(data.T,Y.T,test_size=0.3)
print(np.shape(X_train))
X_train = X_train.reshape(-1, 699, 80, 1)
model =Sequential()
model.add(Conv2D(3,(10,10), activation= 'relu',kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='linear'))
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='mse')
model.fit(x=X_train,y=y_train,epochs=10)
X_test = X_test.reshape(-1, 699, 80, 1)
sol=model.predict(X_test) 
print(sol)
