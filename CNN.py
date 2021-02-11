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
import matplotlib.pyplot as plt
from numpy import save
Xdata = load('data.npy')
Ydata = load('AR_data.npy')
Y=np.dstack(Ydata)
#print(np.shape(data.T))
#print(np.shape(Y.T))
X_train, X_test, y_train, y_test =train_test_split(Xdata.T,Y.T,test_size=0.3)
print(np.shape(X_train))
X_train = X_train.reshape(-1, 699, 80, 1)
X_test = X_test.reshape(-1, 699, 80, 1)
model =Sequential()
model.add(Conv2D(3,(10,10), activation= 'relu',kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='linear'))
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,loss='mse')
model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),epochs=10)
sol=model.predict(X_test) 
print(sol)
save('prediction',sol)        # saving predicted results
save('testData',y_test)
#plt.scatter(y_test.reshape(30),sol.reshape(30))
#plt.xlabel('original')
#plt.ylabel('predicted')
#plt.show()
