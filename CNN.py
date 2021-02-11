import numpy as np	
import sklearn
from sklearn.model_selection import train_test_split
from numpy import load
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# load array
data = load('data.npy')

Y=np.ones(99)
Y=np.dstack(Y)
print(np.shape(data.T))
print(np.shape(Y.T))
X_train, X_test, y_train, y_test =train_test_split(data.T,Y.T,test_size=0.3)
model =Sequential()
model.add(Conv2D(32,(3,3), activation= 'relu',kernel_initialiser='he_uniform', input_shape=((500,80,1))

