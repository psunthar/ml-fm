import numpy as np	
import sklearn
from sklearn.model_selection import train_test_split
from numpy import load
# load array
data = load('data.npy')

Y=np.ones(99)
Y=np.dstack(Y)
print(np.shape(data.T))
print(np.shape(Y.T))
X_train, X_test, y_train, y_test =train_test_split(data.T,Y.T,test_size=0.3)
