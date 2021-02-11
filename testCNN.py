import numpy as np
from numpy import load
import matplotlib.pyplot as plt

sol=load('prediction.npy')    #predicted data
y_test=load('testData.npy')          #testing data
plt.scatter(y_test.reshape(30),sol.reshape(30))
plt.xlabel('original')
plt.ylabel('predicted')
plt.show()

