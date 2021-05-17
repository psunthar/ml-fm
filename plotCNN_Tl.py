import numpy as np
from numpy import load
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error as mse
sol=load('prediction.npy')    #predicted data
y_test=load('testData.npy')          #testing data
sl=len(sol)
yl=len(y_test)
predicted=sol
target=y_test
plt.figure(0)
plt.scatter(target,predicted)
plt.xlabel('original_Tl')
plt.ylabel('Predicted_Tl')
plt.savefig('TLength.png')



#print(predicted)
#print(target)

print(np.sqrt(((predicted-target)**2).mean()))
#print(mse(target,predicted,squared=False))
rmspe = (np.sqrt(np.mean(np.square((target - predicted) / target)))) * 100
print(rmspe)

