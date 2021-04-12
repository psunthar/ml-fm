import numpy as np
from numpy import load
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error as mse
sol=load('prediction.npy')    #predicted data
y_test=load('testData.npy')          #testing data
sl=len(sol)
yl=len(y_test)
predicted=sol.reshape((sl,4))
target=y_test.reshape((yl,4))
plt.figure(0)
plt.scatter(predicted[:,0],target[:,0])
plt.xlabel('predicted_TL')
plt.ylabel('original_TL')
plt.savefig('TLength.png')

plt.figure(1)
plt.scatter(predicted[:,1],target[:,1])
plt.xlabel('predicted_Ls')
plt.ylabel('original_Ls')
plt.savefig('stenosisL.png')


plt.figure(2)
plt.scatter(predicted[:,2],target[:,2])
plt.xlabel('predicted_D0')
plt.ylabel('original_D0')
plt.savefig('D0.png')

plt.figure(3)
plt.scatter(predicted[:,3],target[:,3])
plt.xlabel('predicted_D1')
plt.ylabel('original_D1')
plt.savefig('D1.png')

#print(predicted)
#print(target)
print(np.sqrt(((predicted-target)**2).mean()))
#print(mse(target,predicted,squared=False))
rmspe = (np.sqrt(np.mean(np.square((target - predicted) / target)))) * 100
print(rmspe)
~                       
