import numpy as np
from numpy import load
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error as mse
sol=load('prediction.npy')    #predicted data
y_test=load('testData.npy')          #testing data
sl=len(sol)
yl=len(y_test)
predicted=sol.reshape((sl,3))
target=y_test.reshape((yl,3))
plt.figure(0)
#plt.scatter(target[:,0],predicted[:,0])
#plt.xlabel('original_Ts')
#plt.ylabel('Predicted_Ts')
#plt.savefig('TLength.png')

#plt.figure(1)
plt.scatter(target[:,0],predicted[:,0])
plt.xlabel('original_Ls')
plt.ylabel('Predicted_Ls')
plt.savefig('stenosisL.png')


plt.figure(1)
plt.scatter(target[:,1],predicted[:,1])
plt.xlabel('original_D0')
plt.ylabel('predicted_D0')
plt.savefig('D0.png')

plt.figure(2)
plt.scatter(target[:,2],predicted[:,2])
plt.xlabel('original_D1')
plt.ylabel('predicted_D1')
plt.savefig('D1.png')

#print(predicted)
#print(target)
for i in range(3):
    print(np.sqrt(((predicted[:,i]-target[:,i])**2).mean()))
#print(mse(target,predicted,squared=False))
    rmspe = (np.sqrt(np.mean(np.square((target[:,i] - predicted[:,i]) / target[:,i])))) * 100
    print(rmspe)
~                  
~                       
