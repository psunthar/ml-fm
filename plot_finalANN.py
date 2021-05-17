import numpy as np
from numpy import load
import matplotlib.pyplot as plt


predicted=load('prediction_finalANN.npy')    #predicted data
target=load('target_Data_finalANN.npy') 

plt.figure(0)
plt.scatter(target,predicted)
plt.xlabel('original_del(P)')
plt.ylabel('predicted_del(P)')
plt.savefig('final_prediction.png')
print(np.sqrt(((predicted-target)**2).mean()))
#print(mse(target,predicted,squared=False))
rmspe = (np.sqrt(np.mean(np.square((target - predicted) / target)))) * 100
print(rmspe)


