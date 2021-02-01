################################################### PLOT FILE ##################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import savefig
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import chisquare


predicted=pd.read_csv('predict.csv')
actual=pd.read_csv('testing_Y.csv')
actual1=pd.read_csv('Y_test1.csv')
predicted1=pd.read_csv('predict1.csv')


sol=pd.concat([predicted,actual],axis=1)
sol1=pd.concat([predicted1,actual1],axis=1)

#soldf=pd.DataFrame(sol,columns=['Predicted'])
#soldf['Actual']=y_test
sns.scatterplot(x='del_p',y='predict',data=sol,label='Training_Range')
#plt.legend(['Training_range'])
sns.scatterplot(x='del_p',y='predict',data=sol1,label='Outside_Training')

#####################
f=np.linspace(0,200,200)
plt.plot(f,f)
#plt.legend([''],[''])
plt.show()

#########################################################################################################3

abra=pd.read_csv('X_test_for_ci_plot.csv')
kadabra=pd.read_csv('X_test1_for_ci_plot.csv')
pci=np.divide(predicted['predict'],(0.5*abra['density']*abra['velocity']**2))
pci1=np.divide(predicted1['predict'],(0.5*kadabra['density']*kadabra['velocity']**2))
abra['Ci']=pci

abra.to_csv('abra.csv')

kadabra['Ci']=pci1
kadabra.to_csv('kadabra.csv')
#print(ci)
#abra=pd.concat([ci,abra],axis=1)
#kadabra=pd.concat([ci1,kadabra],axis=1)
#print(abra)
N=sns.scatterplot(x='Re',y='Ci',data=abra,label='Training_range')
M=sns.scatterplot(x='Re',y='Ci',data=kadabra,label='outside_range')
N.set(xscale='log')
M.set(xscale='log')
#plt.show()


#############################################################################################################
Re=np.logspace(1,3,50)

#print(a)

R=2.6
a=-7+15.1*R-2.1*R**2
b=3-3.66*R+.465*R**2
c=-1.29+1.48*R-0.188*R**2

c1=np.divide(a,Re)
c2=np.divide(b,(Re**(0.5)))
ci=c1+c2+c

plt.plot(Re,ci)
plt.xscale('log')
#plt.yscale('log')
#plt.ylim([1, 4])
plt.xlabel('Re')
plt.ylabel('Ci')
plt.show()
#print("Training Range R^2 value")
#r2_score(ci,abra['Ci'].values)
#print("Outside Training range R^2 value")

##############################################################################################################
losses=pd.read_csv('losses.csv')
plt.plot(losses['test_loss'])
plt.plot(losses['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()
#################################################################################################################
aCi=pd.read_csv('r_square_test.csv')        #actual Ci values
aCi1=pd.read_csv('r_square_test1.csv')      #actual Ci1 values
aCi=aCi.to_numpy()
pci=pci.to_numpy()
aCi1=aCi1.to_numpy()
pci1=pci1.to_numpy()
#X=((aCi-pci)**2)/(aCi)
#print(X)
#print(np.sum(X))

aCi=aCi.reshape(934)
aCi1=aCi1.reshape(669)
e=aCi-pci
error=np.divide(e,aCi)
error=np.square(error)
print(sum(error)/934)
e1=aCi1-pci1
error1=np.divide(e1,aCi1)
error1=np.square(error1)
print(sum(error1)/669)


