########################## MAIN FILE#############################

import numpy as np
import pandas as pd

n=10000
bv=np.zeros((n))
d1=np.zeros((n))

visc=np.zeros((n))
dens=np.zeros((n))
R=np.zeros((n))

bv_l=.1
bv_m=.5

d1_l=.001
d1_m=.007

dens_l=1000
dens_m=1080

R_l=2.6
R_m=2.6

visc_l=0.0035
visc_m=0.0055


bv= np.random.uniform(bv_l,bv_m,n)
d1=np.random.uniform(d1_l,d1_m,n)
dens= np.random.uniform(dens_l,dens_m,n)
R= np.random.uniform(R_l,R_m,n)
visc=np.random.uniform(visc_l,visc_m,n)
#print(R)


data=pd.DataFrame(data=bv,columns=['velocity'])

data['dia1']=d1
data['density']=dens
data['R']=R
data['viscosity']=visc
d2=d1*R
data['dia2']=d2
#R=2.6

a=-7+15.1*R-2.1*R**2
b=3-3.66*R+.465*R**2
c=-1.29+1.48*R-0.188*R**2

Re1=dens*bv*d1
Re=np.divide(Re1,visc)
data['Re']=Re



#print(testing_from_Re)
c1=np.divide(a,Re)
c2=np.divide(b,Re**(0.5))
ci=c1+c2+c
#data['Ci']=ci

del_p=ci*(0.5*dens*bv**2)
data['del_p']=del_p
#print(data)
dataX=data[data['Re']<35]
#print(data1)
dataY=data[data['Re']>600]

testing_from_Re=pd.concat([dataX,dataY],axis=0)
testing_from_Re.to_csv('testing_fro_Re.csv',index=False)

#print(testing_from_Re)
data2=data[(data['Re']>=35) & ( data['Re']<=600)]

data['Ci']=ci
data.to_csv('Synthetic_data.csv',index=False)
#data.drop(columns=['R','Re'],axis=1,inplace=True)   
data2.to_csv('training1.csv',index=False)


