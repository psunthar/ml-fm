import numpy as np
import pandas as pd

#df=pd.read_csv("4_outputs.csv")
#df['Stenosis_len']=df['Stenosis_length']*0.0004
#df['D00']=df['D0']*0.0004
#df['D11']=df['D1']*0.0004

Pa=np.random.uniform(10640,13300,2000)
#df['Pa']=Pa
Pv=np.random.uniform(665,665,2000)
Pv=665
Kt=1.52
D0= np.random.uniform(0.002,0.004,2000)
Do=1000*D0
    
D1=np.random.uniform(0.4,0.9,2000)
D1=D1*D0

L=np.random.uniform(.1,.15,2000)
Lo=1000*L
Rv=20000/((Do*Do)*Lo)
Rv=133e6*Rv
#df['Rv']=Rv
A0=(np.pi/4)*(D0**2)
A1=(np.pi/4)*(D1**2)
#A0=A0
#df['A1']=A1
AR=A0/A1
Ps=1-(1/AR)



Ls=np.random.uniform(1,4)
Ls=Ls*D0
Kv=(32*(0.83*L+1.64*D1)/D0)*(A0/A1)**2
Kt=1.52
rho=1000
mu=.004




a=((mu**2)*Kt/(2*rho*(D0**2)))*((AR-1)**2)
b=(Kv*(mu**2)/(rho*(D0**2))) +Rv*A0*mu/(D0*rho)
c=Pv-Pa
d = (b**2) - (4*a*c)

sol1 = (-b-np.sqrt(d))/(2*a)
sol2 = (-b+np.sqrt(d))/(2*a)
print('The solution are {0} and {1}'.format(sol1,sol2))
Re=sol2
Rs=(Re*mu/(A0*D0))*((Kv/Re) +((Kt/2)*((AR -1)**2)))
#print(Re*mu*A0/(rho*D0))
Pd= Pa-(Rs*(Re*mu*A0/(rho*D0)))
target_variable=(Pd-Pv)/(Pa-Pv)
#    TG=np.append(TG,tg)
#    PD=np.append(PD,Pd)
#    rs=np.append(rs,Rs)
#    rv=np.append(rv,Rv)
#    re=np.append(re,Re)
#    pa=np.append(pa,Pa)

   # print(A)
   # print(Rs)
   # print(Rv)   # print(Re)
df=pd.DataFrame(data=Rs,columns=['Rs'])
df['L']=L
df['Ls']=Ls
df['D0']=D0
df['D1']=D1
df['Pv']=Pv
df['Pd']=Pd
df['PS']=Ps
df['Rv']=Rv
df['Re']=Re
df['Pa']=Pa
df['Target_variable']=target_variable
#drop(['Stenosis_length','D0','D1','Do','Lo','A0','A1','AR','Ps'])
df.to_csv("data.csv",index=False)
print(df.head())
