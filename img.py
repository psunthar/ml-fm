import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from numpy import save
imageH=.1
TL=[]
np.array(TL)
Ls=[]
np.array(Ls)
D0=[]
np.array(D0)
D1=[]
np.array(D1)

for r in range(1,5000):
    #total length

    #AR=np.random.uniform(1.1,4)
    d0=np.random.uniform(0.002,.004)
    d1=np.random.uniform(.4,.9)
    d1=d1*d0
    l1=np.random.uniform(1,5)   
    l=np.random.uniform(1,4) 
    l=l*d0
    Tl=np.random.uniform(.1,.15)
    l1=0.20*Tl          #upto stenosis
    k=imageH-d0
    k1=imageH-d1
    dx=0.0004
   # Tot_l=Tl/dx
    TL=np.append(TL,Tl)
   # Ls=np.append(Ls,l)
    dx=0.0004
    d0=int(d0/dx)
    d1=int(d1/dx)
    ls=int(l/dx)
    Ls=np.append(Ls,ls)
    D0=np.append(D0,d0)
    D1=np.append(D1,d1)


    def f1(x):
        if x<l1:
            return k
        if x>=l1 and x<l1+l:
            return k1
        if x>=l1+l:
            return k

   # dx=0.0004
    #img=np.ones((600,100))
    img=np.zeros((int(Tl/dx),int(imageH/dx)))

    for i in range(int(Tl/dx)):
        x=i*dx
        for j in range(int(f1(x)/dx)):
       
            img[i,j]=1
    
    
    #vertical image is generated now  
    space_added=int(500-(Tl/dx)-50)   #white space to be added in front
    b=np.ones((space_added,250))
    c=np.ones((50,250))
    img=np.append(img,b,axis=0)        #adding rows of zeros at the end of matrix to give uniform shape
    img=np.append(c,img,axis=0)        #adding at front
   #img[:-1,:]=a
    
    img1=img.T #horizontal image is generated
   # print(np.shape(img))
    plt.imsave('filename.png', np.array(img1), cmap=cm.gray)
    img2=np.flip(img1,0)
    img3=np.concatenate((img1, img2))
    if r==1:
        a=img3
    else:

        a=np.dstack((a,img3))

       # print(np.shape((a)))
    plt.imsave('image' +str(r)+'.png', np.array(img3), cmap=cm.gray)
#print(np.shape(img3))
save('data', a)
df=pd.DataFrame(data=Ls,columns=['Stenosis_length'])
df1=pd.DataFrame(data=TL, columns=['Total_Length'])
#df['Stenosis_length']=Ls
df['D0']=D0
df['D1']=D1
df.to_csv("3_outputs.csv",index=False)
df1.to_csv("1_output1.csv",index=False)



    
