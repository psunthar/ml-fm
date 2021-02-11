import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy import save
imageH=4
AR1=[]
np.array(AR1)
for r in range(1,100):
    l1=np.random.uniform(1,5)   #upto stenosis
    l=np.random.uniform(2,20)   #stenosis
    Tl=np.random.uniform(40,50) #total length
    
    AR=np.random.uniform(1.1,7)
    AR1.append(AR)
    d0=np.random.uniform(0.5,4)
    d1=(d0)/np.sqrt(AR)
    k=imageH-d0
    k1=imageH-d1

    def f1(x):
        if x<l1:
            return k
        if x>=l1 and x<l1+l:
            return k1
        if x>=l1+l:
            return k
        
    dx=0.1
    #img=np.ones((600,100))
    img=np.zeros((int(Tl/dx),int(imageH/dx)))

    for i in range(int(Tl/dx)):
        x=i*dx
        for j in range(int(f1(x)/dx)):
       # print([i,j,f1(x)])
            img[i,j]=1
    
    b=np.ones((int(700-Tl/dx),40))  
    img=np.append(img,b,axis=0)        #adding rows of zeros at the end of matrix to give uniform shape
    #img[:-1,:]=a
    print(np.shape(img))
    img1=img.T
    plt.imsave('filename.png', np.array(img1), cmap=cm.gray)
    img2=np.flip(img1,0)
    img3=np.concatenate((img1, img2))
    if r==1:
        a=img3
    else:
        
        a=np.dstack((a,img3))
    
       # print(np.shape((a)))
    plt.imsave('image' +str(r)+'.png', np.array(img3), cmap=cm.gray)
print(np.shape(img3))
save('data', a)
save('AR_data',AR1)

