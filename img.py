import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
imageH=4
for r in range(1,100):
    l=np.random.uniform(2,20)
    Tl=np.random.uniform(22,50)
    l1=2
    AR=np.random.uniform(1.1,7)
    d0=2
    d1=(d0)/np.sqrt(AR)
    k=imageH-d0
    k1=imageH-d1



    def f1(x):
        if x<2:
            return k
        if x>=2 and x<2+l:
            return k1
        if x>=2+l:
            return k
        


    xmax=Tl
    dx=0.1
    img=np.zeros((int(Tl/dx),int(imageH/dx)))

    for i in range(int(xmax/dx)):
        x=i*dx
        for j in range(int(f1(x)/dx)):
       # print([i,j,f1(x)])
            img[i,j]=1

#print(img)
    img1=img.T

#print(img2)
    plt.imsave('filename.png', np.array(img1), cmap=cm.gray)
    img2=np.flip(img1,0)
    img3=np.concatenate((img1, img2))

    plt.imsave('image' +str(r)+'.png', np.array(img3), cmap=cm.gray)
