import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy import save
from scipy.ndimage import rotate
from PIL import Image
imageH=50
AR1=[]
np.array(AR1)
for r in range(1,10):
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
    img=np.zeros((int(Tl/dx),int(imageH/dx)))

    for i in range(int(Tl/dx)):
        x=i*dx
        for j in range(int(f1(x)/dx)):
            img[i,j]=1
    #img=rotate(img,30)
    space_added=int(1000-(Tl/dx) -100)   #white space to be added in front
   # space_added=int(space_added/2)   # white space to be added in back

    b=np.ones((space_added,500))
    c=np.ones((100,500))
    img=np.append(img,b,axis=0)              #adding rows of ones(white spaces) at the end of matrix to give uniform shape
    img=np.append(c,img,axis=0)
    img1=img.T
    plt.imsave('filename.png', np.array(img1), cmap=cm.gray)
    img2=np.flip(img1,0)
    img3=np.concatenate((img1, img2))

    plt.imsave('image' +str(r)+'.png', np.array(img3), cmap=cm.gray)

    im=Image.open("image1.png")
    #im.save("test.png")
    width,height=im.size
    print(width)
    print(height)   
    sub_img=im.crop(box=(0,280,500,420)).rotate(10)
    #sub_img.save("test1.png")
    im.paste(sub_img,box=(0,300))
    im.save("test2.png")

    if r==1:
        a=img3
    else:
        
        a=np.dstack((a,img3))
    
    plt.imsave('image' +str(r)+'.png', np.array(img3), cmap=cm.gray)
#print(np.shape(img3))
save('data', a)
save('AR_data',AR1)

