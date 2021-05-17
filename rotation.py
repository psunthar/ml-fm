from scipy.ndimage import rotate
from PIL import Image
import numpy as np
from numpy import asarray
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm


a=np.ones((600,600,3))
a.fill(1)

#a=a.astype('float32')
#print(type(a))
plt.imsave('blank.png', a, cmap=cm.gray)


for r in range(1,5000):
    angle=np.random.randint(1,90)
    canvas=Image.open("blank.png")
    
    im=Image.open("image"+str(r)+".png")

    width,height=im.size
    #print(width)
    #print(height)

    #width,height=im.size

    sub_img=im.crop(box=(5,100,550,550))#(left,upper,right,lower)[0,0] is top left
    if r%10==0:

        sub_img=sub_img.rotate(0)
    else:
        sub_img=sub_img.rotate(angle)


    canvas.paste(sub_img,box=(2,10))

    canvas.save("test2.png")

    alpha = canvas.convert('RGBA').split()[-1]
    bg_colour=(255, 255, 255)
        
    bg = Image.new("RGBA", canvas.size, bg_colour + (255,))
    bg.paste(canvas, mask=alpha)
    bg.save("test3.png")



    mg = Image.open("test3.png").convert('L')
    data=np.array(mg)
    


    data[data > 0] = 1
  
    data=data.astype('float32')

    #plt.imsave('final'+str(r)+'.png', np.array(data), cmap=cm.gray)
    if r==1:
        a=data
    else:

        a=np.dstack((a,data))
np.save('rotate_data',a)
np.save('rotate_data_L100_1',a)




