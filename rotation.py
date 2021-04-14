from scipy.ndimage import rotate
from PIL import Image
import numpy as np
from numpy import asarray
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm


a=np.ones((1798,1799,3))
a.fill(1)

a=a.astype('float64')
print(type(a))
plt.imsave('blank.png', a, cmap=cm.gray)


for r in range(1,800):
    angle=np.random.randint(1,90)
    canvas=Image.open("blank.png")
    
    im=Image.open("image"+str(r)+".png")

    width,height=im.size
    print(width)
    print(height)

    #width,height=im.size

    sub_img=im.crop(box=(5,100,1550,1600))#(left,upper,right,lower)[0,0] is top left
    sub_img=sub_img.rotate(angle)

    canvas.paste(sub_img,box=(50,200))

    canvas.save("test2.png")

    alpha = canvas.convert('RGBA').split()[-1]
    bg_colour=(255, 255, 255)
        
    bg = Image.new("RGBA", canvas.size, bg_colour + (255,))
    bg.paste(canvas, mask=alpha)
    bg.save("test3.png")



    mg = Image.open("test3.png").convert('L')
    data=np.array(mg)
    


    data[data > 0] = 1
  
    data=data.astype('float64')

    plt.imsave('final'+str(r)+'.png', np.array(data), cmap=cm.gray)
    if r==1:
        a=data
    else:

        a=np.dstack((a,data))
np.save('rotate_data',a)




