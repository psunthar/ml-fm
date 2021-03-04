from scipy.ndimage import rotate
from PIL import Image
import numpy as np
from numpy import asarray
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm


a=np.ones((1000,999,3),np.uint8)
a.fill(255)
plt.imsave('blank.png', a, cmap=cm.gray)

for r in range(1,10):
    angle=np.random.randint(1,90)
    canvas=Image.open("blank.png")
    #np.set_printoptions(threshold=np.inf)
    im=Image.open("image"+str(r)+".png")
#im.show()
    width,height=im.size
   
#im.save("test.png")
    width,height=im.size
#print(width)
#print(height)
    sub_img=im.crop(box=(10,200,700,800))
    sub_img=sub_img.rotate(angle)

    canvas.paste(sub_img,box=(50,200))

    canvas.save("test2.png")
#print(canvas.size)
#arr=np.array(canvas)
    alpha = canvas.convert('RGBA').split()[-1]
    bg_colour=(255, 255, 255)
        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
    bg = Image.new("RGBA", canvas.size, bg_colour + (255,))
    bg.paste(canvas, mask=alpha)
    bg.save("test3.png")



    mg = Image.open("test3.png").convert('L')
    data=np.array(mg)
    #print(data.shape)

#data = ~data  # invert B&W
    data[data > 0] = 1
   # print(data)
    plt.imsave('final'+str(r)+'.png', np.array(data), cmap=cm.gray)
    if r==1:
        a=data
    else:

        a=np.dstack((a,data))
np.save('rotate_data',a)




