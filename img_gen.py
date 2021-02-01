import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageOps

def f(x):

    return x

xmax=1
dx=0.1
img=np.zeros((10,10))
for i in range(int(xmax/dx)):

    x=i*dx

    for j in range(int(f(x)/dx)):

        img[i,j]=1

 

print(img)
img1=np.flip(img,0)
#print(img2)

plt.imsave('filename.png', np.array(img1), cmap=cm.gray)

 

im = Image.open('filename.png')
im_flip = ImageOps.flip(im)
im_flip.save('filename1.png')

 

 
image1 = Image.open('filename.png')
image2 = Image.open('filename1.png')

 

(width1, height1) = image1.size
(width2, height2) = image2.size
result_width = max(width1 , width2)
result_height = height1 + height2
result = Image.new('RGB', (result_width, result_height))
result.paste(im=image1, box=(0, 0))
result.paste(im=image2, box=(0, height1))

result.save('test1.png')
