import shutil

import cv2, os
import numpy as np
import random
from random import randint
in_img= "D:/Descargas/NBS2/train/U/79.jpg"
out_img= "D:/Descargas/79.jpg"
img = cv2.imread(in_img)
        #get size
height, width, channels = img.shape
print (in_img,height, width, channels)
        # Create a black image
x = height if height > width else width
y = height if height > width else width
square= np.zeros((x,y,3), np.uint8)
        #
        #This does the job
        #
square[(y-height)/2:y-(y-height)/2, (x-width)/2:x-(x-width)/2] = img
cv2.imwrite(out_img,square)
cv2.imshow("original", img)
cv2.imshow("black square", square)

