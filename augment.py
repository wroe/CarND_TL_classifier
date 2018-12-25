# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 10:42:08 2018

@author: Will
"""

import os
import cv2
import random
import numpy as np


def augment(img):
    h, w, nChannels = np.shape(img)

    # Change Brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rand = random.uniform(0.5, 1)
    hsv[:, :, 2] = rand*hsv[:, :, 2]
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Flip
    if random.choice([True, False]):
        img = cv2.flip( img, 1 )

    # Crop
    top = round(random.uniform(0, .05*h))
    bottom = round(random.uniform(.95*h, h))
    left = round(random.uniform(0, .1*w))
    right = round(random.uniform(.9*w, w))
    #print(str(top) +' '+ str(bottom) +' '+ str(left) +' '+ str(right))
    img = img[top:bottom, left:right]
    img = cv2.resize(img, (w, h))

    return img


#==================================================
if __name__ == '__main__':
# Test Augmentation
    cfg = 'sim_dataset'
    colors = os.listdir(cfg)

    for c in colors:
        dir_ = os.path.join(cfg, c)
        file = os.listdir(dir_)[0]
        path = os.path.join(dir_, file)
        img = cv2.imread(path)
        cv2.imshow('before',img)
        img2 = augment(img)
        cv2.imshow('after', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


