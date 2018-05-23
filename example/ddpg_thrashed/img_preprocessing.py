import numpy as np
import cv2

def preprocess_img(img):
    assert len(img.shape) == 3
    x = np.mean(img/255.0,axis=2)
    small = cv2.resize(x,None,fx=0.25,fy=0.25)
    outshape = small.shape+(1,)
    return small.reshape(outshape)
