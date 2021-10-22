from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

import cv2
path1 = '/content/TestNoFlood/0.jpg'
image = cv2.imread(path1)
orig = image.copy()

image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model("fusion.h5")
# classify the input image
pred_test = model.predict(image)
y_classes = pred_test.argmax(axis=-1)
print(y_classes)
if y_classes==0:
   print('FLOOD')
else:
  print('NOFLOOD')
#pred_test = model.predict(trainX)
y_classes = pred_test.argmax(axis=-1)

import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

#Telabels = []
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images("Test/")))
random.seed(42)
random.shuffle(imagePaths)
Xtest = np.zeros((len(imagePaths),75,75,3))
# loop over the input images
aa=0
for imagePath in range(1,len(imagePaths)):
    image = cv2.imread("Test/"+str(imagePath)+".jpg")
    image = img_to_array(image)
    Xtest[aa,:,:,:]=image
    aa=aa+1

Xtest = np.array(Xtest, dtype="float") / 255.0

imgg=cv2.imread("/content/flood.png")
wholeImg1=cv2.resize(imgg,(500,500))
final_img=wholeImg1
pred_test = model.predict(Xtest)
y_classes = pred_test.argmax(axis=-1)
ii=0;
dd=0;
while ii<=wholeImg1.shape[0]-1:
    jj=0;
    while jj<=wholeImg1.shape[1]-1:
        imD1=cv2.imread('Test/'+str(dd)+'.jpg');
        if y_classes[dd]==0:
            final_img[ii:ii+20,jj:jj+20,0]=0;
            final_img[ii:ii+20,jj:jj+20,1]=0;
            final_img[ii:ii+20,jj:jj+20,2]=0;
        else:
            final_img[ii:ii+20,jj:jj+20,0]=255
            final_img[ii:ii+20,jj:jj+20,1]=255
            final_img[ii:ii+20,jj:jj+20,2]=255
            
        dd=dd+1;
        jj=jj+20;

    ii=ii+20;
cv2.imwrite('PostFloodWaterMap.jpg', final_img)