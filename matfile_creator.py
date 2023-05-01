import numpy as np 
from torch.utils.data import Dataset, DataLoader 
import os  
from PIL import Image, ImageOps 
from torchvision import transforms 
from sklearn.model_selection import train_test_split
from patchify import patchify

from scipy.io import savemat


def savePatches(image, path,patchSize, img_index, offset=0):
    img_index += offset
    patches = patchify(image,(patchSize,patchSize,3), step = patchSize)
    
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j, 0]
            patch = Image.fromarray(patch)
            num = i * patches.shape[1] + j
            img_name= 'img_'+ str(img_index) 
            img_name = img_name + '_patch_'+str(num)+'.jpg'
            patch.save(os.path.join(path,img_name))


x_path = "dataset/test/noisy"
y_path = "dataset/test/ground_truth"
dirs = os.listdir(x_path)

x_dirs = [d  for d in dirs if not d.startswith('._')  ]
x_list=[]
y_list=[]
for x in x_dirs:
    x_list.append(os.path.join(x_path,x))
    y_list.append(os.path.join(y_path,x))


x_test = np.expand_dims(x_list,axis=1)
y_test = np.expand_dims(y_list,axis=1)  

x_test_img = np.zeros(shape=(x_test.shape[0],64,64) )
y_test_img = np.zeros(shape=(y_test.shape[0],64,64))
for i in range(len(x_test)): 

    x_img = Image.open(x_test[i,0])
    y_img = Image.open(y_test[i,0])

    x_img= ImageOps.grayscale(x_img) 
    y_img= ImageOps.grayscale(y_img) 

    x_test_img[i,:,:]= x_img
    y_test_img[i,:,:]= y_img


import matplotlib.pyplot as plt
figure, axis = plt.subplots(1, 2 ) 

axis[ 0].imshow(x_test_img[133] , 'gray') 
axis[ 1].imshow(y_test_img[133] , 'gray')  
plt.show()

mymatfile = dict()
print('x_test shape:')
print(np.shape(x_test))
print('y_test shape:')
print(np.shape(y_test)) 

mymatfile['x_test']=x_test_img
mymatfile['y_test']=y_test_img 
savemat("dataset/mytestdataset.mat", mymatfile)


x_path = "dataset/train/noisy"
y_path = "dataset/train/ground_truth"
dirs = os.listdir(x_path)
x_dirs = [d  for d in dirs if not d.startswith('._')  ]
x_list=[]
y_list=[]
for x in x_dirs:
    x_list.append(os.path.join(x_path,x))
    y_list.append(os.path.join(y_path,x))
 
x_train = np.expand_dims(x_list,axis=1)
y_train = np.expand_dims(y_list,axis=1)  
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.25 )    


x_test_path = "dataset/test/noisy"
y_test_path = "dataset/test/ground_truth"
test_dirs = os.listdir(x_test_path)
x_test_dirs = [d  for d in test_dirs if not d.startswith('._')  ]
x_test_list=[]
y_test_list=[]

for x in x_test_dirs:
    x_test_list.append(os.path.join(x_test_path,x))
    y_test_list.append(os.path.join(y_test_path,x))

x_test = np.expand_dims(x_test_list,axis=1)
y_test = np.expand_dims(y_test_list,axis=1)

 
x_train_img = np.zeros(shape=(x_train.shape[0],64,64) )
y_train_img = np.zeros(shape=(x_train.shape[0],64,64))
x_val_img = np.zeros(shape=(x_val.shape[0],64,64))
y_val_img = np.zeros(shape=(x_val.shape[0],64,64))
x_test_img = np.zeros(shape=(x_test.shape[0],64,64))
y_test_img = np.zeros(shape=(x_test.shape[0],64,64))

for i in range(len(x_train)): 

    x_img = Image.open(x_train[i,0])
    y_img = Image.open(y_train[i,0])

    x_img= ImageOps.grayscale(x_img) 
    y_img= ImageOps.grayscale(y_img) 

    #x_img= np.expand_dims(np.array(x_img ),axis=2)
    #y_img= np.expand_dims(np.array(y_img),axis=2)

    x_train_img[i,:,:]= x_img
    y_train_img[i,:,:]= y_img

for i in range(len(x_val)): 
    x_img = Image.open(x_val[i,0])
    y_img = Image.open(y_val[i,0])

    x_img= ImageOps.grayscale(x_img) 
    y_img= ImageOps.grayscale(y_img) 

    #x_img= np.expand_dims(np.array(x_img ),axis=2)
    #y_img= np.expand_dims(np.array(y_img),axis=2)

    x_val_img[i,:,:]= x_img
    y_val_img[i,:,:]= y_img

for i in range(len(x_test)): 
    x_img = Image.open(x_test[i,0])
    y_img = Image.open(y_test[i,0])

    x_img= ImageOps.grayscale(x_img) 
    y_img= ImageOps.grayscale(y_img) 

    #x_img= np.expand_dims(np.array(x_img ),axis=2)
    #y_img= np.expand_dims(np.array(y_img ),axis=2)

    x_test_img[i,:,:]= x_img
    y_test_img[i,:,:]= y_img    
#for x in x_train: 

import matplotlib.pyplot as plt
figure, axis = plt.subplots(1, 2 ) 

axis[ 0].imshow(x_val_img[133] , 'gray') 
axis[ 1].imshow(y_val_img[133] , 'gray')  
plt.show()
mymatfile = dict()
print('x_train shape:')
print(np.shape(x_train))
print('y_train shape:')
print(np.shape(y_train)) 
print('x_val shape:')
print(np.shape(x_val))
print('y_val shape:')
print(np.shape(y_val)) 
print('x_test shape:')
print(np.shape(x_test))
print('y_test shape:')
print(np.shape(y_test)) 
mymatfile['x_train']=x_train_img
mymatfile['y_train']=y_train_img
mymatfile['x_val']=x_val_img
mymatfile['y_val']=y_val_img
mymatfile['x_test']=x_test_img
mymatfile['y_test']=y_test_img
savemat("dataset/mydataset.mat", mymatfile)