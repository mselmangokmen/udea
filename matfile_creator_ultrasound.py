import numpy as np 
from torch.utils.data import Dataset, DataLoader 
import os  
from PIL import Image, ImageOps 
from torchvision import transforms 
from sklearn.model_selection import train_test_split
from patchify import patchify
import matplotlib.pyplot as plt

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


u_path = "dataset/ultrasound/vid3"
u_patch_path = "dataset/ultrasound/vid3_patches"

dirs = os.listdir(u_path)
u_dirs = [d  for d in dirs if not d.startswith('._')  ]
u_list=[]
for u in u_dirs:
    u_list.append(os.path.join(u_path,u)) 
 
#vid_1_list = np.expand_dims(u_list,axis=1)  
print(u_list)
for i in range(len(u_list)):
    u_image = Image.open(u_list[i])  
    width, height = u_image.size   # Get dimensions 
    left = (width - 512)/2
    top = (height - 512)/2
    right = (width + 512)/2
    bottom = (height + 512)/2
    u_image = u_image.crop((left, top, right, bottom))  
    u_image = np.asarray(u_image)
    print(u_image.size)
    savePatches(u_image,u_patch_path,64,i)
#x_train_img = np.zeros(shape=(x_train.shape[0],64,64) )