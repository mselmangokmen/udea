import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader 
import os  
from PIL import Image, ImageOps 
from torchvision import transforms 
from sklearn.model_selection import train_test_split 
 #https://paperswithcode.com/paper/restormer-efficient-transformer-for-high#code

import numpy as np
from scipy.io import loadmat

 
class myTestDataLoader():
    def __init__(self, batch_size) -> None:
        

        data_path= "dataset/mytestdataset.mat"
        mymatfile = loadmat(data_path)
        x_test = np.array(mymatfile["x_test"] )
        y_test = np.array(mymatfile["y_test"] )  

 
    
        test_set= myDataset(x= x_test, y= y_test  )   

        dataloaders = {
            'test': DataLoader(test_set, batch_size=batch_size, shuffle=True,  ) 
        }
        self.dataloaders = dataloaders 
        
    def getDataLoader(self): 
        return self.dataloaders
    
    
class myDataset(Dataset):
  def __init__(self, x, y):
    #x = [500,64,64]
    self.input_images = x # x_train
    self.target_images = y 
    #y = [500,64,64]
  def __len__(self):
    return len(self.input_images)

  def __getitem__(self, idx):
    #image = self.input_images[idx] image = [64,64]  
    image = self.input_images[idx]/255 
    target = self.target_images[idx]/255
  
    trans = transforms.Compose([
  transforms.ToTensor(), 
])
     
    image_t = trans(image)
    target_t = trans(target)  
    #[1,64,64]
    return  image_t, target_t 
  

class myTestDataLoader():
    def __init__(self, batch_size) -> None:
        

        data_path= "dataset/mytestdataset.mat"
        mymatfile = loadmat(data_path)
        x_test = np.array(mymatfile["x_test"] )
        y_test = np.array(mymatfile["y_test"] )  

 
    
        test_set= myDataset(x= x_test, y= y_test  )   

        dataloaders = {
            'test': DataLoader(test_set, batch_size=batch_size, shuffle=True,  ) 
        }
        self.dataloaders = dataloaders 
        
    def getDataLoader(self): 
        return self.dataloaders

class myDataLoader():
    def __init__(self, batch_size) -> None:
        

        data_path= "dataset/mydataset.mat"
        mymatfile = loadmat(data_path)
        x_train = np.array(mymatfile["x_train"] )
        y_train = np.array(mymatfile["y_train"] )

        x_val = np.array(mymatfile["x_val"] )
        y_val = np.array(mymatfile["y_val"] ) 
        #x_train = np.array(x_train).reshape(len(x_train),1, img_size, img_size) 
        #y_train = np.array(y_train).reshape(len(y_train), 1, img_size, img_size )

        #x_val = np.array(x_val).reshape(len(x_val),1,   img_size, img_size) 
        #y_val = np.array(y_val).reshape(len(y_val),  1, img_size, img_size )

 
    
        train_set= myDataset(x= x_train, y= y_train  )  
        val_set= myDataset(x= x_val, y= y_val   )  

        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True,  ), # [10,1, 64,64]
            'val': DataLoader(val_set, batch_size=batch_size, shuffle=True,  )# [10,1, 64,64]
        }
        self.dataloaders = dataloaders 
        
    def getDataLoader(self): 
        return self.dataloaders

dataLoader = myDataLoader(batch_size=10 )
train_loader = dataLoader.getDataLoader()['train'] 
i = iter(train_loader.__iter__())
x,y= i.__next__()
print(x.shape)
print(y.shape)
print(torch.max(x))
print(torch.max(y))