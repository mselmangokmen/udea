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

 
class myDataset(Dataset):
  def __init__(self, x, y  ):
    self.input_images = x
    self.target_images = y 

  def __len__(self):
    return len(self.input_images)

  def __getitem__(self, idx):
    image = self.input_images[idx]/255
    target = self.target_images[idx]/255
  
    trans = transforms.Compose([
  transforms.ToTensor(), 
])
     
    image_t = trans(image)
    target_t = trans(target)  
    return  image_t, target_t 
  

class myDataLoader():
    def __init__(self, batch_size,img_size) -> None:
        

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
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True,  ),
            'val': DataLoader(val_set, batch_size=batch_size, shuffle=True,  )
        }
        self.dataloaders = dataloaders 
        
    def getDataLoader(self): 
        return self.dataloaders

dataLoader = myDataLoader(batch_size=10,img_size=64)
train_loader = dataLoader.getDataLoader()['train'] 
i = iter(train_loader.__iter__())
x,y= i.__next__()
print(x.shape)
print(y.shape)

#data_path= "dataset/mydataset.mat"
#mymatfile = loadmat(data_path)
#x_train = mymatfile["x_train"]
#y_train = mymatfile["y_train"]

#x_val = mymatfile["x_val"]
#y_val = mymatfile["y_val"]

#x_test = mymatfile["x_test"]
#y_test = mymatfile["y_test"]

#print(mymatfile.keys())

#print('x_train shape:')
#print(np.shape(x_train))
#print('y_train shape:')
#print(np.shape(y_train)) 
#print('x_val shape:')
#print(np.shape(x_val))
#print('y_val shape:')
#print(np.shape(y_val)) 
#print('x_test shape:')
#print(np.shape(x_test))
#print('y_test shape:')
#print(np.shape(y_test)) 
#mymatfile = dict()
#mymatfile['x_train']=x_train_img
#mymatfile['y_train']=y_train_img
#mymatfile['x_val']=x_val_img
#mymatfile['y_val']=y_val_img
#mymatfile['x_test']=x_test_img
#mymatfile['y_test']=y_test_img
#savemat("dataset/mydataset.mat", mymatfile)

#path = "dataset/CroppedImages"
#path2 = "dataset"
#dirs = os.listdir(path)
#dir_list = [d  for d in dirs if not d.startswith('._')  ] 
#x_list = [x  for x in dir_list if 'real' in x.lower()]   

#y_list = [x.replace('real','mean')  for x in x_list  ] 

#x_train, x_test, y_train, y_test = train_test_split(x_list,y_list,test_size=0.15 )

#print(np.shape(x_test))
#for t in x_train:  
#for i in range(len(x_test)): 
#    x_image = Image.open(os.path.join(path,x_test[i]))
#    x_image = np.asarray(x_image)
#    savePatches(x_image,'dataset/test/noisy',64,i,offset=97)

#   y_image = Image.open(os.path.join(path,y_test[i]))
#   y_image = np.asarray(y_image)
#   savePatches(y_image,'dataset/test/ground_truth',64,i,offset=97) 

#print(dir_list2)
#x_list = [x  for x in dir_list if 'real' in x.lower()]   
#y_list = [x.replace('real','mean')  for x in x_list  ] 
#x_train = np.expand_dims(x_list,axis=1)
#y_train = np.expand_dims(y_list,axis=1)
 
#x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.15 )   
#x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=0.15 )   
#print(x_train.shape)
#print(y_train.shape)
#print(x_val.shape)
#print(y_val.shape)
#print(x_test.shape)
#print(y_test.shape)
 
#x_img = cv2.imread(os.path.join(path,x_train[0,0]),0) 
#print(type(x_img))
#print(np.expand_dims(x_img,axis=0).shape)
#print(np.amax(x_img))
#print(y_list)
# using loadtxt()
#test_set = pd.read_csv("mitbih_test.csv")

#train_test = pd.read_csv("mitbih_train.csv" )
#label_names = ['N','S','F','V','Q']

#x_test = test_set.iloc[:,:-1].values
#y_test = test_set.iloc[:,-1:].astype(dtype=int).astype(dtype=str).values

#print((y_test[-10:,:]))
#y_test_one = np.zeros(shape= (y_test.shape[0], len(label_names)), dtype=float)
#for i in range(len(label_names)):  
#    y_test[y_test == str(i)] = label_names[i]
    #indexes = np.where((y_test  == i).all(axis=1))  
    #y_test_one[indexes,i]  = 1
#y_test = y_test.reshape(y_test.shape[0], 1 ) 
#print(y_test.shape)
#print(x_test.shape)
#test_data = {'ecg': x_test, 'labels': y_test}
#data = pd.DataFrame(test_data) 
#data.head() 