
#train_inits= [{'model_type':3,'model_name':"RatUNet"},{'model_type':1,'model_name':"ResUNet"},{'model_type':2,'model_name':"Attention_Unet"},]
#train_inits= [  {'model_type':2,'model_name':"Attention_Unet"},{'model_type':4,'model_name':"ours"} , ]
import gc
import torch
 
from dataset_loaders import  myTestDataLoader
from functions import test_model 
from model.mymodel import UNet
  
train_inits = [{'model_type':3,'model_name':"RatUNet"},{'model_type':1,'model_name':"ResUNet"},{'model_type':2,'model_name':"Attention_Unet"},{'model_type': 4, 'model_name': "AirUNet"}]

#train_inits= [ {'model_type':1,'model_name':"ResUNet"},{'model_type':2,'model_name':"Attention_Unet"},{'model_type':3,'model_name':"RatUNet"},{'model_type':4,'model_name':"Attention_Res_Unet"}]
batch_size = 10
device = torch.device('mps' )
start_factor = 32  
rayleigh_list = [0.1,0.25,0.5,0.75]

testDataLoader = myTestDataLoader(batch_size=10 )
test_loader = testDataLoader.getDataLoader()['test']  
for d in train_inits:
  for r in rayleigh_list:
    gc.collect()
    torch.cuda.empty_cache() 
    model = UNet(out_channel=1,modelType=d['model_type'],startFactor=start_factor).to(device=device)
    model_path = d['model_name'] + '/'+  d['model_name']+ '_ray_' +str(int(r*100)) 
    model.load_state_dict(torch.load(model_path))  
    test_model(model_name= d['model_name'],model= model,test_loader=test_loader,noise_level =r,device= device)