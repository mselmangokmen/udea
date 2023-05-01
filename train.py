#train_inits= [{'model_type':3,'model_name':"RatUNet"},{'model_type':1,'model_name':"ResUNet"},{'model_type':2,'model_name':"Attention_Unet"},]
#train_inits= [  {'model_type':2,'model_name':"Attention_Unet"},{'model_type':4,'model_name':"ours"} , ]
import gc
import torch

from torch import nn,optim
from dataset_loaders import myDataLoader
from functions import train_model  
from model.mymodel import UNet



#train_inits= [{'model_type':3,'model_name':"RatUNet"},{'model_type':1,'model_name':"ResUNet"},{'model_type':2,'model_name':"Attention_Unet"},]
#train_inits= [  {'model_type':2,'model_name':"Attention_Unet"},{'model_type':4,'model_name':"AirUNet"} , ]
train_inits = [{'model_type':3,'model_name':"RatUNet"},{'model_type':1,'model_name':"ResUNet"},{'model_type':2,'model_name':"Attention_Unet"},{'model_type': 4, 'model_name': "AirUNet"}]
batch_size = 50
num_epochs = 100
device = torch.device('mps' )
start_factor = 32
lr = 0.0005

#airUnet 0.892847 start_factor = 32 batch_size = 50 num_epochs = 100
#ResUNet 0.901837 start_factor = 32 batch_size = 50 num_epochs = 100 
#Attention_Unet
rayleigh_list = [ 0.1,0.25,0.5,0.75 ]
#ResUNet 0.901583 start_factor = 32 batch_size = 50 num_epochs = 100 
# epoch 30 :  0.890529 normalizasyon yok 
# epoch 30 :  0.87 batch normalizasyon var sadece en altta
# epoch 30 :  0.--  instance normalizasyon var sadece iki tarafta
# 3 0.852 
# 8 score : 0.889
for d in train_inits:
    for r in rayleigh_list:
        gc.collect()
        torch.cuda.empty_cache()
        model = UNet(out_channel=1, modelType=d['model_type'], startFactor=start_factor).to(device=device)
        #optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
        #                              lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False, maximize=False, foreach=None, capturable=False)
        #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0002)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr , weight_decay=0.0003)
        dataloaders = myDataLoader(batch_size=batch_size).getDataLoader()
        train_model(dataloaders=dataloaders, model_name=d['model_name'], noise_level=r, model=model,
                    optimizer=optimizer, device=device, num_epochs=num_epochs)