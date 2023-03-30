
import torch

import copy
from torch import nn,optim 
from torchinfo import summary  
import numpy as np
from dataset_loaders import myDataLoader 
from model.mymodel import UNet 
from piq import   SSIMLoss, MultiScaleSSIMLoss, FID,FSIMLoss, DISTS, ContentLoss,LPIPS
import lpips
import ssl
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
ssl._create_default_https_context = ssl._create_unverified_context
def lossFun(pred, target,device): 
    mse_criterion = nn.MSELoss().to(device=device)  
    #pred= softmax(pred)
    #ssim_criterion = SSIMLoss().to(device=device)
    #dists_criterion = DISTS().to(device=device)
    #content_criterion = ContentLoss().to(device=device)
    #LPIPS_criterion = lpips.LPIPS(net='alex',).to(device=device)
    #LPIPS_criterion = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    #pred = torch.sigmoid(pred)
    #pred = softmax(pred ) 
    #ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=device)
    #print(  torch.max(pred) )
    #print(  torch.max(target) )
    #lossClass= ssim(pred, target ) 
    lossClass= mse_criterion(pred, target)
    #print( 'min or max pred: ' + str(torch.min(pred).item()))
    #print( 'min or max target: ' + str(torch.min(target).item())) 
 
    #print('pred size: '+ str(pred.shape ) + ' target size: '+ str(target.shape) + ' lossClass: '+ str(lossClass.item()) )
    return lossClass


def calc_loss_and_score(pred, target,device, metrics): 

    #pred =  pred.squeeze( -1)
    #target= target.squeeze( -1)
    
    ssim_loss = lossFun(pred, target,device) 
    metrics['loss'].append( ssim_loss.item() )
    #softmax = nn.Softmax(dim=1)
    #pred = softmax(pred ) 
    #_,pred = torch.max(pred, dim=1) 
    #metrics['correct']  += torch.sum(pred ==target ).item() 
    #metrics['total']  += target.size(0)  

    return ssim_loss
 
 
def print_metrics(main_metrics_train,main_metrics_val,metrics, phase):
   
    #correct= metrics['correct']  
    #total= metrics['total']  
    #accuracy = 100*correct / total
    loss= metrics['loss'] 
    if(phase == 'train'):
        main_metrics_train['loss'].append( np.mean(loss)) 
        #main_metrics_train['accuracy'].append( accuracy ) 
    else:
        main_metrics_val['loss'].append(np.mean(loss)) 
        #main_metrics_val['accuracy'].append(accuracy ) 
    
    #result = "phase: "+str(phase) \
    #+  ' \nloss : {:4f}'.format(np.mean(loss))   +    ' accuracy : {:4f}'.format(accuracy)        +"\n"
    result = "phase: "+str(phase) \
    +  ' \nloss : {:4f}'.format(np.mean(loss))      +"\n"
    return result 


def train_model(dataloaders,model,optimizer,device, num_epochs=100): 
 
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_dict= dict()
    train_dict['loss']= list()
    #train_dict['accuracy']= list() 
    val_dict= dict()
    val_dict['loss']= list()
    #val_dict['accuracy']= list() 

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10) 

        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = dict()
            metrics['loss']=list()
            #metrics['correct']=0
            #metrics['total']=0
 
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.float)
                
                # zero the parameter gradients
                optimizer.zero_grad()


                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) 
                    #print('outputs size: '+ str(outputs.size()) )
                    loss = calc_loss_and_score(pred= outputs,target= labels,device=device, metrics= metrics)   
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                #print('epoch samples: '+ str(epoch_samples)) 
            print(print_metrics(main_metrics_train=train_dict, main_metrics_val=val_dict,metrics=metrics,phase=phase ))
            epoch_loss = np.mean(metrics['loss'])
        
            if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss 

    print('Best val loss: {:4f}'.format(best_loss))


device = torch.device("mps") 
batch_size = 100
model = UNet(out_channel=1,modelType=3).to(device=device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
dataloaders= myDataLoader(batch_size=batch_size, img_size=64).getDataLoader()
model_normal_ce = train_model(dataloaders=dataloaders,model=model,optimizer=optimizer, device=device,num_epochs=100)
torch.save(model.state_dict(), 'myModel')