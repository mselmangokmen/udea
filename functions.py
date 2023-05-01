
import os
import torch
 
import gc 
from PIL import Image 
from skimage import io
import copy 
import numpy as np  
from piq import   SSIMLoss,   ssim 
import ssl
import matplotlib.pyplot as plt

from model.mymodel import UNet  
ssl._create_default_https_context = ssl._create_unverified_context  
from patchify import patchify,unpatchify

def getSSIM(output,target,device):
  output = np.squeeze(output)
  output = np.expand_dims(output,axis=0)
  output = np.expand_dims(output,axis=0) 

  target = np.expand_dims(target,axis=0)
  target = np.expand_dims(target,axis=0) 

  target_tensor = torch.from_numpy(target).float() 
  output_tensor = torch.from_numpy(output).float() 
  
  target_tensor = target_tensor.to(device=device, dtype=torch.float)   
  output_tensor = output_tensor.to(device=device, dtype=torch.float)  
 
  ssim_score = ssim(output_tensor,target_tensor, data_range=1.)
  return ssim_score.item()


def add_rayleigh_noise_on_ultrasound(image, scale): 
    # Resim boyutlarını al
    row, col = image.shape

    # Rastgele gürültü oluştur
    rayleigh_noise = np.random.rayleigh(scale, (row, col ))
 
    # Resim üzerine gürültüyü ekle
    noisy_image = np.clip(image + rayleigh_noise, 0, 1).astype(np.float64)

    return noisy_image

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]



def model_estimate(model,noisy_image,device):
  patchSize=64
  image= np.expand_dims(noisy_image, axis=-1) 
  patches = patchify(image,(patchSize,patchSize,1 ), step = patchSize) 
  patch_list = list() 
  for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
      patch = patches[i, j, 0]   
      patch= np.squeeze(patch )
      patch = np.expand_dims(patch,axis=0)
      patch = np.expand_dims(patch,axis=0) 
      noise_tensor = torch.from_numpy(patch).float() 
      noise_tensor = noise_tensor.to(device=device, dtype=torch.float)    
      output = model(noise_tensor) 
      output_patch=  output.cpu().detach().numpy() 
      output_patch= np.squeeze(output_patch )
      output_patch= np.squeeze(output_patch ) 
      output_patch= np.expand_dims(output_patch,axis=-1)  
      patches[i, j, 0]  =output_patch 
      
  output_image = unpatchify(patches, (512,512,1))
  return output_image
def add_rayleigh_noise_torch(img_tensor, scale):

    batch_size, channels, height, width = img_tensor.shape
    noise = torch.from_numpy(np.random.rayleigh(scale, size=(batch_size, channels, height, width))).float()
    noisy_img_tensor = img_tensor + noise 
    noisy_img_tensor = torch.clamp(noisy_img_tensor, 0.0, 1.0)
    return noisy_img_tensor

def add_rayleigh_noise(image, scale): 
    # Resim boyutlarını al
    row, col = image.shape

    # Rastgele gürültü oluştur
    rayleigh_noise = np.random.rayleigh(scale, (row, col ))
 
    # Resim üzerine gürültüyü ekle
    noisy_image = np.clip(image + rayleigh_noise, 0, 1).astype(np.float64)

    return noisy_image
def lossFun(pred, target,device):
    #softmax = nn.Softmax(dim=1)
    #mse_criterion = nn.L1Loss()  

    #mse_criterion = nn.MSELoss()   
    ssim_criterion = SSIMLoss(data_range=1.)  
    ssim_score = ssim(pred,target, data_range=1.)
    lossClass= ssim_criterion(pred, target)  
    #print(lossClass.item())
    return lossClass, ssim_score


def calc_loss_and_score(pred, target,device, metrics): 
 
    
    ssim_loss,ssim_score = lossFun(pred, target,device) 
    metrics['loss'].append( ssim_loss.item() )  
    metrics['score'].append( ssim_score.item() )  
    return ssim_loss


def calc_loss_and_score_test(pred, target,device, metrics): 
 
    
    ssim_loss,ssim_score = lossFun(pred, target,device) 
    metrics['loss'] = ssim_loss.item() 
    metrics['score'] =  ssim_score.item() 
    metrics['loss_list'].append( ssim_loss.item() )  
    metrics['score_list'].append( ssim_score.item() )  
    return ssim_loss.item()
 
 
def print_metrics(main_metrics_train,model_name,main_metrics_val,metrics, phase,epoch, num_epochs,rayleigh):
    
    loss= metrics['loss'] 
    score= metrics['score'] 
    if(phase == 'train'):
        main_metrics_train['loss'].append( np.mean(loss)) 
        main_metrics_train['score'].append( np.mean(score)) 
        #main_metrics_train['accuracy'].append( accuracy ) 
    else:
        main_metrics_val['loss'].append(np.mean(loss)) 
        main_metrics_val['score'].append(np.mean(score)) 
    result = 'Epoch {}/{}'.format(epoch, num_epochs - 1)
    
    result += ' Raylegih level: '+ str(rayleigh) + ' model name : '+ str(model_name) +' \n'
    result += '-' * 10
    result += '\n'
    result += "phase: "+str(phase)  +  ' \nloss : {:4f}'.format(np.mean(loss))      +"\n"  +  ' \nscore : {:4f}'.format(np.mean(score))      +"\n"
    return result 

def print_test_metrics(test_metrics,  rayleigh,batch_num, model_name ):
    
    loss= test_metrics['loss'] 
    score= test_metrics['score']    
    result="" 
    result += '\n Best Batch Results : ' + str(batch_num) +' \n'
    result += ' Model Name : '+ model_name + ' \n'
    result += ' Raylegih level: '+ str(rayleigh) + ' \n' 
    result +=  ' \nLoss : {:4f}'.format(loss)      +"\n"  +  ' \nscore : {:4f}'.format(score)      +"\n"
    return result 

def print_avg_test_metrics(test_metrics,  rayleigh, model_name ): 
    loss= test_metrics['loss_list'] 
    score= test_metrics['score_list']    
    
    result="" 
    result += '=' * 10 
    result += '\n Average Results :   \n'
    result += '\n Model Name : '+ model_name + ' \n' 
    result += ' Raylegih level: '+ str(rayleigh) + ' \n'
    result +=  ' \nLoss : {:4f}'.format(np.mean(loss))      +"\n"  +  ' \nscore : {:4f}'.format(np.mean(score))      +"\n"
  
    return result 

def print_save_figure(train_dict,val_dict,num_epochs,fname): 
  fig, ax = plt.subplots( nrows=1, ncols=1 ) 
  ax.plot(range(1,num_epochs+1), train_dict['loss'], label='Training Loss')
  ax.legend(loc="upper right")
  ax.plot(range(1,num_epochs+1), val_dict['loss'], label='Validation Loss')
  ax.legend(loc="upper right")
  fig.savefig(fname+'_loss.png') 
  plt.close(fig) 
  fig, ax = plt.subplots( nrows=1, ncols=1 ) 
  ax.plot(range(1,num_epochs+1), train_dict['score'], label='Training Score')
  ax.plot(range(1,num_epochs+1), val_dict['score'], label='Validation Score')
  fig.savefig(fname+'_score.png') 
  plt.close(fig)

def print_save_test_figure(test_dict,batch_num,fname): 
  fig, ax = plt.subplots( nrows=1, ncols=1 ) 
  ax.plot(range(1,batch_num+1), test_dict['loss_list'], label='Test Loss') 
  fig.savefig(fname+'_loss.png') 
  plt.close(fig) 
  fig, ax = plt.subplots( nrows=1, ncols=1 ) 
  ax.plot(range(1,batch_num+1), test_dict['score_list'], label='Test Score') 
  fig.savefig(fname+'_score.png') 
  plt.close(fig)

def train_model(dataloaders,model_name,model,optimizer,noise_level,device, num_epochs=100): 
 
      train_dict= dict()
      train_dict['loss']= list()  
      train_dict['score']= list() 
      val_dict= dict()
      val_dict['loss']= list()  
      val_dict['score']= list()  


      model_path = model_name

      isExist = os.path.exists(model_path)
      if not isExist:
 
        os.makedirs(model_path)
      model_name_new = model_name +'_ray_' +str(int(noise_level*100)) 
      best_model_wts = copy.deepcopy(model.state_dict())
      best_loss = 99999
      best_score = -99999 
      train_string = "" 
      temp_str=""
      

      for epoch in range(num_epochs):
          #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
          #print('-' * 10) 

          for phase in ['train', 'val']: # phase = train
              if phase == 'train':
                  for param_group in optimizer.param_groups:
                      print("LR", param_group['lr'])

                  model.train()  # Set model to training mode
              else:
                  model.eval()   # Set model to evaluate mode

              metrics = dict()
              metrics['loss']=list()
              metrics['score']=list()
              #metrics['correct']=0
              #metrics['total']=0
            
              for inputs, labels in dataloaders[phase]: 
                  # inputs = [75, 1, 64,64]
                  # labels = [75, 1, 64,64]  GT image
                  inputs = add_rayleigh_noise_torch(inputs,noise_level) 
                  # noisy_images = [75, 1, 64,64]
                  inputs = inputs.to(device=device, dtype=torch.float) 
                  labels = labels.to(device=device, dtype=torch.float)# expected output
                  
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
              epoch_result= print_metrics(main_metrics_train=train_dict, rayleigh=noise_level, model_name=model_name,
                main_metrics_val=val_dict,metrics=metrics,phase=phase,epoch= epoch, num_epochs=num_epochs )
              train_string +=epoch_result
              print(epoch_result)
              epoch_loss = np.mean(metrics['loss'])
              epoch_score= np.mean(metrics['score'])

              if phase == 'val' and epoch_score > best_score:
                      #print("saving best model")
                      best_score = epoch_score  
                      best_model_wts = copy.deepcopy(model.state_dict())
                      temp_str= '\n'+ ('-'*10) + '\n' + 'best score : '+ str(best_score)+'\n' \
                      + 'epoch num : '+ str(epoch)+'\n'+ ('-'*10) + '\n' 
      train_string+=temp_str
      text_file = open(model_name +'/'+ model_name_new +'.txt', "w+")
      text_file.write(train_string)
      text_file.close()
      print('Best val loss: {:4f}'.format(best_score))

      torch.save(best_model_wts, model_name +'/'+ model_name_new)
      print_save_figure(train_dict,val_dict,num_epochs,model_name +'/'+ model_name_new)


def test_model(model_name,model,test_loader,noise_level,device): 
 
  isExist = os.path.exists('test_results')
  if not isExist:
 
        os.makedirs('test_results')
  model.eval()  
  test_dict= dict()
  test_dict['loss_list']= list()  
  test_dict['score_list']= list()   
  test_dict['loss']= 0.0
  test_dict['score']= 0.0
  model_name_new = model_name +'_avg_ray_' +str(int(noise_level*100)) 
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 99999 
  best_batch_result=""  
  batch_result= ''
  batch_cnt=0
  for inputs, labels in test_loader:  
    with torch.no_grad():
      inputs = add_rayleigh_noise_torch(inputs,noise_level) 
      inputs = inputs.to(device=device, dtype=torch.float) # 75,1,64,64
      labels = labels.to(device=device, dtype=torch.float)
      outputs = model(inputs) 
      
      loss = calc_loss_and_score_test(pred= outputs,target= labels,device=device, metrics= test_dict)     
      
      batch_result= print_test_metrics(test_metrics = test_dict,  rayleigh= noise_level,batch_num= batch_cnt,model_name= model_name )

      batch_cnt +=1 
      #print(batch_result)
      if best_loss > loss:
        best_loss = loss   
        best_batch_result= batch_result  

  total_result= print_avg_test_metrics(test_metrics = test_dict,  rayleigh= noise_level,model_name= model_name  )
  total_result += "\n "+ best_batch_result
  text_file = open('test_results/'+model_name_new +'.txt', "w+")
  text_file.write(total_result)
  text_file.close()
  print(total_result) 
  #print_save_test_figure(test_dict,batch_cnt,model_name_new) 



def test_ultrasound_image(startFactor, video_image_path ,model_name,model_type,image_index,noise_level,device):
    image_result_path = 'image_'+ str(image_index) 
    folder_path = 'ultrasound_results/' + image_result_path + '/'+  model_name
    model_path = model_name + '/'+  model_name+ '_ray_' +str(int(noise_level*100)) 

    clean_image_path = folder_path + '/' + 'clean_image_' + str(image_index) +'.png'
    noisy_image_path = folder_path + '/' + 'noisy_image_' + 'ray_' +str(int(noise_level*100)) +'_image_'+ str(image_index) +'.png'
    predicted_image_path = folder_path+   '/'+  model_name +'_image_'+ str(image_index) + '_ray_' +str(int(noise_level*100)) +'.png'
    #bm3d_image_path = folder_path + '/' + 'bm3d_image_' + str(image_index) +'.png'
    isExist = os.path.exists(folder_path)
    if not isExist: 
        os.makedirs(folder_path)

    
    files = os.listdir(video_image_path) 

    file_dirs = [os.path.join(video_image_path,f) for f in files if  not  f.startswith('._')] 
    image = io.imread(file_dirs[image_index], as_gray=True)   
    image = image.astype(np.float64)  
    image = crop_center(image,512,512) 

    results =''
    results +='='*20
    gc.collect()
    torch.cuda.empty_cache() 
    model = UNet(out_channel=1,modelType=model_type,startFactor=startFactor).to(device=device)
    model.load_state_dict(torch.load(model_path))  
    results +='\nModel Name : '+ model_name+ '\n'
    results +='\nRayleigh Level : '+ str(noise_level) + '\n'
    model.eval()

    with torch.no_grad():
      noisy_image = add_rayleigh_noise_on_ultrasound(image,noise_level) 
      noisy_image_model = np.copy(noisy_image)
      #noisy_image_bm3d = np.copy(noisy_image)
      clean_image_model = model_estimate(model,noisy_image_model ,device)
      #clean_image_bm3d = BM3D_denoise( noisy_image_bm3d,noise_level)  

      ssim_score_model = getSSIM(clean_image_model,image,device= device )
      #ssim_score_bm3d = getSSIM(clean_image_bm3d,image,device= device  ) 
      #results +='\n ssim with bm3d : '+ str(ssim_score_bm3d) + '\n'
      results +='\n ssim with model : '+ str(ssim_score_model) + '\n'
      
      clean_image = Image.fromarray(np.uint8(255 * image))  
      clean_image.save(clean_image_path)
      noisy_image = Image.fromarray(np.uint8(255 * noisy_image))   
      noisy_image.save(noisy_image_path)

      clean_image_model= np.squeeze(clean_image_model)
      clean_image_model = Image.fromarray(np.uint8(255 * clean_image_model))  
      clean_image_model.save(predicted_image_path)

      #clean_image_bm3d= np.squeeze(clean_image_bm3d)
      
      #clean_image_bm3d = Image.fromarray(np.uint8(255 * clean_image_bm3d))  
      #clean_image_bm3d.save(bm3d_image_path)

      results +='='*20
      results +='\n' 
    return results


