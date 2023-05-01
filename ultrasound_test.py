
  

import torch
from functions import  test_ultrasound_image



#train_inits = [{'model_type':3,'model_name':"RatUNet"},{'model_type':1,'model_name':"ResUNet"},{'model_type':2,'model_name':"Attention_Unet"},{'model_type': 4, 'model_name': "AirUNet"}]

train_inits = [ {'model_type':3,'model_name':"RatUNet"},{'model_type':2,'model_name':"Attention_Unet"} ,{'model_type':1,'model_name':"ResUNet"},{'model_type': 4, 'model_name': "AirUNet"}]

#train_inits= [ {'model_type':1,'model_name':"ResUNet"},{'model_type':2,'model_name':"Attention_Unet"},{'model_type':3,'model_name':"RatUNet"},{'model_type':4,'model_name':"Attention_Res_Unet"}]
device = torch.device('mps')
startFactor=32
rayleigh_list = [0.1,0.25 ,0.5,0.75] 

video_image_path = "dataset/ultrasound/vid2"  


image_index= 22
results =''
for d in train_inits:
  for r in rayleigh_list:    
        image_result_path = 'image_'+ str(image_index) 
        folder_path = 'ultrasound_results/' + image_result_path
        results += test_ultrasound_image(image_index=image_index,device=device,
         model_name=d['model_name'],model_type= d['model_type'],noise_level=r,startFactor=startFactor, video_image_path=video_image_path)
 
text_file = open(folder_path + 'test_results.txt', "w+")
text_file.write(results)
text_file.close()
print(results)