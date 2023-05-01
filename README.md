
## Attention-guided U-Net Model with Improved Residual Blocks

<p> This project is created for mid-term project of CS685 class. The project aims denoising ultrasound images. <br>

You can reach the related article by clicking this [link](https://github.com/mselmangokmen/udea/blob/main/air_unet.pdf). <br>
If you want to run a denoising process on ultrasound images directly, please download the full trained models and datasets from the link provided below. <br> https://drive.google.com/file/d/1gb-WAMf2atUzktwq-wZ4DJv-W4z-yD09/view?usp=sharing <br>
Don't forget to run 'pip install -r requirements.txt' for installing all required libraries. <br>


The proposed model Air U-Net is defined as model type 4 in mymodel file. In mymodel file 4 different models are employed and you can choose any model among Rat U-Net, Res U-Net, Attention U-Net and our proposed model Air U-Net for traning. <br>
The models were trained saved in the folders which named same as model name. Also, training results containg loss values in epochs and a plot are saved in the directories. <br>
During the training 4 different rayleigh noise applied on input images start from 0.1, 0.25, 0.5 and 0.75 respectively. Each model is trained for 100 epochs for each Rayleigh noise. For each rayleigh noise level, models are saved seperately in the directories such as AirUNet_ray_10, AirUNet_ray_25, AirUNet_ray_50, AirUNet_ray_75, that represent different Raylegh noise levels from from 0.1, 0.25, 0.5 and 0.75 respectively. <br>
 ## #Training

 The training code is provided below. You can select and choose any model with chancing the 'train_inits'. Also, you can train any with a specific Rayleigh noise by modifying 'rayleigh_list'. Traning code below will traing each model you added in train_inits with specified Rayleigh noise levels and save in different folders with the same model name such as 'RatUNet', 'ResUNet', 'Attention_Unet', 'Air_Unet'. 
```python
import gc
import torch
 
from dataset_loaders import myDataLoader
from functions import train_model  
from model.mymodel import UNet
 
train_inits = [{'model_type':3,'model_name':"RatUNet"},{'model_type':1,'model_name':"ResUNet"},{'model_type':2,'model_name':"Attention_Unet"},{'model_type': 4, 'model_name': "AirUNet"}]
batch_size = 50
num_epochs = 100
device = torch.device('mps' )
start_factor = 32
lr = 0.0005
 
rayleigh_list = [ 0.1,0.25,0.5,0.75 ] 
for d in train_inits:
    for r in rayleigh_list:
        gc.collect()
        torch.cuda.empty_cache()
        model = UNet(out_channel=1, modelType=d['model_type'], startFactor=start_factor).to(device=device) 
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr , weight_decay=0.0003)
        dataloaders = myDataLoader(batch_size=batch_size).getDataLoader()
        train_model(dataloaders=dataloaders, model_name=d['model_name'], noise_level=r, model=model,
                    optimizer=optimizer, device=device, num_epochs=num_epochs)
```

The DND dataset used for training [^1]. The DND dataset contains real-life noisy images.  <br>

 ## #Test
A training file has 'mat' extension created and saved in dataset directory with the name of 'mydataset.mat' and also test dataset file has been saved separately with the name of 'mydatasettest.mat'. <br>
If you run the testing code provided below, a new directory will be created and each test results will be saved in it with different names regarding Rayleigh noise level and model name such as 'AirUNet_avg_ray_25.txt' . 

```python 
import  gc 
import  torch 
from  dataset_loaders  import  myTestDataLoader 
from  functions  import  test_model 
from  model.mymodel  import  UNet 
test_inits = [{'model_type':3,'model_name':"RatUNet"},{'model_type':1,'model_name':"ResUNet"},{'model_type':2,'model_name':"Attention_Unet"},{'model_type': 4, 'model_name': "AirUNet"}]
 
batch_size = 10 
device = torch.device('mps' ) 
start_factor = 32

rayleigh_list = [0.1,0.25,0.5,0.75] 
testDataLoader = myTestDataLoader(batch_size=10 ) 
test_loader = testDataLoader.getDataLoader()['test'] 
for  d  in  test_inits: 
for  r  in  rayleigh_list: 
gc.collect() 
torch.cuda.empty_cache() 
model = UNet(out_channel=1,modelType=d['model_type'],startFactor=start_factor).to(device=device) 
model_path = d['model_name'] + '/'+ d['model_name']+ '_ray_' +str(int(r*100)) 
model.load_state_dict(torch.load(model_path)) 
test_model(model_name= d['model_name'],model= model,test_loader=test_loader,noise_level =r,device= device) 
```

 ## #Ultrasound Denoising
Testing the models on ultrasound images, you can run 'ultrasound_test.py' file and please sure that you downloaded all dataset from google drive link above. There are two different dataset for testing the models on real ultrasound images. The ultrasound images provided in dataset folder have video frames from US-4 dataset [^2] have real life ultrasound images of videos. There are two different video frames in separate directories and you can choose any video frame by editing image_index and video_image_path variables. 

```python  
import  torch 
from  functions  import  test_ultrasound_image
 
test_inits = [ {'model_type':3,'model_name':"RatUNet"},{'model_type':2,'model_name':"Attention_Unet"} ,{'model_type':1,'model_name':"ResUNet"},{'model_type': 4, 'model_name': "AirUNet"}] 
device = torch.device('mps') 
startFactor=32 
rayleigh_list = [0.1,0.25 ,0.5,0.75] 
video_image_path = "dataset/ultrasound/vid2" 
image_index= 22 
results ='' 
for  d  in  test_inits: 
for  r  in  rayleigh_list: 
image_result_path = 'image_'+ str(image_index) 
folder_path = 'ultrasound_results/' + image_result_path 
results += test_ultrasound_image(image_index=image_index,device=device, 
model_name=d['model_name'],model_type= d['model_type'],noise_level=r,startFactor=startFactor, video_image_path=video_image_path) 
text_file = open(folder_path + 'test_results.txt', "w+") 
text_file.write(results) 
text_file.close() 
print(results)
```
The denoised images will be saved in in ultrasound_results directory with image index you chose in the code above. Please be sure that you downloaded pretrained models and dataset files. 

[^1]: Real-world Noisy Image Denoising: A New Benchmark
 [^2]: Pretraining deep ultrasound image diagnosis model through video contrastive representation learning
