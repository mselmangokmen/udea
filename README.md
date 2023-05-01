## Attention-guided U-Net Model with Improved Residual Blocks

<p> This project is created for mid-term project of CS685 class. The project aims denoising ultrasound images. <br>

You can reach the related article by clicking this link. <br>

The proposed model Air U-Net is defined as model type 4 in mymodel file. In mymodel file 4 different models are employed and you can choose any model among Rat U-Net, Res U-Net, Attention U-Net and our proposed model Air U-Net for traning. <br>
The models were trained saved in the folders which named same as model name. Also, training results containg loss values in epochs and a plot are saved in the directories. <br>
During the training 4 different rayleigh noise applied on input images start from 0.1, 0.25, 0.5 and 0.75 respectively. Each model is trained for 100 epochs for each Rayleigh noise. For each rayleigh noise level, models are saved seperately in the directories such as AirUNet_ray_10, AirUNet_ray_25, AirUNet_ray_50, AirUNet_ray_75, that represent different Raylegh noise levels from from 0.1, 0.25, 0.5 and 0.75 respectively. <br>
 
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

The DND dataset used for training [^1]. The DND dataset contains real-life noisy images.  


[^1]: Real-world Noisy Image Denoising: A New Benchmark
 
