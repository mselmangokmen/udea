## Attention-guided U-Net Model with Improved Residual Blocks]{ Attention-guided U-Net Model with Improved Residual Blocks for Ultrasound Image Denoising

<p> This project is created for mid-term project of CS685 class. The project aims denoising ultrasound images. <br>

You can reach the related article by clicking this link. <br>

The proposed model Air U-Net is defined as model type 4 in mymodel file. In mymodel file 4 different models are employed and you can choose any model among Rat U-Net, Res U-Net, Attention U-Net and our proposed model Air U-Net for traning. <br>
The models were trained saved in the folders which named same as model name. Also, training results containg loss values in epochs and a plot are saved in the directories. <br>
During the training 4 different rayleigh noise applied on input images start from 0.1, 0.25, 0.5 and 0.75 respectively. Each model is trained for 100 epochs for each Rayleigh noise. For each rayleigh noise level, models are saved seperately in the directories such as AirUNet_ray_10, AirUNet_ray_25, AirUNet_ray_50, AirUNet_ray_75, that represent different Raylegh noise levels from from 0.1, 0.25, 0.5 and 0.75 respectively. <br>
```ruby
require 'redcarpet'
markdown = Redcarpet.new("Hello World!")
puts markdown.to_html
```
