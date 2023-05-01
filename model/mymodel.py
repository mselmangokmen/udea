

import torch 
from torch import nn
from model.blocks.attention_block import AttentionBlock
from model.blocks.channel_self_attention import ChannelSelfAttention
from model.blocks.denoising_attention_block import DenoisingAttentionBlock
from model.blocks.double_conv import DoubleConv
from model.blocks.improved_down_sample import ImprovedDownSample
from model.blocks.improved_residual_block import ImprovedResidualBlock
from model.blocks.residual_double_block import ResidualDoubleConv
from model.blocks.spatial_self_attention import SpatialSelfAttention

#https://github.com/JavierGurrola/RDUNet
#

class UNet(nn.Module):
# model 1: RESUNet
# model 2: Attention UNet (normal blocks)
# model 3: RatUNet (res blocks)
# model 4: Attention UNet with residual blocks
# model 5: RatUNet with normal blocks


    def __init__(self, out_channel=1,modelType=2,startFactor=16):

        super().__init__() 
        self.modelType= modelType
        self.channelAttention = ChannelSelfAttention(in_channels=startFactor*2 + startFactor)  
        self.spatialAttention = SpatialSelfAttention(in_channels=startFactor*2 + startFactor)   
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, 
                                    mode='bilinear', align_corners=True)    
        self.sigmoid = nn.Sigmoid()
        if modelType==1:# [10,1, 64,64]
          self.dconv_down1 = ResidualDoubleConv(1, startFactor) # # [10,64, 64,64]
          self.dconv_down2 = ResidualDoubleConv(startFactor, startFactor*2) # [10,128, 64,64]
          self.dconv_down3 = ResidualDoubleConv(startFactor*2, startFactor*4)
          self.dconv_down4 = ResidualDoubleConv(startFactor*4, startFactor*8)
          self.dconv_down5 = ResidualDoubleConv(startFactor*8, startFactor*16)  

          self.dconv_up4 = ResidualDoubleConv(startFactor*16 + startFactor*8, startFactor*8 )
          self.dconv_up3 = ResidualDoubleConv(startFactor*8 + startFactor*4, startFactor*4)
          self.dconv_up2 = ResidualDoubleConv(startFactor*4 + startFactor*2, startFactor*2)
          self.dconv_up1 = ResidualDoubleConv(startFactor*2 + startFactor, startFactor)
          self.conv_last = nn.Conv2d(startFactor, out_channel, 1) 
           
        elif modelType==2:
          self.dconv_down1 = DoubleConv(1, startFactor)  
          self.dconv_down2 = DoubleConv(startFactor, startFactor*2)
          self.dconv_down3 = DoubleConv(startFactor*2, startFactor*4)
          self.dconv_down4 = DoubleConv(startFactor*4, startFactor*8)
          self.dconv_down5 = DoubleConv(startFactor*8, startFactor*16)  

          self.up4 = DoubleConv(startFactor*16 , startFactor*8)
          self.Att4 = AttentionBlock(F_g=startFactor*16,F_l=startFactor*8,F_int=startFactor*8)
          #self.Att4 = AttentionGate(startFactor*8,startFactor*8)
          self.dconv_up4 = DoubleConv(startFactor*16  , startFactor*8)

          self.up3 = DoubleConv(startFactor*8, startFactor*4)
          self.Att3 = AttentionBlock(F_g=startFactor*8,F_l=startFactor*4,F_int=startFactor*4)
          #self.Att3 = AttentionGate(startFactor*4,startFactor*4)
          self.dconv_up3 = DoubleConv(startFactor*8,  startFactor*4)

          self.up2 = DoubleConv( startFactor*4,  startFactor*2)
          self.Att2 = AttentionBlock(F_g=startFactor*4,F_l=startFactor*2,F_int= startFactor*2)
          #self.Att2 = AttentionGate(startFactor*2,startFactor*2)
          self.dconv_up2 = DoubleConv(startFactor*4  , startFactor*2)

          self.up1 = DoubleConv(startFactor*2, startFactor)
          self.Att1 = AttentionBlock(F_g=startFactor*2,F_l=startFactor,F_int=startFactor)
          #self.Att1 = AttentionGate(startFactor,startFactor)
          self.dconv_up1 = DoubleConv(startFactor*2  , startFactor) 

          self.conv_last = nn.Conv2d(startFactor, out_channel, 1)

        elif modelType==3:
          self.dconv_down1 = ResidualDoubleConv(1, startFactor)
          self.dconv_down2 = ResidualDoubleConv(startFactor, startFactor*2)
          self.dconv_down3 = ResidualDoubleConv(startFactor*2, startFactor*4)
          self.dconv_down4 = ResidualDoubleConv(startFactor*4, startFactor*8)
          self.dconv_down5 = ResidualDoubleConv(startFactor*8, startFactor*16)  

          self.dconv_up4 = ResidualDoubleConv(startFactor*16 + startFactor*8, startFactor*8 )
          self.dconv_up3 = ResidualDoubleConv(startFactor*8 + startFactor*4, startFactor*4)
          self.dconv_up2 = ResidualDoubleConv(startFactor*4 + startFactor*2, startFactor*2)
          self.dconv_up1 = ResidualDoubleConv(startFactor*2 + startFactor, startFactor)
          self.conv_last = nn.Conv2d(startFactor, out_channel, 1) 

        elif modelType==4:
          self.dconv_down1 = ImprovedResidualBlock(1, startFactor,64 )
          #self.res_conn1 = nn.Conv2d(1, startFactor  , kernel_size=3, padding=1,bias=True)
          self.res_conn1= DoubleConv(1, startFactor)
          nn.Conv2d(startFactor, out_channel, 1) 
          self.dconv_down2 = ImprovedResidualBlock(startFactor, startFactor*2,32 )
          #self.res_conn2 = nn.Conv2d(startFactor, startFactor*2, kernel_size=3, padding=1,bias=True )
          self.res_conn2= DoubleConv(startFactor, startFactor*2)
          self.dconv_down3 = ImprovedResidualBlock(startFactor*2, startFactor*4,16 )
          #self.res_conn3 = nn.Conv2d(startFactor*2, startFactor*4 , kernel_size=3, padding=1,bias=True)
          self.res_conn3= DoubleConv(startFactor*2, startFactor*4)
          self.dconv_down4 = ImprovedResidualBlock(startFactor*4, startFactor*8,8 )
          #self.res_conn4 = nn.Conv2d(startFactor*4, startFactor*8, kernel_size=3, padding=1,bias=True )
          self.res_conn4= DoubleConv(startFactor*4, startFactor*8)
          self.dconv_down5 = ImprovedResidualBlock(startFactor*8, startFactor*16,4 )   
          self.up4 = ResidualDoubleConv(startFactor*16 , startFactor*8)
          self.Att4 = DenoisingAttentionBlock(F_g=startFactor*16,F_l=startFactor*8,F_int=startFactor*8)
          #self.Att4 = AttentionGate(startFactor*8,startFactor*8)
          self.dconv_up4 = ResidualDoubleConv(startFactor*16  , startFactor*8)

          self.up3 = ResidualDoubleConv(startFactor*8, startFactor*4)
          self.Att3 = DenoisingAttentionBlock(F_g=startFactor*8,F_l=startFactor*4,F_int=startFactor*4)
          #self.Att3 = AttentionGate(startFactor*4,startFactor*4)
          self.dconv_up3 = ResidualDoubleConv(startFactor*8,  startFactor*4)

          self.up2 = ResidualDoubleConv( startFactor*4,  startFactor*2)
          self.Att2 = DenoisingAttentionBlock(F_g=startFactor*4,F_l=startFactor*2,F_int= startFactor*2)
          #self.Att2 = AttentionGate(startFactor*2,startFactor*2)
          self.dconv_up2 = ResidualDoubleConv(startFactor*4  , startFactor*2)

          self.up1 = ResidualDoubleConv(startFactor*2, startFactor)
          self.Att1 = DenoisingAttentionBlock(F_g=startFactor*2,F_l=startFactor,F_int=startFactor)
          #self.Att1 = AttentionGate(startFactor,startFactor)
          self.dconv_up1 = ResidualDoubleConv(startFactor*2  , startFactor) 

          self.conv_last = nn.Conv2d(startFactor, out_channel, 1, bias=True) 

    def forward(self, x): # [10,1,64,64] 
      if self.modelType==1  : 
        conv1 = self.dconv_down1(x) 
        x = self.maxpool(conv1)
        # [10,64,32,32] 
        conv2 = self.dconv_down2(x)
        # [10,128,32,32] 
        x = self.maxpool(conv2)
        
        # [10,128,16,16] 
        conv3 = self.dconv_down3(x)
        # [10,256,16,16] 

        x = self.maxpool(conv3)   

        # [10,256,8,8] 
        conv4 = self.dconv_down4(x)

        # [10,512,8,8] 

        x = self.maxpool(conv4)

        # [10,512,4,4] 

        x = self.dconv_down5(x)  

        # [10,1024,4,4] 
        x = self.upsample(x)     
        
        # [10,1024,8,8]    
        x = torch.cat([x, conv4], dim=1)   
        # x= [10,1024 + 512 ,8,8]    
        x = self.dconv_up4(x) 
          # x= [10, 512 ,8,8]    
        x = self.upsample(x)        

          # x= [10, 512 ,16,16]  
        x = torch.cat([x, conv3], dim=1)   
          # x= [10, 512 + 256 ,16,16]       
        x = self.dconv_up3(x)

          # x= [10,  256 ,16,16]       
        x = self.upsample(x)        

          # x= [10,  256 ,32,32]    
        x = torch.cat([x, conv2], dim=1)    
          # x= [10,  256 + 128,32,32]      
        x = self.dconv_up2(x)

          # x= [10,  128,32,32]     
        x = self.upsample(x)        
          # x= [10,  128,64,64]     
        x = torch.cat([x, conv1], dim=1)   
          # x= [10,  128 + 64,64,64]   
        x = self.dconv_up1(x) 
          # x= [10,64,64,64]   
        x = self.conv_last(x)
          # x= [10,1,64,64]   
        out = self.sigmoid(x) 
          # x= [10,1,64,64]   
        return out   

      elif self.modelType==2:  
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.dconv_down5(x)   
        x = self.upsample(x)  
        x = torch.cat([  self.Att4(g=x,x=conv4),self.up4(x)], dim=1)
        x = self.dconv_up4(x)  

        x = self.upsample(x) 
        x = torch.cat([  self.Att3(g=x,x=conv3),self.up3(x)], dim=1)
        x = self.dconv_up3(x)
        
        x = self.upsample(x)  
        x = torch.cat([  self.Att2(g=x,x=conv2),self.up2(x)], dim=1)
        x = self.dconv_up2(x) 
        
        x = self.upsample(x) 
        x = torch.cat([  self.Att1(g=x,x=conv1),self.up1(x)], dim=1)
        x = self.dconv_up1(x)
        x = self.conv_last(x)
        out = self.sigmoid(x) 
        return out
      elif self.modelType==3  : 
        conv1 = self.dconv_down1(x)
        resx= torch.clone(x) 
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.dconv_down5(x)  

        x = self.upsample(x)        
        x = torch.cat([x, conv4], dim=1) 
        x = self.dconv_up4(x) 

        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)        
        x = self.dconv_up3(x)

        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)      
        x = self.dconv_up2(x)

        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)  
        x= self.channelAttention(x)  
        x= self.spatialAttention(x)   
        x = self.dconv_up1(x)  
        x= resx -x  
        x = self.conv_last(x)
        out = self.sigmoid(x) 
        return out   
      elif self.modelType==4:  
        
        conv1 = self.dconv_down1(x) 
        res1= self.res_conn1(x)
        
        x = self.avgpool(conv1)
 
        conv2 = self.dconv_down2(x)
        res2= self.res_conn2(x) 

        x = self.avgpool(conv2)

        conv3 = self.dconv_down3(x) 
        res3= self.res_conn3(x) 

        x = self.avgpool(conv3)

        conv4 = self.dconv_down4(x) 
        res4= self.res_conn4(x) 

        x = self.avgpool(conv4)

        x = self.dconv_down5(x)    
        x = self.upsample(x)  
        x = torch.cat([  self.Att4(g=x,x=conv4),self.up4(x)], dim=1)
        x = self.dconv_up4(x)  

        x =  (x - res4)
        x = self.upsample(x) 
        x = torch.cat([  self.Att3(g=x,x=conv3),self.up3(x)], dim=1)
        x = self.dconv_up3(x)
        
        x =  (x - res3)
        x = self.upsample(x)  
        x = torch.cat([  self.Att2(g=x,x=conv2),self.up2(x)], dim=1)

        x = self.dconv_up2(x) 

        #x =  (x - res2)
        #print(x.shape)
        x = self.upsample(x)  
        x = torch.cat([  self.Att1(g=x,x=conv1),self.up1(x)], dim=1)
   
        x = self.dconv_up1(x)     
        #x =  (x - res1)
        x = self.conv_last(x)
        #x = x - ress
        out = self.sigmoid(x)
        return out    
     