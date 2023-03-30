

import torch 
from torch import nn
from model.blocks.attention_block import AttentionBlock
from model.blocks.double_conv import DoubleConv
from model.blocks.residual_double_block import ResidualDoubleConv

#https://github.com/JavierGurrola/RDUNet
#

class UNet(nn.Module):

    def __init__(self, out_channel=1,modelType=2,startFactor=16):

        super().__init__() 
        self.modelType= modelType
        if modelType==1:
          self.dconv_down1 = ResidualDoubleConv(1, startFactor)
          self.dconv_down2 = ResidualDoubleConv(startFactor, startFactor*2)
          self.dconv_down3 = ResidualDoubleConv(startFactor*2, startFactor*4)
          self.dconv_down4 = ResidualDoubleConv(startFactor*4, startFactor*8)
          self.dconv_down5 = ResidualDoubleConv(startFactor*8, startFactor*16)  
        elif modelType==2:
          self.dconv_down1 = DoubleConv(1, startFactor)
          self.dconv_down2 = DoubleConv(startFactor, startFactor*2)
          self.dconv_down3 = DoubleConv(startFactor*2, startFactor*4)
          self.dconv_down4 = DoubleConv(startFactor*4, startFactor*8)
          self.dconv_down5 = DoubleConv(startFactor*8, startFactor*16)  
        elif modelType==3:
          self.dconv_down1 = DoubleConv(1, startFactor)
          self.dconv_down2 = DoubleConv(startFactor, startFactor*2)
          self.dconv_down3 = DoubleConv(startFactor*2, startFactor*4)
          self.dconv_down4 = DoubleConv(startFactor*4, startFactor*8)
          self.dconv_down5 = DoubleConv(startFactor*8, startFactor*16)  
 
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))       

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, 
                                    mode='bilinear', align_corners=True)        
        self.sigmoid = nn.Sigmoid()
        if modelType==1:
          self.dconv_up4 = ResidualDoubleConv(startFactor*16 + startFactor*8, startFactor*8 )
          self.dconv_up3 = ResidualDoubleConv(startFactor*8 + startFactor*4, startFactor*4)
          self.dconv_up2 = ResidualDoubleConv(startFactor*4 + startFactor*2, startFactor*2)
          self.dconv_up1 = ResidualDoubleConv(startFactor*2 + startFactor, startFactor)
          self.conv_last = nn.Conv2d(startFactor, out_channel, 1)
        elif modelType==2:
          self.dconv_up4 = DoubleConv(startFactor*16 + startFactor*8, startFactor*8 )
          self.dconv_up3 = DoubleConv(startFactor*8 + startFactor*4, startFactor*4)
          self.dconv_up2 = DoubleConv(startFactor*4 + startFactor*2, startFactor*2)
          self.dconv_up1 = DoubleConv(startFactor*2 + startFactor, startFactor)
          self.conv_last = nn.Conv2d(startFactor, out_channel, 1)
 

        elif modelType==3: 
          self.up4 = DoubleConv(startFactor*16 , startFactor*8)
          self.Att4 = AttentionBlock(F_g=startFactor*8,F_l=startFactor*8,F_int=128)
          self.dconv_up4 = DoubleConv(startFactor*16  + startFactor*8, startFactor*8)

          self.up3 = DoubleConv(startFactor*8, startFactor*4)
          self.Att3 = AttentionBlock(F_g=startFactor*4,F_l=startFactor*4,F_int=64)
          self.dconv_up3 = DoubleConv(startFactor*8 + startFactor*4,  startFactor*4)

          self.up2 = DoubleConv( startFactor*4,  startFactor*2)
          self.Att2 = AttentionBlock(F_g=startFactor*2,F_l=startFactor*2,F_int=32)
          self.dconv_up2 = DoubleConv(startFactor*4 + startFactor*2, startFactor*2)

          self.up1 = DoubleConv(startFactor*2, startFactor)
          self.Att1 = AttentionBlock(F_g=startFactor,F_l=startFactor,F_int=16)
          self.dconv_up1 = DoubleConv(startFactor*2 + startFactor, startFactor) 

          self.conv_last = nn.Conv2d(startFactor, out_channel, 1)

           
    def forward(self, x):
      if self.modelType==1 or self.modelType==2 : 
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
        x = self.dconv_up1(x)
        
        x = self.conv_last(x)
        out = self.sigmoid(x)
 
        return out 
  

      elif self.modelType==3:  
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
        x = torch.cat([x, self.Att4(g=self.up4(x),x=conv4)], dim=1)
        x = self.dconv_up4(x) 

        x = self.upsample(x) 
        x = torch.cat([x, self.Att3(g=self.up3(x),x=conv3)], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)  
        x = torch.cat([x, self.Att2(g=self.up2(x),x=conv2)], dim=1) 
        x = self.dconv_up2(x) 

        x = self.upsample(x) 
        x = torch.cat([x, self.Att1(g=self.up1(x),x=conv1)], dim=1)
        x = self.dconv_up1(x)
        x = self.conv_last(x)
        out = self.sigmoid(x)
        return out