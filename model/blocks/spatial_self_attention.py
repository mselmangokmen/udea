

import torch
from torch import nn
from einops import rearrange

class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__() 
        self.conv1=nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0, bias=True  ) 
        self.conv2=nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0, bias=True  )  
        self.conv3=nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1 , padding=0, bias=True ) 
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        #print('x shape: '+ str(x.shape)) 
        x1 = self.conv1(x)
        x1 = self.avgpool(x1)
        #print('x1 avgpool: '+ str(x1.shape)) 
        x1=rearrange(x1, 'b c h w -> b (h w) c ') 
        #print('x1 rearrange: '+ str(x1.shape))   
        x1= self.softmax(x1)
        #print('x1 softmax: '+ str(x1.shape))   
  
        x2=self.conv2(x)
        #print('x2 shape conv2 : '+ str(x2.shape)) 
        x2=rearrange(x2, 'b c h w -> b c (h w)') 
        #print('x2 rearrange: '+ str(x2.shape))   
        #x2=torch.unsqueeze(x2,dim=-1)
        #print('x2 shape rearranged : '+ str(x2.shape)) 
        x3=torch.matmul(x1,x2)
        #print('x3 shape: '+ str(x3.shape))   
        x3= self.sigmoid(x3)
        #print('x3 sigmoid: '+ str(x3.shape))   
        x3=rearrange(x3, 'b c (h w)-> b c h w',h=64,w=64) 
        #print('x3 rearrange : '+ str(x3.shape))  
        out = x * x3
        #print('out rearrange : '+ str(out.shape))  
        return out