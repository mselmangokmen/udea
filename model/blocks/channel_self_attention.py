

import torch
from torch import nn
from einops import rearrange
class ChannelSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__() 
        self.conv1=nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True  ) 
        self.conv2=nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0, bias=True) 
        self.conv3=nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1 , padding=0, bias=True ) 
          
        
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        #print('x shape: '+ str(x.shape)) 
        x1 = self.conv1(x)
        #print('x1 shape: '+ str(x1.shape)) 
        x1=rearrange(x1, 'b c h w -> b c (h w)')
        x1=torch.unsqueeze(x1,dim=-2) 
        x1= self.softmax(x1)
        #print('x1 shape softmax : '+ str(x1.shape)) 
  
        x2=self.conv2(x)
        #print('x2 shape conv2 : '+ str(x2.shape)) 
        x2=rearrange(x2, 'b c h w -> b c (h w)') 
        x2=torch.unsqueeze(x2,dim=-1) 
        x3=torch.matmul(x1,x2)
        #print('x3 shape : '+ str(x3.shape)) 
        x3=self.conv3(x3)  
        x3 = self.sigmoid(x3)
        #print('x3 sigmoid : '+ str(x3.shape)) 
        out = x3*x1
        #print('out shape : '+ str(out.shape)) 
        out = torch.squeeze(out)
        #print('out shape : '+ str(out.shape)) 
        if(len(out.shape)<=2):
          out = torch.unsqueeze(out,dim=0)
        out=rearrange(out, 'b c (h w) -> b c h w',h=64,w=64) 
        #print('out shape rearrange: '+ str(out.shape)) 
        return out