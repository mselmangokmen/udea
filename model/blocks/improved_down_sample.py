from model.blocks.double_conv import DoubleConv

from torch import nn

class ImprovedDownSample(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels,mid_channels=(in_channels+ out_channels)//2) 
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.lppool = nn.LPPool2d(2, kernel_size=2, stride=2, ceil_mode=False)           
        self.Conv2d =nn.Conv2d(in_channels, out_channels, 3, padding=1,bias=True)
        self.norm = nn.BatchNorm2d(out_channels)  
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):  
        residual = x.clone() 

        x = self.double_conv(x)
        x = self.avgpool(x)
        residual = self.Conv2d(residual)     
        residual = self.lppool(residual)  
        residual = self.norm(residual)      
        residual = self.relu(residual)         


        #x = self.avgpool(x)   


        out = (x + residual)/2
     
        #out = self.double_conv(out)
        return out
