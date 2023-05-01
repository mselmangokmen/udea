

from torch import nn

from model.blocks.double_conv import DoubleConv



class ResidualDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels,mid_channels=(in_channels+ out_channels)//2) 
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        else:
            self.residual_conv = None
    
    def forward(self, x):
        residual = x
        out = self.double_conv(x)
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
            
        out = (out + residual)
        return out