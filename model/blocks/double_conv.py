

from torch import nn




class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__() 

        mid_channels= (out_channels+ in_channels)//2
        self.seq =  nn.Sequential( # [10,1,64,64]
        nn.Conv2d(in_channels, mid_channels, 3, padding=1,bias=True),
        #nn.BatchNorm2d(mid_channels),  # [10,64,64,64]
        nn.InstanceNorm2d(mid_channels),# [10,64,64,64]
        #nn.ReLU(),
        nn.LeakyReLU(0.1, inplace=True),   
 

        nn.Conv2d(mid_channels, out_channels, 3, padding=1,bias=True),
        #nn.BatchNorm2d(out_channels),
        nn.InstanceNorm2d(mid_channels),
        #nn.ReLU(),
        nn.LeakyReLU(0.1, inplace=True), 
        nn.Dropout(0.2)
        # [10,64,64,64]
    )
    
    def forward(self, x):
        out = self.seq(x)
        return out
