

from torch import nn




class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.seq =  nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, 3, padding=1,bias=True),
        nn.BatchNorm2d(mid_channels),
        
        #nn.LeakyReLU(0.2, inplace=True),
        nn.ReLU( ),
        #nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, 3, padding=1,bias=True),
        nn.BatchNorm2d(out_channels),
        #nn.LeakyReLU(0.2, inplace=True),
        nn.ReLU( ),
        #nn.Softmax(),
        #nn.ReLU(inplace=True),
        nn.Dropout(0.2)
    )
    
    def forward(self, x):
        out = self.seq(x)
        return out

 