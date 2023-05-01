from model.blocks.double_conv import DoubleConv


from torch import nn
class ImprovedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,w_size):
        super().__init__()
        mid_channel = (out_channels + in_channels)//2
        self.conv1=   nn.Conv2d(in_channels, mid_channel, kernel_size=3, padding=1,bias=True)
        self.conv3=   nn.Conv2d(mid_channel, out_channels, kernel_size=3, padding=1,bias=True)
        self.conv2=   DoubleConv(mid_channel,out_channels) 
        self.lap=   nn.Conv2d(in_channels, mid_channel, kernel_size=3, padding=1,bias=True)

        #self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2) 
        #self.lppool = nn.LPPool2d(2, kernel_size=2, stride=2, ceil_mode=False)         
        #self.maxpool = nn.MaxPool2d(2)         
        self.avgpool = nn.AdaptiveAvgPool2d( (w_size,w_size) ) 
        self.norm = nn.InstanceNorm2d(out_channels)# [10,64,64,64]

        #self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=True)
 
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.drop =  nn.Dropout(0.1)
    def forward(self, x): 
        residual = x.clone() 
        #residual = self.upsample(residual)  
        residual = self.avgpool(residual)
        residual = self.lap(residual) 
        residual = self.norm(residual) 
        residual= self.relu(residual)
        residual= self.drop(residual)

        x= self.conv1(x)

        x = (x+ residual )
        out = self.conv3(x) 
        out = self.norm(out) 
        out= self.relu(out)
        out= self.drop(out)

        return  out