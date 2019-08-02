import torch.nn as nn
class CA_Block(nn.Module):
    def __init__(self,in_channels,reduction):
        super(CA_Block,self).__init__()
        self.se_module=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels,in_channels//reduction,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction,in_channels,kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=x*self.se_module(x)
        return x
    
class RCAB_Block(nn.Module):
    def __init__(self,in_channels,reduction):
        super(RCAB_Block,self).__init__()
        self.rcab=nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            CA_Block(in_channels, reduction)
        )
    def forward(self,x):
        return x+self.rcab(x)
       
class RG_Block(nn.Module):
    def __init__(self,in_channels,num_crab,reduction):
        super(RG_Block,self).__init__()
        self.rg_block=[RCAB_Block(in_channels,reduction) for _ in range(num_crab)]
        self.rg_block.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
        self.rg_block=nn.Sequential(*self.rg_block)
    def forward(self,x):
        return x+self.rg_block(x)
    

class RCAN(nn.Module):
    def __init__(self):
        super(RCAN, self).__init__()
        """
        超参数设置
        """
        scale = 4                 #放大倍数
        feature_channels = 64     #特征图通道数
        num_rg = 10               #RG的个数，论文中设置为10
        num_rcab = 20             #rcab的个数，论文中设置为20
        reduction = 16            #缩放倍数，论文中设置为16
        
        #浅层特征提取层
        self.sf = nn.Conv2d(3, feature_channels, kernel_size=3, padding=1)
        
        #residual in residual (RIR) 深度特征提取
        self.rgs = nn.Sequential(*[RG_Block(feature_channels, num_rcab, reduction) for _ in range(num_rg)])
        self.conv1 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        
        #上采样层
        self.upscale = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        
        #最后一个卷积层，输出为3通道的彩色图像
        self.conv2 = nn.Conv2d(feature_channels, 3, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.sf(x)
        residual = x
        x = self.rgs(x)
        x = self.conv1(x)
        x += residual                #长跳跃连接，LSC
        x = self.upscale(x)
        x = self.conv2(x)
        return x
