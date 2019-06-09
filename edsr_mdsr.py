# -*- coding: utf-8 -*-
import os
import math
import torch
import torch.nn as nn
#import torch.nn.functional as F
import numpy as np


def conv(inp, oup, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = ((kernel_size -1) * dilation + 1) // 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias),
    )
    
    
class ResBlock(nn.Module):
    def __init__(self, inp, kernel_size=3, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        modules = []
        for i in range(2):
            modules.append(conv(inp, inp, kernel_size, bias=bias))
            if bn: 
                modules.append(nn.BatchNorm2d(inp))
            if i == 0: 
                modules.append(act)
        self.body = nn.Sequential(*modules)
        self.res_scale = res_scale
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

#upsample
class Upsampler(nn.Sequential):
    def __init__(self, scale, inp, bn=False, act=False, bias=True, choice=0):
        modules = []
        if choice == 0: #subpixel
           if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
               for _ in range(int(math.log(scale, 2))):
                   modules.append(conv(inp, 4 * inp, 3, bias=bias))
                   modules.append(nn.PixelShuffle(2))
                   if bn:
                       modules.append(nn.BatchNorm2d(inp))
                   if act:
                       modules.append(act())
           elif scale == 3:
               modules.append(conv(inp, 9 * inp, 3, bias=bias))
               modules.append(nn.PixelShuffle(3))
               if bn: 
                   modules.append(nn.BatchNorm2d(inp))
               if act: 
                   modules.append(act())
           else:
               raise NotImplementedError
        elif choice == 1: #decov反卷积
           modules.append(nn.ConvTranspose2d(inp, inp, scale, stride=scale))
        else: #bilinear  #线性插值上采样
           modules.append(nn.Upsample(mode='bilinear', scale_factor=scale, align_corners=True))
        super(Upsampler, self).__init__(*modules)



class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range,rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)  #方差
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False

'''
resBlock=ResBlock(64)

print(resBlock)


# head

input_channel = 5      #输入通道
inp = 64
head = nn.Sequential(conv(3, inp, input_channel) )
print(head)


# body  
scale_list=[1,2,3,4]
act = nn.ReLU(True)
scale = scale_list[0]
num_block=16
res_scale = 0.1

body = nn.Sequential( *[ResBlock(inp, bias = True, act = act, res_scale = res_scale) for _ in range( num_block) ] )
body.add_module( str(num_block),conv(inp, inp, 3) )
print(body)


#MDSR pre_process
inp=64
scale_list=[2,3,4]
act = nn.ReLU(True)
scale = scale_list[0]
num_block=16
res_scale = 0.1

pre_process = nn.ModuleDict( 
        [str(scale), 
         nn.Sequential( ResBlock( inp, bias = True, act = act, res_scale = res_scale ), 
         ResBlock( inp, bias = True, act = act, res_scale = res_scale ) ) ] for scale in scale_list )
print(pre_process)

#EDSR upsample
# tail     
scale=2
inp=64

output_channel=3
if scale > 1:
    tail = nn.Sequential( *[ Upsampler(scale, inp, act = False, choice = 0), conv(inp, 3, output_channel) ] )
else:
    tail = nn.Sequential( *[ conv(inp, 3, output_channel) ] )      
print(tail)         



#MDSR upsample
inp=64
scale_list=[2,3,4]
act = nn.ReLU(True)
scale = scale_list[0]
num_block=16
res_scale = 0.1

upsample = nn.ModuleDict( [ str(scale), Upsampler(scale, inp, act = False, choice = 0) ] for scale in [2,3,4])
print(upsample)
'''




class EDSR(nn.Module):
    def __init__(self, scale_list, model_path = 'weight/EDSR_weight.pt' ):
        super(EDSR, self).__init__()
        # args
        scale = scale_list[0]
        input_channel = 5      #第一个kernel_size大小
        output_channel = 3     #输出通道
        num_block = 16         #block的数量
        inp = 64               #
        rgb_range = 255  
        res_scale = 0.1
        act = nn.ReLU(True)   #激活函数
        
        # head （残差结构前的一个卷积）
        self.head = nn.Sequential(conv(3, inp, input_channel) )

        # body （残差结构）
        self.body = nn.Sequential( *[ResBlock(inp, bias = True, act = act, res_scale = res_scale) for _ in range(num_block)] )
        self.body.add_module( str(num_block), conv(inp, inp, 3) )
        
        # tail（上采样）
        if scale > 1:
            self.tail = nn.Sequential(*[ Upsampler(scale, inp, act = False, choice = 0), conv(inp, 3, output_channel) ] )
        else:
            self.tail = nn.Sequential(*[ conv(inp, 3, output_channel)])
        self.sub_mean = MeanShift(rgb_range, sign = -1)
        self.add_mean = MeanShift(rgb_range, sign = 1)
        
        self.model_path = model_path 
        self.load()

    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def forward_pred( self, x, scale_id ):
        scale = torch.from_numpy( np.array( [scale_id] ) )
        return self.forward( x, scale )
    
    
    def _initialize_weights(self):
        for (name, m) in self.named_modules():
            if name.endswith('_mean'):
                print ('Do not initilize {}'.format(name) )
            elif isinstance(m, nn.Conv2d) and isinstance( m, nn.ReLU ):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d) and isinstance( m, nn.LeakyReLU):
                nn.init.kaiming_normal_(m.weight, a = 0.05, mode = 'fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    def savemodel(self):
        torch.save( self.state_dict(), self.model_path );

    def load(self):
        if os.path.exists(self.model_path):
            model = torch.load(self.model_path)
            self.load_state_dict(model)
        else:
            self._initialize_weights()
   
     
class MDSR(nn.Module):
    def __init__(self, scale_list, model_path='weight/MDSR_weight.pt'):
        super(MDSR, self).__init__()
        
        # args
        self.scale_list = scale_list
        input_channel = 3
        output_channel = 3
        num_block = 32 
        inp = 64
        rgb_range = 255 
        res_scale = 0.1
        act = nn.ReLU(True)
        #act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        # head
        self.head = nn.Sequential( conv(3, inp, input_channel) )

        # pre_process
        self.pre_process = nn.ModuleDict( [str(scale), nn.Sequential( ResBlock( inp, bias = True, act = act, res_scale = res_scale ), ResBlock( inp, bias = True, act = act, res_scale = res_scale ) ) ] for scale in self.scale_list )

        # body
        self.body = nn.Sequential( *[ResBlock(inp, bias = True, act = act, res_scale = res_scale) for _ in range( num_block ) ] )
        self.body.add_module(str(num_block),conv(inp, inp, 3))

        #upsample
        self.upsample = nn.ModuleDict( [ str(scale), Upsampler(scale, inp, act = False, choice = 0) ]for scale in self.scale_list)

        # tail
        self.tail = nn.Sequential( conv(inp, 3, output_channel) )

        self.sub_mean = MeanShift(rgb_range, sign = -1)
        self.add_mean = MeanShift(rgb_range, sign = 1)

        self.model_path = model_path
        self.load()

    def forward(self, x, scale):
        scale_id = str(scale[0].item() )
        x = self.sub_mean(x)
        x = self.head(x)
        x = self.pre_process[scale_id](x)

        res = self.body(x)
        res += x 

        x = self.upsample[scale_id](res)
        x = self.tail(x)
        x = self.add_mean(x)
        return x

    def forward_pred( self, x, scale_id ):
        scale = torch.from_numpy( np.array( [scale_id] ) )
        return self.forward( x, scale )
    
    
    def _initialize_weights(self):
        for (name, m) in self.named_modules():
            if name.endswith('_mean'):
                print ('Do not initilize {}'.format(name) )
            elif isinstance(m, nn.Conv2d) and isinstance( m, nn.ReLU ):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d) and isinstance( m, nn.LeakyReLU):
                nn.init.kaiming_normal_(m.weight, a = 0.05, mode = 'fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def savemodel(self):
        torch.save( self.state_dict(), self.model_path );

    def load(self):
        if os.path.exists(self.model_path):
            model = torch.load(self.model_path)
            self.load_state_dict(model)
        else:
            self._initialize_weights()