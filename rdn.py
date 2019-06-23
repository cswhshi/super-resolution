import torch
import torch.optim as optim
import torch.nn as nn
import time

class basic(nn.Module):
    '''
    give you some method
    '''

    def __init__(self,opts = None):
        super(basic,self).__init__()
        self.model_name = str(type(self))
    def load(self,path):
        self.load_state_dict(torch.load(path))
    def save(self,name= None):
        if name == None:
            prefix = './check_point/'+self.model_name+"_"
            name = time.strftime(prefix +"%m%d_%H:%M:%S.path") 
        torch.save(self.state_dict(),name)
        return name


 
class one_conv(nn.Module):
    def __init__(self,inchanels,growth_rate,kernel_size = 3):
        super(one_conv,self).__init__()
        self.conv = nn.Conv2d(inchanels,growth_rate,kernel_size=kernel_size,padding = kernel_size>>1,stride= 1)
        self.relu = nn.ReLU()
    def forward(self,x):
        output = self.relu(self.conv(x))
        return torch.cat((x,output),1)    
    
    
    
class RDB(nn.Module):
    '''
    C：RDB中的conv层数  6
    G：the growth rate 32
    G0：local and global feature fusion layers 64filter
    '''
    def __init__(self,G0,C,G,kernel_size = 3):
        super(RDB,self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G,G))
        self.conv = nn.Sequential(*convs)
        
        #local_feature_fusion 
        self.LFF = nn.Conv2d(G0+C*G,G0,kernel_size = 1,padding = 0,stride =1)
    def forward(self,x):
        out = self.conv(x)
        lff = self.LFF(out)
        #local residual learning
        return lff + x

'''    
rdb=RDB(64,6,32)    
rdb
'''


class rdn(basic):
    def __init__(self,opts):
        '''
        opts: the system para
        '''
        super(rdn,self).__init__()
        '''
        D: RDB number 20
        C: the number of conv layer in RDB 6
        G: the growth rate 32
        G0:local and global feature fusion layers 64filter
        '''
        
        self.D = opts["D"]
        self.C = opts["C"]
        self.G = opts["G"]
        self.G0 = opts["G0"]
        
        print("D:{},C:{},G:{},G0:{}".format(self.D,self.C,self.G,self.G0))
        
        kernel_size =opts["kernel_size"]  #卷积核大小
        
        input_channels = opts["input_channels"]  #输入特征图的大小
        out_channels = opts["out_channels"]
        #shallow feature extraction 
        self.SFE1 = nn.Conv2d(input_channels,self.G0,kernel_size=kernel_size,padding = kernel_size>>1,stride= 1)
        self.SFE2 = nn.Conv2d(self.G0,self.G0,kernel_size=kernel_size,padding = kernel_size>>1,stride =1)
       
        #RDB for paper we have D RDB block
        '''20个RDB'''
        self.RDBS = nn.ModuleList()
        for d in range(self.D):
            self.RDBS.append(RDB(self.G0,self.C,self.G,kernel_size))
        
        
        #Global feature fusion
        self.GFF = nn.Sequential(
               nn.Conv2d(self.D*self.G0,self.G0,kernel_size = 1,padding = 0 ,stride= 1),
               nn.Conv2d(self.G0,self.G0,kernel_size,padding = kernel_size>>1,stride = 1),
        )
        
        #upsample net 
        '''
        进行4倍的上采样
        '''
        self.up_net = nn.Sequential(
                nn.Conv2d(self.G0,self.G*4,kernel_size=kernel_size,padding = kernel_size>>1,stride = 1),
                nn.PixelShuffle(2),
                nn.Conv2d(self.G,self.G*4,kernel_size = kernel_size,padding =kernel_size>>1,stride = 1),
                nn.PixelShuffle(2),
                nn.Conv2d(self.G,out_channels,kernel_size=kernel_size,padding = kernel_size>>1,stride = 1)
        )
        #init
        for para in self.modules():
            if isinstance(para,nn.Conv2d):
                nn.init.kaiming_normal_(para.weight)
                if para.bias is not None:
                    para.bias.data.zero_()

    def forward(self,x):
        #f-1
        f__1 = self.SFE1(x)
        out  = self.SFE2(f__1)
        
        RDB_outs = []
        for i in range(self.D):#D: RDB number 20
            out = self.RDBS[i](out)
            RDB_outs.append(out)
        out = torch.cat(RDB_outs,1)  #将RDB的全部特征都拼接起来
        
        ##全局特征融合
        out = self.GFF(out)   
        out = f__1+out     #Global Residual Learning
        
        return self.up_net(out)  #上采样

'''
opts={"D":20,"C":6,"G":32,"G0":64,"kernel_size":3,"input_channels":1,"out_channels":1}
rdn_net=rdn(opts)

'''





