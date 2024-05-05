import torch
from torch import nn
import torch.nn.functional as F
class BR(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(BR, self).__init__()
        self.block=nn.Sequential(
        nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1,padding=1,bias=False),
        nn.ReLU(),
        nn.Conv2d(out_ch,out_ch,kernel_size=1,stride=1,padding=0,bias=False)
        )
    def forward(self,out):
        out=self.block(out)
        return out










class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class scSE(nn.Module):
    def __init__(self, in_channels,out):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)
        self.block=BR(in_channels,out)
    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return self.block(U_cse+U_sse)
class MSV(nn.Module):
    def __init__(self):
        super(MSV, self).__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(64,16,kernel_size=11,stride=1,padding=5,bias=False),
            nn.ReLU(),
            nn.Conv2d(16,16,kernel_size=9,stride=1,padding=4,bias=False),
            nn.ReLU(),
            nn.Conv2d(16,64,kernel_size=7,stride=1,padding=3,bias=False),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU()
        )
    def forward(self,x):
        x=x+self.block1(x)
        x=x+self.block2(x)
        x = x + self.block3(x)
        return  x
# [1,3,512,512]
class PM_LSMN(nn.Module):
    def __init__(self):
        super(PM_LSMN, self).__init__()
        self.block1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2 = scSE(64, 64)
        self.block3 = MSV()
        self.outconv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1) + x1
        x3 = self.block2(x2) + x2
        x = x + self.outconv(self.block3(x3))
        return x





















class atten(nn.Module):
    def __init__(self,in_ch):
        super(atten, self).__init__()
        self.block=scSE(in_ch,in_ch)
    def forward(self,x):
        x=self.block(x)+x
        x=self.block(x)+x
        return x
class remove_net_du(nn.Module):
    def __init__(self):
        super(remove_net_du, self).__init__()
        self.block1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.block2=atten(64)
        self.block3=MSV()
        self.down=nn.AvgPool2d(2)
        self.up=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.outconv=nn.Conv2d(64,3,kernel_size=3,stride=1,padding=1,bias=False)
    def forward(self,x):
        x1=self.block1(x)
        x2=self.block2(x1)
        x3=self.down(x2)
        x4=self.block2(x3)
        x5=self.down(x4)
        x6=self.block2(x5)
        x7 = self.down(x6)
        x8 = self.block3(x7)
        x9=self.up(x8)+x6
        # x9=F.interpolate(x8,scale_factor=2,mode="bilinear",align_corners=True)+x6
        x10=self.block2(x9)
        x11 = self.up(x10) + x4
        x12 = self.block2(x11)
        x13 = self.up(x12) + x2
        x14 = self.block2(x13)
        out=self.outconv(x14)+x
        return out
# net = remove_net()
# x = torch.rand(1,3,256,256)
# print(net(x).shape)