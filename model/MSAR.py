import torch
from torch import nn
import torch.nn.functional as F
class CAB(nn.Module):
    def __init__(self):
        super(CAB, self).__init__()
        self.block1=nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.block2=nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.block3=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64,8,kernel_size=1,stride=1,padding=0,bias=False),
            nn.ReLU(),
            nn.Conv2d(8,64,kernel_size=1,stride=1,padding=0,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        x1=self.block1(x)+x
        x2=self.block2(x1)
        x=x+x2*self.block3(x2)
        return x
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
class MSAR(nn.Module):
    def __init__(self):
        super(MSAR, self).__init__()
        self.conv=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.block1=CAB()
        self.block2=MSV()
        self.outconv=nn.Conv2d(64,3,kernel_size=3,stride=1,padding=1,bias=False)
    def forward(self,x):
        x=self.conv(x)
        x=self.block1(self.block1(self.block1(x)))
        x=self.block2(x)
        x=self.outconv(x)
        return x
# a=torch.randn(1,3,512,512)
# net=MSAR()
# print(net(a,a).shape)