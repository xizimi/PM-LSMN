import torch
import torchvision
import torch
import torch.nn as nn
class res(nn.Module):
    def __init__(self):
        super(res, self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
    def forward(self,x):
        for i in range(0,5):
            x=x+self.block(x)
        return x
class colorg(nn.Module):
    def __init__(self):
        super(colorg, self).__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.block3=nn.Sequential(
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.re=res()
    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.re(x)
        x=self.block3(x)
        x=self.block4(x)
        return x
class colord(nn.Module):
    def __init__(self):
        super(colord, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
    def forward(self,x):
        x=self.block1(x)
        return x
# a=torch.randn(1,3,512,512)
# net1=colorg()
# net2=colord()
# print(net2(a).shape)