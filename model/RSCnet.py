import torch
import torchvision
import torch
import torch.nn as nn
class RSCnet(nn.Module):
    def __init__(self):
        super(RSCnet, self).__init__()
        self.block1=nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1,bias=False)
        self.block2=nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1,bias=False)
        self.block3=nn.ConvTranspose2d(32,32,kernel_size=3,stride=1,padding=1,bias=False)
        self.block4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block5=nn.Conv2d(64, 3 , kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self,x): 
        x1=self.block1(x)
        x2=self.block2(x1)
        x3 = self.block2(x2)
        x4 = self.block2(x3)
        x5 = self.block2(x4)
        x6=torch.cat([self.block3(x5),x5],dim=1)
        x7 = torch.cat([self.block4(x6), x4], dim=1)
        x8 = torch.cat([self.block4(x7), x3], dim=1)
        x9 = torch.cat([self.block4(x8), x2], dim=1)
        x10 = torch.cat([self.block4(x9), x1], dim=1)
        x=self.block5(x10)
        return x
# a=torch.randn(1,3,512,512)
# net=RSCnet()
# print(net(a).shape)