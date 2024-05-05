import torch
from torch import nn
import torch.nn.functional as F
class stc(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(stc, self).__init__()
        self.flag = out_ch
        self.CNN = nn.Sequential(
            nn.Conv2d(60, 60, kernel_size=(3, 3), dilation= 2,padding=2,bias=False),
            nn.ReLU(),
            nn.Conv2d(60, 60, kernel_size=(3, 3), dilation=3, padding=3,bias=False),
            nn.ReLU(),
            nn.Conv2d(60, 60, kernel_size=(3, 3), dilation=2, padding=2,bias=False),
            nn.ReLU()
        )
        self.conv1=nn.Conv2d(in_ch,30,kernel_size=3,stride=1,padding=1,bias=False)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(60,20,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv3 = nn.Conv2d(60, 20, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv4 = nn.Conv2d(60, 20, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv5 = nn.Conv2d(in_ch, 60, kernel_size=3, stride=1, padding=1, bias=False)
        self.CNN2 = nn.Sequential(
            nn.Conv2d(60, 60, kernel_size=(3, 3), dilation=1, padding=1, bias=False),
            nn.ReLU(),)
        self.outconv=nn.Conv2d(60,out_ch,kernel_size=3,stride=1,padding=1,bias=False)
    def forward(self,x1,x2,x3):
        x11=self.conv1(x1)
        x22=self.conv1(x2)
        x12=torch.cat([x11,x22],dim=1)
        # x12=torch.cat([torch.cat([self.conv2(x12),self.conv3(x12)],dim=1),self.conv4(x12)],dim=1)
        x121=torch.cat([torch.cat([self.conv2(x12),self.conv3(x12)],dim=1),self.conv4(x12)],dim=1)+x12
        x0=self.CNN2(x121)
        x10=self.conv5(x3)
        x8 = self.CNN2(x0)+x10
        x9=self.CNN2(self.CNN(x8)+x10)+x0
        out=self.outconv(x9)
        return out
# a=torch.randn(1,1,256,256)
# net=stc(1,1)
# print(net(a,a,a).shape)