import torch
from torch import nn
import torch.nn.functional as F
class msconv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(msconv, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class LEFF(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LEFF, self).__init__()
        # 也相当于分组为1的分组卷积
        self.conv1=nn.Sequential(
            nn.Conv2d(in_ch,in_ch,kernel_size=1,stride=1,padding=0,bias=False),
            nn.GELU(),
        )
        self.conv2=msconv(in_ch,out_ch)
        self.ge=nn.GELU()
        self.conv3=nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, input):
        out=self.conv3(self.conv2(self.conv1(input)))
        return out

class convs(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(convs, self).__init__()
        self.conv=nn.Conv2d(in_ch,in_ch,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv2=msconv(in_ch,out_ch)
        self.conv3 = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self,x):
        out=self.conv3(self.conv2(self.conv(x)))
        return out
class ccb(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(ccb, self).__init__()
        # 也相当于分组为1的分组卷积
        self.bn=nn.BatchNorm2d(in_ch)
        self.conv=convs(in_ch,out_ch)
        self.le=LEFF(in_ch,out_ch)
    def forward(self,input):
        out1=self.conv(self.bn(input))+input
        out=out1+self.le(self.le(out1))
        return out
class MDRN(nn.Module):
    def __init__(self):
        super(MDRN, self).__init__()
        self.in_conv=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.out_conv=nn.Sequential(
            nn.Conv2d(16,3,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.block=msconv(16,16)
        self.bn=nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
    def forward(self,x):
        x1=self.in_conv(x)
        for i in range(0,26):
            x1=x1+self.bn(self.block(x1))
        x=self.out_conv(x1)+x
        return x



# net =MDRN()
# x = torch.rand(1,3,256,256)
# print(net(x).shape)