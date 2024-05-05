import torch
from torch import nn
import torch.nn.functional as F
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

class CMP(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(CMP, self).__init__()
        self.conv1=nn.Conv2d(in_ch,out_ch,kernel_size=1,bias=False,stride=1)
        self.conv2=nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=6, padding=6, bias=False,stride=1),
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=12, padding=12, bias=False,stride=1),
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=18, padding=18, bias=False,stride=1),
        )
        self.dr=nn.Dropout(0.5)
    def forward(self,x):
        x1=self.conv1(x)
        x2=self.conv2(x)
        x3=self.conv3(x)
        x4=self.conv4(x)
        return self.se(self.dr(x1+x2+x3+x4))
class PPM(nn.Module):
    def __init__(self, in_dim, out_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
        self.block=nn.Sequential(
            nn.Conv2d(in_dim+4*out_dim,in_dim,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, bias=False)
        )
    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            temp = f(x)
            temp = F.interpolate(temp, x_size[2:], mode="bilinear", align_corners=True)
            out.append(temp)
        out=torch.cat(out,1)
        return
class FPM(nn.Module):
    def __init__(self,ch):
        super(FPM, self).__init__()
        self.block1=nn.Conv2d(ch,256,kernel_size=1,dilation=1,padding=0,bias=False,stride=1)
        self.block2 = nn.Conv2d(ch, 256, kernel_size=3, dilation=6, padding=6, bias=False,stride=1)
        self.block3 = nn.Conv2d(ch, 256, kernel_size=3, dilation=12, padding=12, bias=False,stride=1)
        self.block4 = nn.Conv2d(ch, 256, kernel_size=3, dilation=18, padding=18, bias=False,stride=1)
        self.block5 = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.dr=nn.Dropout(0.5)
    def forward(self,out):
        out1=self.block1(out)
        out2=self.block2(out)
        out3 = self.block3(out)
        out4 = self.block4(out)
        out5 = self.block5(out)
        out6=self.dr(torch.cat([torch.cat([torch.cat([torch.cat([torch.cat([out1,out],dim=1),out2],dim=1),out3],dim=1),out4],dim=1),out5],dim=1))

        return out6
class conv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(conv, self).__init__()
        self.block=nn.Sequential(
        nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1,padding=1,bias=False),
        nn.Conv2d(out_ch,out_ch,kernel_size=1,stride=1,padding=0,bias=False)
        )
    def forward(self,out):
        out=self.block(out)
        return out
class BR(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(BR, self).__init__()
        self.block=nn.Sequential(
        nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1,padding=1,bias=False),
        nn.ReLU(),
        nn.Conv2d(out_ch,out_ch,kernel_size=3,stride=1,padding=1,bias=False)
        )
    def forward(self,out):
        out=self.block(out)
        return out
def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer

class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x

def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, 1),
        nn.AvgPool2d(2, 2)
    )
    return trans_layer

class DESnet(nn.Module):
    def __init__(self, in_channel, growth_rate=32, block_layers=[6, 12, 24, 16]):
        super(DESnet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
            )
        self.DB1 = self._make_dense_block(64, growth_rate,num=block_layers[0])
        self.TL1 = self._make_transition_layer(256)
        self.DB2 = self._make_dense_block(128, growth_rate, num=block_layers[1])
        self.TL2 = self._make_transition_layer(512)
        self.DB3 = self._make_dense_block(256, growth_rate, num=block_layers[2])
        self.upsample = nn.Upsample(scale_factor=2)
        self.fp1 = FPM(1024)
        self.br1 = BR(3072, 1024)
        self.cs1=scSE(1024,512)
        self.cs2=scSE(512,256)
        self.cs3=scSE(256,64)
        self.cs4=scSE(64,64)
        self.br2=BR(64,2)
    def forward(self, x):
        x = self.block1(x)
        x1 = self.DB1(x)
        x = self.TL1(x1)
        x2 = self.DB2(x)
        x = self.TL2(x2)
        x3 = self.DB3(x)
        x=self.br1(self.fp1(x3))
        x=self.upsample(self.cs1(x))+x2
        x=self.upsample(self.cs2(x))+x1
        x=self.upsample(self.cs3(x))
        x=self.upsample(self.cs4(x))
        x=self.br2(x)
        return x

    def _make_dense_block(self,channels, growth_rate, num):
        block = []
        block.append(dense_block(channels, growth_rate, num))
        channels += num * growth_rate

        return nn.Sequential(*block)
    def _make_transition_layer(self,channels):
        block = []
        block.append(transition(channels, channels // 2))
        return nn.Sequential(*block)
# net = DESnet(3)
# x = torch.rand(1,3,256,256)
# print(net(x).shape)

