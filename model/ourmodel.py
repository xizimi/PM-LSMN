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
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse

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
        self.se=scSE(out_ch)
    def forward(self,x):
        x1=self.conv1(x)
        x2=self.conv2(x)
        x3=self.conv3(x)
        x4=self.conv4(x)
        return self.se(self.dr(x1+x2+x3+x4))
# class PPM(nn.Module):
#     def __init__(self, in_dim, out_dim, bins):
#         super(PPM, self).__init__()
#         self.features = []
#         for bin in bins:
#             self.features.append(nn.Sequential(
#                 nn.AdaptiveAvgPool2d(bin),
#                 nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_dim),
#                 nn.ReLU(inplace=True)
#             ))
#         self.features = nn.ModuleList(self.features)
#         self.block=nn.Sequential(
#             nn.Conv2d(in_dim+4*out_dim,in_dim,kernel_size=3,stride=1,padding=1,bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, bias=False)
#         )
#     def forward(self, x):
#         x_size = x.size()
#         out = [x]
#         for f in self.features:
#             temp = f(x)
#             temp = F.interpolate(temp, x_size[2:], mode="bilinear", align_corners=True)
#             out.append(temp)
#         out=torch.cat(out,1)
#         return
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
class muti_net(nn.Module):
    def __init__(self):
        super(muti_net, self).__init__()
        self.block1=nn.Sequential(
        nn.Conv2d(3,64,kernel_size=3,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block2_1=nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
        )
        self.block2_2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
        )
        self.block3_1=nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
        )
        self.block3_2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
        )
        self.block4_1 = nn.Sequential(
            nn.Conv2d(512,256, kernel_size=1, stride=1, padding=0, bias=False,dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, bias=False,dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False,dilation=2),
            nn.BatchNorm2d(1024),
        )
        self.block4_2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False,dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, bias=False,dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False,dilation=2),
            nn.BatchNorm2d(1024),
        )
        self.block5_1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False,dilation=4),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=4, bias=False,dilation=4),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False,dilation=4),
            nn.BatchNorm2d(2048),
        )
        self.block5_2 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False,dilation=4),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=4, bias=False,dilation=4),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False,dilation=4),
            nn.BatchNorm2d(2048),
        )
        # self.block6=nn.Sequential(
        #     nn.Conv2d(3072,2048,kernel_size=1,stride=1,bias=False),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU()
        # )
        self.fp3=FPM(2048)
        # self.conv3=conv(5120,256)
        # self.br3=BR(256,256)
        # self.fp2 = FPM(512)
        # self.conv2 = conv(2048, 256)
        # self.br2 = BR(256, 256)
        # self.fp1 = FPM(256)
        # self.conv1 = conv(1536, 128)
        self.br1 = BR(5120, 2048)
        # self.br=BR(128,64)
        self.downsample1=nn.Conv2d(64,256,kernel_size=1,stride=1,bias=False)
        self.downsample2=nn.Conv2d(512,1024,kernel_size=1,stride=1,bias=False)
        self.downsample3 = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False)
        # self.upsample1=upsampling(256,128)
        self.upsample =nn.Upsample(scale_factor=2)
        self.re=nn.ReLU()
        # self.upsample2=upsampling(128,64)
        self.br5 = BR(64,2)
        self.br6=BR(64,64)
        self.cmp1=CMP(2048,1024)
        self.cmp2=CMP(1024,512)
        self.cmp3=CMP(512,256)
        self.cmp4=CMP(256,64)
    def forward(self,out):
        out0=self.block1(out)
        out1=self.maxpool(out0)
        out2=self.re(self.block2_1(out1)+self.downsample1(out1))
        for i in range(0,2):
            out2=self.re(out2+self.block2_2(out2))
        out3=self.re(self.block3_1(out2))
        for p in range(0,3):
            out3=self.re(out3+self.block3_2(out3))
        out4=self.re(self.downsample2(out3)+self.block4_1(out3))
        for l in range(0,5):
            out4=self.re(out4+self.block4_2(out4))
        out = self.re(self.downsample3(out4) + self.block5_1(out4))
        for s in range(0, 2):
            out = self.re(out + self.block5_2(out))
        out=self.br1(self.fp3(out))+out
        out=self.br6(self.upsample(self.cmp4(self.upsample(self.cmp3(self.cmp2(self.cmp1(out)+out4)+out3))+out2)+out1))
        out=self.br5(self.upsample(out+out0))
        # out5=torch.cat([out4,out],dim=1)
        # out5=self.br3(self.conv3(self.fp3(self.block6(out5))))
        # out3=self.br2(self.conv2(self.fp2(out3)))
        # out3=self.br6(out5+out3)
        # out3=self.upsample1(out3)
        # out2=self.br(out3+self.br1(self.conv1(self.fp1(out2))))
        # out2=self.br4(self.upsample2(out2))
        # out=self.br5(self.upsample3(out2))
        return out
# net=muti_net()
# a=torch.randn(1,3,512,512)
# print(net(a).shape)