import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model.backbone.vgg16 import VGG16
from torch.utils.tensorboard import SummaryWriter


class Double_CBR(nn.Module):
    """
    Double_CBR 是 Conv BN Relu 两次堆叠
    """

    def __init__(self, in_channel, out_channel, is_pooling=False):
        super(Double_CBR, self).__init__()
        self.is_pooling = is_pooling
        self.CBR2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6()
        )
        if self.is_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        if self.is_pooling:
            x = self.pool(x)
        # print(x.shape)
        x = self.CBR2(x)
        return x


class Unet_Skip_Up(nn.Module):
    """
    Unet_Skip_Up 是 Unet 跳跃链接模块
    """

    def __init__(self, in_channel, out_channel):
        super(Unet_Skip_Up, self).__init__()
        self.CBR2 = Double_CBR(in_channel, out_channel, is_pooling=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, feat_encoder, feat_up):
        feat_up = self.upsample(feat_up)
        output_feat = torch.cat([feat_encoder, feat_up], dim=1)
        output_feat = self.CBR2(output_feat)
        return output_feat


class Unet_Encoder(nn.Module):
    def __init__(self, in_channel=4):
        super(Unet_Encoder, self).__init__()
        self.CBR_512 = Double_CBR(in_channel, 32, is_pooling=False)
        self.CBR_256 = Double_CBR(32, 64, is_pooling=True)
        self.CBR_128 = Double_CBR(64, 128, is_pooling=True)
        self.CBR_64 = Double_CBR(128, 256, is_pooling=True)
        self.CBR_32 = Double_CBR(256, 256, is_pooling=True)

    def forward(self, x):
        feat_512 = self.CBR_512(x)
        feat_256 = self.CBR_256(feat_512)
        feat_128 = self.CBR_128(feat_256)
        feat_64 = self.CBR_64(feat_128)
        feat_32 = self.CBR_32(feat_64)

        return feat_512, feat_256, feat_128, feat_64, feat_32


class Att_block(nn.Module):
    def __init__(self, channel_g, channel_x, channel_mid):
        super(Att_block, self).__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(channel_g, channel_mid, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel_mid)
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(channel_x, channel_mid, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel_mid)
        )
        self.weight = nn.Sequential(
            nn.Conv2d(channel_mid, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, g, x):
        g1 = self.upsample(g)
        g1 = self.Wg(g1)
        x1 = self.Wx(x)
        psi = F.relu6(g1 + x1)
        weight = self.weight(psi)

        return weight * x


class Attention_Unet_Vgg(nn.Module):
    """
    Attention_Unet_Vgg 是在跳跃链接处加入了 注意力门机制, 使用下层特征指导跳跃链接的上层特征, 具体原理参见论文。

    parameter: num_class: 默认二分类
    parameter: in_channels: 默认彩图
    parameter: pretrain: 是否进行迁移学习

    下采样： Maxpooling
    上采样： UpsampleBilinear
    """

    def __init__(self, num_class=2, in_channels=3):
        super(Attention_Unet_Vgg, self).__init__()
        print(self.__doc__)
        self.encoder = Unet_Encoder(4)
        self.skip_up_64 = Unet_Skip_Up(512, 128)
        self.skip_up_128 = Unet_Skip_Up(256, 64)
        self.skip_up_256 = Unet_Skip_Up(128, 32)
        self.skip_up_512 = Unet_Skip_Up(64, 32)
        self.att_block_64 = Att_block(256, 256, 128)
        self.att_block_128 = Att_block(128, 128, 64)
        self.att_block_256 = Att_block(64, 64, 32)
        self.cls_conv = nn.Conv2d(32, num_class, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        feat_512, feat_256, feat_128, feat_64, feat_32 = self.encoder(x)
        feat_64_w = self.att_block_64(feat_32, feat_64)
        up_64 = self.skip_up_64(feat_64_w, feat_32)

        feat_128_w = self.att_block_128(up_64, feat_128)
        up_128 = self.skip_up_128(feat_128_w, up_64)

        feat_256_w = self.att_block_256(up_128, feat_256)
        up_256 = self.skip_up_256(feat_256_w, up_128)

        up_512 = self.skip_up_512(feat_512, up_256)

        results = self.cls_conv(up_512)
        return results

class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1

class Unet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.decode4 = Decoder(512, 256+256, 256)
        self.decode3 = Decoder(256, 256+128, 256)
        self.decode2 = Decoder(256, 128+64, 128)
        self.decode1 = Decoder(128, 64+64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        e1 = self.layer1(input) # 64,128,128
        e2 = self.layer2(e1) # 64,64,64
        e3 = self.layer3(e2) # 128,32,32
        e4 = self.layer4(e3) # 256,16,16
        f = self.layer5(e4) # 512,8,8
        d4 = self.decode4(f, e4) # 256,16,16
        d3 = self.decode3(d4, e3) # 256,32,32
        d2 = self.decode2(d3, e2) # 128,64,64
        d1 = self.decode1(d2, e1) # 64,128,128
        d0 = self.decode0(d1) # 64,256,256
        out = self.conv_last(d0) # 1,256,256
        return out
# if __name__ == "__main__":
#     attention_unet_vgg = Unet(25)
#
#     dummy_input = torch.randn(1, 4, 512, 512)
#     outputs = attention_unet_vgg(dummy_input)
#     print(outputs.shape)

    # writer = SummaryWriter("arch_plot/" + base_unet_vgg_official._get_name())
    # writer.add_graph(base_unet_vgg_official, torch.randn((1, 3, 512, 512)))
    # writer.close()
