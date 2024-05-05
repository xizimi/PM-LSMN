import torch
import torch.nn as nn
import math


class AOD_net(nn.Module):

    def __init__(self):
        super(AOD_net, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        ############   每个卷积层只用三个核     ##############
        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)  ## 连接1、2层3+3=6，输出3
        self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)  ##连接2，3层3+3=6，输出3
        self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)  # 连接1，2，3，4层3+3+3+3=12，输出3

    def forward(self, x):
        source = []
        source.append(x)
        #########    K-estimation     ###########
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))

        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.e_conv3(concat1))

        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv4(concat2))

        concat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.relu(self.e_conv5(concat3))
        ####### 带入公式  ############
        clean_image = self.relu((x5 * x) - x5 + 1)

        return clean_image
# a=torch.randn(1,3,512,512)
# net=AOD_net()
# print(net(a).shape)