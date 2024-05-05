# from keras.layers import  Convolution2D, MaxPooling2D, UpSampling2D
import torch
from torch import nn
# def cloud_segnet(x):
#     x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x) #nb_filter, nb_row, nb_col
#     x = MaxPooling2D((2, 2), border_mode='same')(x)
#     x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
#     x = MaxPooling2D((2, 2), border_mode='same')(x)
#     x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
#     encoded = MaxPooling2D((2, 2), border_mode='same')(x)
#     x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
#     x = UpSampling2D((2, 2))(x)
#     x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
#     x = UpSampling2D((2, 2))(x)
#     x = Convolution2D(16, 3, 3, activation='relu', border_mode='valid')(x)
#     x = UpSampling2D((2, 2))(x)
#     decoded = Convolution2D(1, 5, 5, activation='sigmoid', border_mode='same')(x)
#     return decoded
# a=torch.randn(1,3,256,256)
# print(cloud_segnet(a))
class cloud_segnet(nn.Module):
    def __init__(self):
        super(cloud_segnet, self).__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,bias=False,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 8, kernel_size=3, bias=False,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 8, kernel_size=3, bias=False,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder=nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, bias=False,padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 8, kernel_size=3, bias=False,padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 16, kernel_size=3, bias=False,padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16,2,kernel_size=3,bias=False,padding=1)
        )
    def forward(self,s):
        return self.decoder(self.encoder(s))
# a=torch.randn(1,3,300,300)
# b=cloud_segnet()
# print(b(a).shape)