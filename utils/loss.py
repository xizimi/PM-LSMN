import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision

import cv2


# from utils.metrics import calculate_area, mean_dice, mean_iou, accuracy
def transform(input):
    dict={
    '1': [200, 0, 0],
    '2':
    [0, 200, 0],
    '3':
    [150, 250, 0],
    '4':
    [150, 200, 150],
    '5':
    [200, 0, 200],
    '6':
    [150, 0, 250],
    '7':
    [150, 150, 250],
    '8':
    [200, 150, 200],
    '9':
    [250, 200, 0],
    '10':
    [200, 200, 0],
    '11':
    [0, 0, 200],
    '12':
    [250, 0, 150],
    '13':
    [0, 150, 200],
    '14':
    [0, 200, 250],
    '15':
    [150, 200, 250],
    '16':
    [250, 250, 250],
    '17':
    [200, 200, 200],
    '18':
    [200, 150, 150],
    '19':
    [250, 200, 150],
    '20':
    [150, 150, 0],
    '21':
    [250, 150, 150],
    '22':
    [250, 150, 0],
    '23':
    [250, 200, 250],
    '24':
    [200, 150, 0],
    '0':
    [0, 0, 0],}
    c,w,h=input.shape
    output=np.ones((3,w,h))
    for i in range(0, 25):
        # print(np.argwhere(input==i)[0][1])
        if(len(np.argwhere(input==i))>=1):
            for p in np.argwhere(input==i):
                output[0,p[1],p[2]]=dict['%d'%i][0]
                output[1,p[1],p[2]] = dict['%d' % i][1]
                output[2, p[1],p[2]] = dict['%d' % i][2]
    return output


def ce_loss(pred: Tensor, label: Tensor):
    """
    使用 nn.CrossEntropyLoss 计算交叉熵
    :param pred: N * C * H * W
    :param label: N * H * W
    :return: 交叉熵损失值
    """
    return nn.CrossEntropyLoss()(pred, label) * 2


def calculate_area(pred: Tensor, label: Tensor, num_classes: int = 2):
    """
    计算 preds 和 labels 的各类的公共区域，以及各类的区域
    :param pred: N C H W
    :param label: N H W
    :param num_classes: 2
    :return:
    """
    # convert the label to onehot
    label = F.one_hot(label, 2).permute(0, 3, 1, 2).float()  # N * C * H * W ,
    pred = F.softmax(pred, dim=1).float()  # N * C * H * W
    inter = label * pred

    label_area = torch.sum(label, dim=2)
    label_area = torch.sum(label_area, dim=2)  # N * C
    pred_area = torch.sum(pred, dim=2)
    pred_area = torch.sum(pred_area, dim=2)  # N * C
    intersect_area = torch.sum(inter, dim=2)
    intersect_area = torch.sum(intersect_area, dim=2)  # N * C

    return intersect_area, pred_area, label_area


def dice_loss(pred: Tensor, label: Tensor, eps=1e-5):
    """
    计算dice loss
    :param pred:
    :param label:
    :param eps:
    :return:
    """
    intersect_area, pred_area, label_area = calculate_area(pred, label)
    diceloss = intersect_area * 2 / (label_area + pred_area + eps)
    diceloss = torch.mean(diceloss, dim=1)  # 对类别dice取平均
    diceloss = torch.mean(diceloss, dim=0)  # 对Batch dice取平均
    # diceloss = 1 - diceloss
    diceloss = -torch.log(diceloss)

    return diceloss

def colorgan_loss(pred: Tensor, label: Tensor):
    return color_loss(pred,label)+l_loss(pred,label)
def iou_loss(pred: Tensor, label: Tensor, eps=1e-5):
    """
    计算iou loss
    :param pred:
    :param label:
    :param eps:
    :return:
    """
    intersect_area, pred_area, label_area = calculate_area(pred, label)
    iouloss = intersect_area / (label_area + pred_area - intersect_area + eps)
    iouloss = torch.mean(iouloss, dim=1)  # 对类别dice取平均
    iouloss = torch.mean(iouloss, dim=0)  # 对Batch dice取平均
    # iouloss = 1 - iouloss
    iouloss = -torch.log(iouloss)

    return iouloss

def index(n):
    return torch.tensor([n]).cuda()
def color_loss(x,y):
    r=(torch.index_select(x,1,index(0))+torch.index_select(y,1,index(0)))/2
    R=torch.index_select(x,1,index(0))-torch.index_select(y,1,index(0))
    G=torch.index_select(x,1,index(1))-torch.index_select(y,1,index(1))
    B=torch.index_select(x,1,index(2))-torch.index_select(y,1,index(2))
    loss_r=torch.sqrt(R*R*(2+r/256)+4*G*G+(2+(255-r)/256*B*B))
    return torch.mean(loss_r)
def mse_loss(cloud:Tensor,label:Tensor):
    loss_mse = nn.MSELoss().cuda()
    return loss_mse(cloud, label)
def l_loss(cloud:Tensor,label:Tensor):
    l=nn.L1Loss().cuda()
    return l(cloud, label)
def ce_dice_loss(pred: Tensor, label: Tensor):
    """
    ce + dice 损失
    :param pred:
    :param label:
    :return:
    """
    diceloss = dice_loss(pred, label)
    celoss = ce_loss(pred, label)
    return celoss + diceloss

def Mse_vgg_loss(cloud:Tensor,label:Tensor):

    vgg_based = torchvision.models.vgg19(pretrained=True).cuda()
    vgg_1 = vgg_based.features[0:4]
    vgg_2 = vgg_based.features[0:9]
    vgg_3 = vgg_based.features[0:18]
    loss_mse=nn.MSELoss().cuda()
    chw=cloud.shape[0]*cloud.shape[1]*cloud.shape[2]*cloud.shape[3]
    loss_vgg19=(torch.pow(torch.norm(vgg_1(cloud)-vgg_1(label)),2)+torch.pow(torch.norm(vgg_2(cloud)-vgg_2(label)),2)+torch.pow(torch.norm(vgg_3(cloud)-vgg_3(label)),2))/chw
    return loss_mse(cloud,label)+loss_vgg19
def mse_vgg_color(cloud:Tensor,label:Tensor):
    vgg_based = torchvision.models.vgg19(pretrained=True).cuda()
    vgg_1 = vgg_based.features[0:4]
    vgg_2 = vgg_based.features[0:9]
    vgg_3 = vgg_based.features[0:18]
    loss_mse = nn.MSELoss().cuda()
    chw = cloud.shape[0] * cloud.shape[1] * cloud.shape[2] * cloud.shape[3]
    loss_vgg19 = (torch.pow(torch.norm(vgg_1(cloud) - vgg_1(label)), 2) + torch.pow(torch.norm(vgg_2(cloud) - vgg_2(label)), 2) + torch.pow(torch.norm(vgg_3(cloud) - vgg_3(label)), 2)) / chw
    loss1=loss_mse(cloud, label)
    loss2=loss_vgg19
    loss3=color_loss(cloud,label)
    return loss1+loss2/(loss2/loss1).detach()+loss3/(loss3/loss1).detach()
# def vgg_color(cloud:Tensor,label:Tensor):
#     vgg_based = torchvision.models.vgg19(pretrained=True).cuda()
#     vgg_1 = vgg_based.features[0:4]
#     vgg_2 = vgg_based.features[0:9]
#     vgg_3 = vgg_based.features[0:18]
#     loss_mse = nn.MSELoss().cuda()
#     chw = cloud.shape[0] * cloud.shape[1] * cloud.shape[2] * cloud.shape[3]
#     loss_vgg19 = (torch.pow(torch.norm(vgg_1(cloud) - vgg_1(label)), 2) + torch.pow(torch.norm(vgg_2(cloud) - vgg_2(label)), 2) + torch.pow(torch.norm(vgg_3(cloud) - vgg_3(label)), 2)) / chw
#     loss1=loss_mse(cloud, label)
#     loss2=loss_vgg19
#     loss3=color_loss(cloud,label)
#     return loss1+loss2/(loss2/loss1).detach()+loss3/(loss3/loss1).detach()
def mse_color(cloud:Tensor,label:Tensor):
    # vgg_based = torchvision.models.vgg19(pretrained=True).cuda()
    # vgg_1 = vgg_based.features[0:4]
    # vgg_2 = vgg_based.features[0:9]
    # vgg_3 = vgg_based.features[0:18]
    loss_mse = nn.MSELoss().cuda()
    # chw = cloud.shape[0] * cloud.shape[1] * cloud.shape[2] * cloud.shape[3]
    #loss_vgg19 = (torch.pow(torch.norm(vgg_1(cloud) - vgg_1(label)), 2) + torch.pow(torch.norm(vgg_2(cloud) - vgg_2(label)), 2) + torch.pow(torch.norm(vgg_3(cloud) - vgg_3(label)), 2)) / chw
    loss1=loss_mse(cloud, label)
    loss3=color_loss(cloud,label)
    return loss1+loss3/(loss3/loss1).detach()
def mse_vgg(cloud:Tensor,label:Tensor):
    vgg_based = torchvision.models.vgg19(pretrained=True).cuda()
    vgg_1 = vgg_based.features[0:4]
    vgg_2 = vgg_based.features[0:9]
    vgg_3 = vgg_based.features[0:18]
    loss_mse = nn.MSELoss().cuda()
    chw = cloud.shape[0] * cloud.shape[1] * cloud.shape[2] * cloud.shape[3]
    loss_vgg19 = (torch.pow(torch.norm(vgg_1(cloud) - vgg_1(label)), 2) + torch.pow(torch.norm(vgg_2(cloud) - vgg_2(label)), 2) + torch.pow(torch.norm(vgg_3(cloud) - vgg_3(label)), 2)) / chw
    loss1=loss_mse(cloud, label)
    loss2=loss_vgg19
    return loss1+loss2/(loss2/loss1).detach()
def thick_loss(cloud:Tensor,label:Tensor):
    # print(torch.norm((cloud-label),p=1))
    return torch.sqrt(torch.square(torch.norm((cloud-label),p=1)/(cloud.shape[0]*512*512*3))+torch.Tensor([[[[1e-6]]]]).cuda())
def ce_dice_iou_loss(pred: Tensor, label: Tensor):
    """
    ce + dice + iou 损失
    :param pred:
    :param label:
    :return:
    """
    diceloss = dice_loss(pred, label)
    celoss = ce_loss(pred, label)
    iouloss = iou_loss(pred, label)
    return celoss + diceloss + iouloss
def L2_loss(pred: Tensor, label: Tensor):
    x=torch.mean(torch.norm(pred-label,p=2))
    return x
class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()
        #一个正则项，添加到分母和分子中，以防止除以零或对接近 1 的值取反余弦函数。
        self.eps = 2.2204e-16

    def forward(self, im1, im2):
        assert im1.shape == im2.shape
        H, W, C = im1.shape
        im1 = np.reshape(im1, (H * W, C))
        im2 = np.reshape(im2, (H * W, C))
        core = np.multiply(im1, im2)
        mole = np.sum(core, axis=1)
        im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
        im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
        deno = np.multiply(im1_norm, im2_norm)
        sam = np.rad2deg(np.arccos(((mole + self.eps) / (deno + self.eps)).clip(-1, 1)))
        return np.mean(sam)
def sam_loss(pred, label):
    sam=Loss_SAM()
    return sam(pred,label)
# 定义一个函数来计算ERGAS
def calculate_ergas(reference_image, processed_image):
    # 获取图像的尺寸
    height, width, num_bands = reference_image.shape

    # 初始化变量用于计算各个波段的MSE和RMSE
    mse_values = []
    rmse_values = []

    for band in range(num_bands):
        # 计算MSE（均方误差）
        mse = np.mean((reference_image[:, :, band] - processed_image[:, :, band]) ** 2)
        mse_values.append(mse)

        # 计算RMSE（均方根误差）
        rmse = np.sqrt(mse)
        rmse_values.append(rmse)

    # 计算每个波段的平均亮度
    average_brightness = [np.mean(reference_image[:, :, band]) for band in range(num_bands)]

    # 定义常数参数
    N = num_bands
    L = 256  # 假设灰度级数为256

    # 计算ERGAS
    ergas_values = []
    for mse, rmse, Y in zip(mse_values, rmse_values, average_brightness):
        ergas_values.append((100 / L) * np.sqrt(1 / mse) * (rmse / Y) ** 2)

    ergas = np.sqrt((1 / N) * np.sum(ergas_values))

    return ergas
# if __name__ == "__main__":
#     a=np.array([[[0,1,7],[5,3,9],[13,21,22]]])
#     print(transform(a))
    # b=np.ones((3,3,3))
    # c={'0':[5,5,5]}
    # for i in range(0,1):
    #     b[np.argwhere(a==i)]=c['%d'%i]
    #     print(b)
        # print(np.argwhere(a==i))
    # print(a.shape)