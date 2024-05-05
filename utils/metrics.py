import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import cv2
import numpy as np
# from torch.utils.data import DataLoader

# from dataset import GID_Dataset

"""
计算一个batch中的平均指标，有宏平均和微平均两种方式；
"""


def calculate_metrics(pred: Tensor, label: Tensor, num_classes: int = 2):
    intersect_area, pred_area, label_area = calculate_area(pred, label)
    _, acc = accuracy(intersect_area, pred_area)
    _, meaniou = mean_iou(intersect_area, pred_area, label_area)
    _, meandice = mean_dice(intersect_area, pred_area, label_area)
    kappav = kappa(intersect_area, pred_area, label_area)
    return acc, meaniou, meandice, kappav
def calculate_metrics_gid(pred: Tensor, label: Tensor, num_classes: int = 2):
    intersect_area, pred_area, label_area = calculate_area(pred, label)
    _, acc = accuracy(intersect_area, pred_area)
    kappav = kappa(intersect_area, pred_area, label_area)
    return acc, kappav
import torch
import torch.nn.functional as F
from math import exp
import numpy as np


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average).item()

def calculate_area(pred: Tensor, label: Tensor, num_classes: int = 25):
    """
    计算 preds 和 labels 的各类的公共区域，以及各类的区域
    :param pred: N C H W
    :param label: N H W
    :param num_classes: 2
    :return:
    """
    # convert the label to onehot
    label = F.one_hot(label, 2).float()  # N * H * W * C,
    pred = F.softmax(pred, dim=1).float()
    pred = F.one_hot(torch.argmax(pred, dim=1), 2).float()  # N * H * W * C

    pred_area = []
    label_area = []
    intersect_area = []

    for i in range(num_classes):
        pred_i = pred[:, :, :, i]
        label_i = label[:, :, :, i]
        pred_area_i = torch.sum(pred_i).unsqueeze(0)  # 1
        label_area_i = torch.sum(label_i).unsqueeze(0)  # 1
        intersect_area_i = torch.sum(pred_i * label_i).unsqueeze(0)  # 1
        pred_area.append(pred_area_i)
        label_area.append(label_area_i)
        intersect_area.append(intersect_area_i)

    pred_area = torch.cat(pred_area)  # num_classes
    label_area = torch.cat(label_area)  # num_classes
    intersect_area = torch.cat(intersect_area)  # num_classes

    return intersect_area, pred_area, label_area


def mean_dice(intersect_area, pred_area, label_area):
    """
    Calculate dice.

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        np.ndarray: iou on all classes.
        float: mean iou of all classes.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    union = pred_area + label_area
    class_dice = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            dice = 0
        else:
            dice = intersect_area[i] * 2 / union[i]
        class_dice.append(dice)
    mdice = np.mean(class_dice)
    return np.array(class_dice), mdice


def mean_iou(intersect_area, pred_area, label_area):
    """
    Calculate iou.

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        np.ndarray: iou on all classes.
        float: mean iou of all classes.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    union = pred_area + label_area - intersect_area
    class_iou = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            iou = 0
        else:
            iou = intersect_area[i] / union[i]
        class_iou.append(iou)
    miou = np.mean(class_iou)
    return np.array(class_iou), miou


def accuracy(intersect_area, pred_area):
    """
    Calculate accuracy

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes..
        pred_area (Tensor): The prediction area on all classes.

    Returns:
        np.ndarray: accuracy on all classes.
        float: mean accuracy.
    """
    intersect_area = intersect_area.cpu().numpy()
    pred_area = pred_area.cpu().numpy()
    class_acc = []
    for i in range(len(intersect_area)):
        if pred_area[i] == 0:
            acc = 0
        else:
            acc = intersect_area[i] / pred_area[i]
        class_acc.append(acc)
    macc = np.sum(intersect_area) / np.sum(pred_area)
    return np.array(class_acc), macc


def kappa(intersect_area, pred_area, label_area):
    """
    Calculate kappa coefficient

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes..
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        float: kappa coefficient.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    total_area = np.sum(label_area)
    po = np.sum(intersect_area) / total_area
    pe = np.sum(pred_area * label_area) / (total_area * total_area)
    kappav = (po - pe) / (1 - pe)
    return kappav
def segmentation_metrics(ground_truth, prediction):
    """
    计算并返回多个图像分割评估指标。
    :param ground_truth: 真实的分割结果，形状为(1, 512, 512)
    :param prediction: 预测的分割结果，形状为(1, 512, 512)
    :return: 包含Dice系数、Jaccard相似系数、精确率、召回率和F1分数的字典
    """

    if ground_truth.shape != prediction.shape:
        raise ValueError("The shapes of the inputs must be the same")

    # 将输入展平为一维数组以便计算
    gt_flat = ground_truth.flatten()
    pred_flat = prediction.flatten()

    # 计算交集和并集
    intersection = np.logical_and(gt_flat, pred_flat).sum()
    union = np.logical_or(gt_flat, pred_flat).sum()
    true_positives = intersection
    false_positives = union - intersection
    false_negatives = np.sum(gt_flat) - intersection

    # 计算Dice系数
    dice = (2. * intersection) / (union) if union != 0 else 0

    # 计算Jaccard相似系数
    jaccard = intersection / float(union) if union != 0 else 0

    # 计算精确率
    precision = true_positives / float(union) if union != 0 else 0

    # 计算召回率
    recall = true_positives / float(np.sum(gt_flat)) if np.sum(gt_flat) != 0 else 0

    # 计算F1分数
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 返回所有指标的结果
    return  dice, jaccard, precision, recall, f1_score
def decode_segmap(image, nc=25):
    # image = np.concatenate([image[i] for i in range(image.shape[0])], axis=1)
    image=image.squeeze()
    label_colors = np.array([
    [0, 0, 0],
    [200, 0, 0],
    [0, 200, 0],
    [150, 250, 0],
    [150, 200, 150],
    [200, 0, 200],
    [150, 0, 250],
    [150, 150, 250],
    [200, 150, 200],
    [250, 200, 0],
    [200, 200, 0],
    [0, 0, 200],
    [250, 0, 150],
    [0, 150, 200],
    [0, 200, 250],
    [150, 200, 250],
    [250, 250, 250],
    [200, 200, 200],
    [200, 150, 150],
    [250, 200, 150],
    [150, 150, 0],
    [250, 150, 150],
    [250, 150, 0],
    [250, 200, 250],
    [200, 150, 0]
    ])

    # r = np.zeros_like(image).astype(np.uint8)
    # g = np.zeros_like(image).astype(np.uint8)
    # b = np.zeros_like(image).astype(np.uint8)
    #
    # for l in range(0, nc):
    #     idx = image == l
    #     r[idx] = label_colors[l, 0]
    #     g[idx] = label_colors[l, 1]
    #     b[idx] = label_colors[l, 2]
    #
    # rgb = np.stack([r, g, b], axis=2).transpose(2,0,1)

    rgb_list = []
    for i in range(4):
        r = np.zeros_like(image[i]).astype(np.uint8)
        g = np.zeros_like(image[i]).astype(np.uint8)
        b = np.zeros_like(image[i]).astype(np.uint8)

        for l in range(nc):
            idx = image[i] == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2).transpose(2, 0, 1)
        rgb_list.append(rgb)

    rgb = np.array(rgb_list)

    return rgb

# if __name__ == "__main__":
    # # preds = torch.rand((1, 2, 512, 512))
    # # labels = torch.argmax(torch.rand((1, 2, 512, 512)), dim=1)
    # # simple test
    # preds = cv2.imread("../preout/pre_6.png", cv2.IMREAD_GRAYSCALE)
    # preds[preds > 1] = 1
    # labels = cv2.imread("../preout/label_6.png", cv2.IMREAD_GRAYSCALE)
    # labels[labels > 1] = 1
    # preds = torch.tensor(preds).unsqueeze(0).long()
    # preds = F.one_hot(preds, 2).float().permute(0, 3, 1, 2)
    # labels = torch.tensor(labels).unsqueeze(0).long()
    #
    # accuracys, mean_ious, mean_dices, kappas = calculate_metrics(preds, labels)
    # print(mean_ious)
    # print(mean_dices)
    # print(accuracys)
    # print(kappas)