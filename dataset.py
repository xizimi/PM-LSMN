import torch
from skimage import io
import os
import cv2
import matplotlib.pyplot as plt
import numpy
from torch import from_numpy
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import numpy as np

from utils.metrics import segmentation_metrics, decode_segmap


def convert_label_to_segmentation_matrix(label_image):
    """
    将单通道的标签图像转换为多通道的语义分割矩阵。

    参数:
    - label_image: 一个形状为(512, 512)的二维NumPy数组，代表标签图像。
    其中元素值的范围是0-24，对应25个不同的类别。

    返回:
    - segmentation_matrix: 一个形状为(25, 512, 512)的三维NumPy数组，代表转换后的
    语义分割矩阵，每个类别对应一个通道。
    """
    # 确保输入图像的值在0-24之间

    # 初始化一个25x512x512的三维数组，用于存储语义分割矩阵
    segmentation_matrix = np.zeros((25, 512, 512), dtype=np.uint8)

    # 遍历原始图像中的每个像素
    for i in range(label_image.shape[0]):
        for j in range(label_image.shape[1]):
            # 根据像素值填充到对应的通道
            segmentation_matrix[label_image[i, j], i, j] = 1

    return segmentation_matrix
class UltraDataset(Dataset):
    def __init__(self, txt_path, image_height=512, image_weight=512, image_aug=False):
        super(UltraDataset, self).__init__()

        assert os.path.exists(txt_path), "%s 路径有问题！" % txt_path
        with open(txt_path, 'r') as f:
            self.data_list = f.readlines()

        self.image_height = image_height
        self.image_weight = image_weight
        self.image_aug = image_aug

        # 检查txt中的文件是否都存在,可选

    def __getitem__(self, index) -> Tensor:
        image_path, mask_path = self.data_list[index].split()
        # image_path=self.data_list[index]
        # print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        if self.image_aug:
            pass
        image = image.transpose(2, 0, 1)
        #
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        # mask[mask > 1] = 1  # 针对二分类
        mask = from_numpy(mask)
        # mask = mask.unsqueeze(0)

        return from_numpy(image), mask

    def __len__(self):
        return len(self.data_list)

class Dataset3(Dataset):
    def __init__(self, txt_path, image_height=512, image_weight=512, image_aug=False):
        super(Dataset3, self).__init__()

        assert os.path.exists(txt_path), "%s 路径有问题！" % txt_path
        with open(txt_path, 'r') as f:
            self.data_list = f.readlines()

        self.image_height = image_height
        self.image_weight = image_weight
        self.image_aug = image_aug

        # 检查txt中的文件是否都存在,可选

    def __getitem__(self, index) -> Tensor:
        image_path, mask_path = self.data_list[index].split()
        # image_path=self.data_list[index]
        # print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        if self.image_aug:
            pass
        image = image.transpose(2, 0, 1)
        #
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (self.image_weight*2, self.image_height*2), interpolation=cv2.INTER_LINEAR)
        # mask[mask > 1] = 1  # 针对二分类
        mask = mask.transpose(2, 0, 1)
        mask = from_numpy(mask)
        # mask = mask.unsqueeze(0)

        return from_numpy(image), mask

    def __len__(self):
        return len(self.data_list)
class Dataset(Dataset):
    def __init__(self, txt_path, image_height=512, image_weight=512, image_aug=False):
        super(Dataset, self).__init__()

        assert os.path.exists(txt_path), "%s 路径有问题！" % txt_path
        with open(txt_path, 'r') as f:
            self.data_list = f.readlines()

        self.image_height = image_height
        self.image_weight = image_weight
        self.image_aug = image_aug

        # 检查txt中的文件是否都存在,可选

    def __getitem__(self, index) -> Tensor:
        image_path, mask_path = self.data_list[index].split()
        # image_path=self.data_list[index]
        # print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        if self.image_aug:
            pass
        image = image.transpose(2, 0, 1)
        #
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        # mask[mask > 1] = 1  # 针对二分类
        mask = mask.transpose(2, 0, 1)
        mask = from_numpy(mask)
        # mask = mask.unsqueeze(0)

        return from_numpy(image), mask

    def __len__(self):
        return len(self.data_list)

class Dataset_thick(Dataset):
    def __init__(self, txt_path, image_height=512, image_weight=512, image_aug=False):
        super(Dataset, self).__init__()

        assert os.path.exists(txt_path), "%s 路径有问题！" % txt_path
        with open(txt_path, 'r') as f:
            self.data_list = f.readlines()

        self.image_height = image_height
        self.image_weight = image_weight
        self.image_aug = image_aug

        # 检查txt中的文件是否都存在,可选

    def __getitem__(self, index) -> Tensor:
        image_path, mask_path,mask_thickpath = self.data_list[index].split()
        # image_path=self.data_list[index]
        # print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        if self.image_aug:
            pass
        image = image.transpose(2, 0, 1)
        #
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        # mask[mask > 1] = 1  # 针对二分类
        mask = mask.transpose(2, 0, 1)
        mask = from_numpy(mask)
        # mask = mask.unsqueeze(0)
        mask_thick=cv2.imread(mask_thickpath, cv2.IMREAD_COLOR)
        mask_thick = cv2.resize(mask_thick, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        mask_thick = mask_thick.transpose(2, 0, 1)
        mask_thick = from_numpy(mask_thick)
        return from_numpy(image), mask,mask_thick

    def __len__(self):
        return len(self.data_list)

class Dataset_thick_one(Dataset):
    def __init__(self, txt_path, image_height=512, image_weight=512, image_aug=False):
        super(Dataset, self).__init__()

        assert os.path.exists(txt_path), "%s 路径有问题！" % txt_path
        with open(txt_path, 'r') as f:
            self.data_list = f.readlines()

        self.image_height = image_height
        self.image_weight = image_weight
        self.image_aug = image_aug

        # 检查txt中的文件是否都存在,可选

    def __getitem__(self, index) -> Tensor:
        image_path, mask_path, mask_thickpath = self.data_list[index].split()
        # image_path=self.data_list[index]
        # print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        image = numpy.array(image).reshape(self.image_weight, self.image_height,1)
        if self.image_aug:
            pass
        image = image.transpose(2, 0, 1)
        #
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        mask = numpy.array(mask).reshape(self.image_weight, self.image_height, 1)

        # mask[mask > 1] = 1  # 针对二分类
        mask = mask.transpose(2, 0, 1)
        mask = from_numpy(mask)
        # mask = mask.unsqueeze(0)
        mask_thick = cv2.imread(mask_thickpath, cv2.IMREAD_GRAYSCALE)
        mask_thick = cv2.resize(mask_thick, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        mask_thick = numpy.array(mask_thick).reshape(self.image_weight, self.image_height, 1)
        mask_thick = mask_thick.transpose(2, 0, 1)
        mask_thick = from_numpy(mask_thick)
        return from_numpy(image), mask, mask_thick
    def __len__(self):
        return len(self.data_list)


class Dataset2(Dataset):
    def __init__(self, txt_path, image_height=512, image_weight=512, image_aug=False):
        super(Dataset, self).__init__()

        assert os.path.exists(txt_path), "%s 路径有问题！" % txt_path
        with open(txt_path, 'r') as f:
            self.data_list = f.readlines()

        self.image_height = image_height
        self.image_weight = image_weight
        self.image_aug = image_aug

        # 检查txt中的文件是否都存在,可选

    def __getitem__(self, index) -> Tensor:
        image1_path,image2_path, mask_path = self.data_list[index].split('',3)
        # image_path=self.data_list[index]
        # print(image_path)
        image = cv2.imread(image1_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        if self.image_aug:
            pass
        image = image.transpose(2, 0, 1)
        image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = cv2.resize(image2, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        if self.image_aug:
            pass
        image2 = image2.transpose(2, 0, 1)
        #
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        # mask[mask > 1] = 1  # 针对二分类
        mask = mask.transpose(2, 0, 1)
        mask = from_numpy(mask)
        # mask = mask.unsqueeze(0)

        return from_numpy(image),from_numpy(image2), mask

    def __len__(self):
        return len(self.data_list)

class UltraDataset_ds(Dataset):
    """
    用于 深监督训练 的data generator
    """
    def __init__(self, txt_path, image_height=512, image_weight=512, image_aug=False):
        super(UltraDataset_ds, self).__init__()

        assert os.path.exists(txt_path), "%s 路径有问题！" % txt_path
        with open(txt_path, 'r') as f:
            self.data_list = f.readlines()

        self.image_height = image_height
        self.image_weight = image_weight
        self.image_aug = image_aug

        # 检查txt中的文件是否都存在,可选

    def __getitem__(self, index) -> Tensor:
        image_path, mask_path = self.data_list[index].split()

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        if self.image_aug:
            pass
        image = image.transpose(2, 0, 1)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask_512 = cv2.resize(mask, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        mask_512[mask_512 > 1] = 1  # 针对二分类

        mask_256 = cv2.resize(mask, (self.image_weight//2, self.image_height//2), interpolation=cv2.INTER_LINEAR)
        mask_256[mask_256 > 1] = 1  # 针对二分类

        mask_128 = cv2.resize(mask, (self.image_weight//4, self.image_height//4), interpolation=cv2.INTER_LINEAR)
        mask_128[mask_128 > 1] = 1  # 针对二分类

        return from_numpy(image), from_numpy(mask_512), from_numpy(mask_256), from_numpy(mask_128)

    def __len__(self):
        return len(self.data_list)

class text_dataset(Dataset):
    def __init__(self, txt_path, image_height=512, image_weight=512, image_aug=False):
        super(Dataset, self).__init__()
        self.image_height = image_height
        self.image_weight = image_weight
        self.image_aug = image_aug
        self.dir=txt_path
        # 检查txt中的文件是否都存在,可选

    def __getitem__(self, index) -> Tensor:

        image = cv2.imread(self.dir, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        if self.image_aug:
            pass
        image = image.transpose(2, 0, 1)

        return from_numpy(image)

    def __len__(self):
        return 1

class GID_Dataset(Dataset):
    def __init__(self, txt_path, image_height=512, image_weight=512, image_aug=False):
        super(Dataset, self).__init__()

        assert os.path.exists(txt_path), "%s 路径有问题！" % txt_path
        with open(txt_path, 'r') as f:
            self.data_list = f.readlines()

        self.image_height = image_height
        self.image_weight = image_weight
        self.image_aug = image_aug

        # 检查txt中的文件是否都存在,可选

    def __getitem__(self, index) -> Tensor:
        image_path, mask_path = self.data_list[index].split()
        image = io.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.image_aug:
            pass
        image = image.transpose(2, 0, 1)
        #
        label = cv2.imread(mask_path, -1)
        label=convert_label_to_segmentation_matrix(label)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # mask = cv2.resize(mask, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        # mask[mask > 1] = 1  # 针对二分类
        # mask = mask.transpose(2, 0, 1)
        mask = from_numpy(label)
        # mask = mask.unsqueeze(0)

        return from_numpy(image), mask

    def __len__(self):
        return len(self.data_list)
if __name__ == "__main__":
    # data = UltraDataset_ds("./data/train.txt")
    data = GID_Dataset("./data/val_gid.txt")
    dataloader = DataLoader(data,batch_size=4)
    dataiter = iter(dataloader)
    # img, label = data.__getitem__(50)
    img,label = dataiter.next()

    print(img.shape, label.shape)

    # plt.subplot(1, 2, 1)
    # plt.imshow(img.permute(1, 2, 0).numpy(), 'gray')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2)
    # plt.imshow(label.numpy() * 255, 'gray')
    # plt.xticks([]), plt.yticks([])
    # plt.show()