import os
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import UltraDataset, Dataset

import torch

from model.colorgan import colorg
from model.remove_model import PM_LSMN
from utils.loss import sam_loss, calculate_ergas
from utils.metrics import SSIM
def calculate_psnr(tensor1, tensor2):
    # 将输入张量转换为浮点数张量
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()
    # 计算 MSE
    mse_val = torch.mean((tensor1 - tensor2) ** 2)

    # 计算 PSNR
    psnr_val = 10 * torch.log10(1 / mse_val)
    return psnr_val.item()

if __name__ == "__main__":
    """ 准备数据 """
    val_dataset = Dataset("data/val1.txt", image_height=512, image_weight=512, image_aug=False)
    # val_dataset = Dataset("data/val1.txt", image_height=512, image_weight=512, image_aug=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)  # , pin_memory=True
    """ 模型载入 """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net=PM_LSMN()
    net = net.to(device)
    log_path = "logs/" + net._get_name() + ' mse_colorloss /'
    # list=os.listdir()
    weight="epoch36_total_loss0.0098_val_loss0.0092.pth"
    net.load_state_dict(torch.load(log_path + weight))

    """ 预测结果保留 """
    savepath = log_path + "eval_out_test/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        os.makedirs(savepath + "predict/")
    mean_ssim=0
    mean_psnr=0
    mean_sam=0
    mean_ergas=0
    pbar = tqdm(enumerate(val_loader), total=len(val_dataset) // val_loader.batch_size)
    for iteration, (imgs, label) in pbar:
        with torch.no_grad():
            # 简单处理
            img_np = imgs.numpy()
            imgs = imgs.to(device).float() / 255.0
            preds = net(imgs)
            label = label.to(device).float() / 255.0
            sam = sam_loss(preds.cpu().numpy().squeeze(),label.cpu().numpy().squeeze())
            ergas = calculate_ergas(preds.cpu().numpy().squeeze(),label.cpu().numpy().squeeze())
            ss_cal=SSIM()
            ssim=ss_cal(preds,label)
            psnr=calculate_psnr(preds,label)
        mean_psnr+=psnr
        mean_ssim+=ssim
        mean_sam+=sam
        mean_ergas+=ergas
        save_image(preds,savepath + "predict/" + "pre_%s.png" %( iteration+1))
    mean_ssim /= iteration + 1
    mean_psnr /= iteration + 1
    mean_sam /= iteration + 1
    mean_ergas /= iteration + 1
    message = """The prediction results with %s: mean_ssim: %.3f | mean_psnr: %.3f | mean_sam: %.3f | mean_ergas: %.3f """ \
              % (weight, mean_ssim,mean_psnr,mean_sam,mean_ergas)
    with open(savepath + "results.txt", 'w') as f:
        f.write(message)