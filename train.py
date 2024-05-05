import datetime
import os
import time
import numpy
import torchvision.models.detection
from torchvision.utils import save_image
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from dataset import  Dataset
import model
import utils
from evaluation import calculate_psnr
from model.remove_model import  PM_LSMN
from utils.loss import sam_loss, calculate_ergas
from utils.metrics import SSIM
if __name__ == "__main__":
    """ 准备数据 """
    size=512
    batchsize=4
    lr=0.00001
    train_dataset = Dataset("data/train1.txt", image_height=size, image_weight=size, image_aug=False)
    val_dataset = Dataset("data/val1.txt", image_height=size, image_weight=size, image_aug=False)

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)  # , pin_memory=True
    val_loader = DataLoader(val_dataset, batch_size=batchsize,shuffle=True)  # , pin_memory=True

    """ 模型载入 """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net=PM_LSMN()
    net = net.to(device)
    epochs = 50
    config = dict(size=size,learingrate = lr, batchsize =batchsize,  architecture = "remove_net",dataset_id = "RICE1",epoch = epochs)
    wan_int=wandb.init(project='tran_project',
               name='remove_net_%s'%(datetime.datetime.now()),
               config=config,
               resume=False)
    """ 超参数 """
    log_path = "logs/" + net._get_name() + ' wandb_text /'
    total_loss = 0
    avg_loss = 0
    """ 日志记录 """
    logger = utils.logger.Log_loss(log_path)
    best_val_loss = 1000
    val_total_loss = 0
    val_avg_loss = 0
    best_PSNR=0
    """ 优化器 """
    loss_all=utils.loss.mse_vgg_color
    optimizer = optim.Adam(params=net.parameters(),
                           lr=0.0001,
                           # momentum=0.949,
                           weight_decay=0.0005,
                           betas=(0.9,0.999)
                           )
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    wan_int.config.update(dict(size=size,epoch=epochs, lr=lr, batch_size=batchsize))
    """ 训练 | 验证 """
    for epoch in range(epochs):
        """ 训练 """
        net.train()
        pbar = tqdm(enumerate(train_loader), desc='Training for %g / %s epoch ' % (epoch + 1, epochs),
                    total=len(train_dataset) // train_loader.batch_size)
        for iteration, (imgs, label) in pbar:
            # 简单处理
            with torch.no_grad():
                imgs = imgs.to(device).float()/255
                label = label.to(device).float()/255
            # 模型推理
            optimizer.zero_grad()
            preds = net(imgs)
            loss = loss_all(preds, label)
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / (iteration + 1)
            lr = optimizer.param_groups[0]['lr']
            train_log = {}
            val_log = {}
            train_log['train_epoch: %d'] = epoch+1
            train_log['train loss: %.3f'] = avg_loss
            train_log['learn rate'] = lr
            traininfo = "Training: epoch: %d | train loss: %.3f | lr: %.3f " \
                        % (epoch + 1, avg_loss, lr)
            pbar.set_description(traininfo)
        # 清零 total_loss
        total_loss = 0
        mean_ssim = 0
        mean_psnr = 0
        """ 验证 """
        net.eval()
        pbar = tqdm(enumerate(val_loader), total=len(val_dataset) // val_loader.batch_size)
        for iteration, (imgs, label) in pbar:
            with torch.no_grad():
                # 简单处理
                imgs = imgs.to(device).float()/255
                label = label.to(device).float()/255
                # 模型推理
                optimizer.zero_grad()
                preds = net(imgs)
                val_loss = loss_all(preds, label)
                val_total_loss += val_loss.item()
                val_avg_loss = val_total_loss / (iteration + 1)
                ss_cal = SSIM()
                ssim = ss_cal(preds, label)
                psnr = calculate_psnr(preds, label)
                mean_psnr += psnr
                mean_ssim += ssim
            # 打印信息
            valinfo = "Evaluating: epoch: %d | val loss: %.3f | lr: %.3f " \
                      % (epoch + 1, val_avg_loss,lr)
            pbar.set_description(valinfo)
            if best_PSNR<=psnr:
                best_PSNR = psnr
                initial_image=imgs.float()
                true_image=label.float()
                pred_image=preds.float()
        wandb.log({'train_loss': avg_loss, 'text_loss': val_avg_loss, 'epoch': epoch, 'learning rate': lr,'PSNR': mean_psnr/(iteration+1),'SSIM': mean_ssim/(iteration+1),
                   'img': {'initial':wandb.Image(initial_image),
                       'true': wandb.Image(true_image),
                     'pred': wandb.Image(pred_image)}}
                  )
        print(mean_psnr/(iteration+1),mean_ssim/(iteration+1))
        val_total_loss = 0
        # 更新学习率
        scheduler.step()
        # 记录日志并绘图
        logger.append_info(traininfo, valinfo)
        logger.append_loss(avg_loss, val_avg_loss)
        """ 保留最优权重 """
        print('finish evaluation')
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            print("Saving state: val loss: %s, in the %s epoch" % (best_val_loss, epoch + 1))
            torch.save(net.state_dict(), log_path + 'epoch%d_total_loss%.4f_val_loss%.4f.pth' %
                       ((epoch + 1), avg_loss, val_avg_loss))
