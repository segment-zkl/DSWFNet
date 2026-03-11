import os
import random
import cv2
import numpy as np
import torch
import warnings
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from utils.LossFunction import combined_loss, compute_road_iou
from torch import nn
from tqdm import tqdm
from pathlib import Path
from dataload.DataForTrain import TrainDatasetFromFolder
from model.DSWFNet import DSWFNet
from Parameter import DEVICE,SET_NAME,sever_root,BATCH_SIZE,PRE_EPOCH,START_EPOCH,END_EPOCH,NUM_WORKER,FILE_NAME, LEARN_RATE, WEIGHT_DECAY


device = DEVICE
warnings.filterwarnings("ignore")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def seed_training(seed):
    random.seed(seed)  # Python 内置随机数
    np.random.seed(seed)  # Numpy 随机数
    torch.manual_seed(seed)  # PyTorch 随机数
    torch.cuda.manual_seed(seed)  # CUDA 设备随机数
    torch.cuda.manual_seed_all(seed)  # 多 GPU 设备随机数
    torch.backends.cudnn.deterministic = True  # 使 CuDNN 计算确定性化
    torch.backends.cudnn.benchmark = True  # 关开启 CuDNN 自动优化，加快训练，降低复现性


####################定义函数####################

def main():
    ####################定义路径和损失####################
    ##########定义训练信息##########
    start_epoch = START_EPOCH  # 开始时的已训练epoch数
    end_epoch = END_EPOCH  # 结束时的已训练epoch数
    pre_epoch = PRE_EPOCH  # 训练开始时的预训练epoch数
    batch_size = BATCH_SIZE  # batch数
    num_workers = NUM_WORKER  # 子进程数
    train_set_name = SET_NAME  # 数据集名字
    file_name = FILE_NAME  # 本代码文件名字
    set_name = SET_NAME
    gap_epoch = 10  # 保存的epoch间隔
    seed_training(2025)

    # 获取当前脚本（train.py）所在的目录
    script_dir = Path(__file__).resolve().parent
    # 获取 Mycode 目录
    project_root = script_dir.parent
    # 生成测试数据集的路径
    train_root = project_root / "dataset" / train_set_name / "train"
    val_root = project_root / "dataset" / train_set_name / "val"
    print("训练数据路径:", train_root)
    print("训练数据路径:", val_root)

    # ---------------------- 网络初始化 ----------------------

    if start_epoch == 0:
        write_mode = 'w'
        if pre_epoch == 0:
            model = DSWFNet()
        else:
            # 构建模型路径
            model_path = os.path.join(sever_root, 'checkpoint', file_name, train_set_name,
                                      f'model_best_train.pth')
            model = DSWFNet()
            # 加载 checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            # 加载 `state_dict`
            model.load_state_dict(checkpoint['model_state_dict'])

    else:
        write_mode = 'a'
        # 构建模型路径
        model_path = os.path.join(sever_root, 'checkpoint', file_name, train_set_name,
                                  f'model_best_train.pth')
        model = DSWFNet()
        # 加载 checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        # 加载 `state_dict`
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    # ---------------------- 网络初始化结束 ----------------------

    # ---------------------- 数据加载 ----------------------
    train_data_set = TrainDatasetFromFolder(train_root, work_mode=1, shuffle=True)
    val_data_set = TrainDatasetFromFolder(val_root, work_mode=0, shuffle=False)

    ##  数据加载器
    train_data_loader = data.DataLoader(train_data_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_data_loader = data.DataLoader(val_data_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    batch_num_per_epoch = len(train_data_loader)  # 一个epoch的batch数

    # ---------------------- 模型保存路径 ----------------------
    epoch_log = open(f'{sever_root}log/{file_name}/{train_set_name}_epoch.txt', write_mode)
    val_log = open(f'{sever_root}log/{file_name}/{train_set_name}_val.txt', write_mode)
    other_log = open(f'{sever_root}log/{file_name}/{train_set_name}_other.txt', 'w')

    other_log.write(f'dataset:[{train_set_name}] total_epoch[{end_epoch}] batch_size[{batch_size}] num_workers[{num_workers}] pre_train[{pre_epoch}]\n')
    other_log.flush()

    if not os.path.exists(f'{sever_root}checkpoint/{file_name}'):
        os.mkdir(f'{sever_root}checkpoint/{file_name}')

    # ---------------------- 损失函数和优化器 ----------------------

    ##########损失函数和优化器##########
    val_loss_best = 10000 # 当前最优 val_loss
    train_loss_best = 10000  # 当前最优 train_loss
    val_loss_up_num = 0  # val_loss 连续上升计数
    train_loss_up_num = 0  # train_loss 连续上升计数
    best_val_iou = 0.0

    # 定义损失函数
    learn_rate = LEARN_RATE
    weight_decay = WEIGHT_DECAY

    optimizer_SFWDNet= optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay,
                             betas=(0.9, 0.99))  # 生成网络优化器
    warmup_scheduler = LinearLR(optimizer_SFWDNet, start_factor=0.1, total_iters=5)
    cosine_scheduler = CosineAnnealingLR(optimizer_SFWDNet, T_max=end_epoch - 5)
    scheduler = SequentialLR(optimizer_SFWDNet, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])

    # 初始化各类损失值
    ####################迭代训练####################
    print('*' * 100)
    bar = tqdm(range(start_epoch + 1, end_epoch + 1))

    train_loss_list = []
    val_loss_list = []
    val_iou_list = []
    for epoch in bar:  # 迭代训练
        # 初始化各项损失
        loss_focal_epoch,  loss_dice_epoch = [0.0] * 2
        train_loss , val_loss_epoch= [0.0] * 2
        model.train()
        act_words = ''
        for batch, batch_data in enumerate(train_data_loader, 1):  # 遍历数据
            ##########图像数据##########
            sat_img, road_img = batch_data[0].to(device, dtype=torch.float32, non_blocking=True), batch_data[1].to(
                device, dtype=torch.float32, non_blocking=True)

            optimizer_SFWDNet.zero_grad()
            predict_img = model(sat_img)  # 道路预测图像

            loss_total_train, loss_dict_train = combined_loss(
                predict_img, road_img,
                lambda_focal=1.0,
                lambda_dice=1.0
            )

            # 反向传播 + 更新权重
            loss_total_train.backward()
            optimizer_SFWDNet.step()

            ########## 记录损失 ##########
            loss_focal_epoch += loss_dict_train['focal']
            loss_dice_epoch += loss_dict_train['dice']
            train_loss += loss_dict_train['total']

            ##########batch输出信息##########
            batch_words = 'batch[{:4d}/{:4d}] loss[1:(Focal:{:.4f}),2:(Dice:{:.4f}),3:(Total:{:.4f})]'.format(
                batch, batch_num_per_epoch,
                loss_dict_train['focal'], loss_dict_train['dice'],
                loss_dict_train['total']
            )
            bar.set_postfix(loss_info=batch_words)

        # 记录当前 train_loss
        current_train_loss = train_loss
        # 判断是否为最优模型
        if current_train_loss < train_loss_best:
            train_loss_best = current_train_loss
            train_loss_up_num = 0
            # 保存模型（最优）
            model_save_path = os.path.join(sever_root, 'checkpoint', file_name, train_set_name,
                                           f'model_train.pth')
            torch.save({
                        'model_state_dict': model.state_dict()
            }, model_save_path)
            
        else:
            train_loss_up_num += 1

        ##########epoch输出信息##########
        current_lr = optimizer_SFWDNet.param_groups[0]['lr']
        epoch_words = (
            f"Epoch [{epoch:3d}] "
            f"LR: {current_lr:.2e} | "
            f"Losses [Focal: {loss_focal_epoch:.4f}, "
            f"Dice: {loss_dice_epoch:.4f}, "
            f"TrainTotal: {train_loss:.4f}]"
        )
        epoch_log.write('\n' + epoch_words)
        epoch_log.flush()

        # 验证损失计算部分
        val_loss_epoch = 0.0
        pred_list = []
        target_list = []

        model.eval()

        with torch.no_grad():
            for val_batch, val_batch_data in enumerate(val_data_loader, 1):
                sat_img = val_batch_data[0].to(device)  # 遥感图像
                road_img = val_batch_data[1].to(device)  # 道路图像
                predict_img = model(sat_img)  # 预测图像
                ##########Loss##########
                loss_total_val, loss_dict_val = combined_loss(
                    predict_img, road_img,
                    lambda_focal=1.0,
                    lambda_dice=1.0
                )
                val_loss_epoch += loss_total_val  #损失越小越好

                # 收集预测与真实图像用于总体IoU计算
                pred_list.extend([p.detach().cpu() for p in predict_img])
                target_list.extend([t.detach().cpu() for t in road_img])

        # 总体IoU计算（所有图像整体TP/FP/FN后计算一次IoU）
        val_iou_epoch = compute_road_iou(pred_list, target_list)

        val_words = f' ValLossSum[{val_loss_epoch:.4f}]  RoadIoU_Total[{val_iou_epoch:.4f}]'
        val_log.write('\n' + val_words)
        val_log.flush()

        current_val_loss = val_loss_epoch
        if current_val_loss < val_loss_best:
            val_loss_best = current_val_loss
            val_loss_up_num = 0
            model_save_path = os.path.join(sever_root, 'checkpoint', file_name, train_set_name,
                                           f'model_val.pth')
            torch.save({
                        'model_state_dict': model.state_dict()
            }, model_save_path)
            epoch_log.write(f"Epoch {epoch}: Saved model_val.pth with ValLoss={current_val_loss:.4f}\n")
        else:
            val_loss_up_num += 1

        if val_iou_epoch > best_val_iou:
            best_val_iou = val_iou_epoch
            model_save_path = os.path.join(sever_root, 'checkpoint', file_name, train_set_name, f'model_best_iou.pth')
            torch.save({
                'model_state_dict': model.state_dict()
            }, model_save_path)
            epoch_log.write(f"Epoch {epoch}: Saved model_best_iou.pth with RoadIoU={val_iou_epoch:.4f}\n")

        scheduler.step()

        ##########保存模型##########
        if epoch % gap_epoch == 0:  # 如果达到保存条件

            model_path = os.path.join(sever_root, 'checkpoint', file_name, train_set_name,
                                        f'model_{epoch}.pth')
            # 确保保存路径存在
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            # 保存生成网络和判别网络的模型
            torch.save({'model_state_dict':model.state_dict()}, model_path)  # 只保存模型参数
            # 可选的日志记录
            epoch_log.write(f"Model saved at epoch {epoch}\n")
            epoch_log.flush()

        train_loss_list.append(
    current_train_loss.detach().cpu().item() if isinstance(current_train_loss, torch.Tensor) else current_train_loss
)
        val_loss_list.append(
    current_val_loss.detach().cpu().item() if isinstance(current_val_loss, torch.Tensor) else current_val_loss
)

        torch.cuda.empty_cache()

        val_iou_list.append(val_iou_epoch)

    bar.close()

if __name__ == '__main__':
    main()
