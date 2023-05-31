import os
import torch
import shutil
import datetime
import argparse
import numpy as np

from tqdm import tqdm
from typing import Tuple
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from nets.ema import EMA
from nets.yolo import YOLO
from nets.loss import Loss
from utils.log_loss import LossLog
from utils.log_eval import EvalLog
from utils.dataset import YoloDataset
from utils.dataloader import YoloDataLoader
from utils.util_box import non_max_suppression
from utils.util import read_txt, get_classes, show_info, weight_init
from utils.util_lr import get_lr_scheduler, set_optimizer_lr, get_lr


def train(
        cfg, model_train: nn.Module, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        optimizer: optim.Optimizer, ema: EMA, loss: Loss, loss_log: LossLog, eval_log: EvalLog,
        epoch: int, epoch_step_train: int, epoch_step_val: int
):
    train_loss, train_cls, train_box, train_dfl = 0, 0, 0, 0
    val_loss, val_cls, val_box, val_dfl = 0, 0, 0, 0

    print('\nStart an Epoch Train......', flush=True)
    progress_bar = tqdm(desc=f'Epoch {epoch + 1}/{cfg.epoch}', total=epoch_step_train, mininterval=0.3, postfix=dict)
    model_train.train()
    for step, batch in enumerate(train_loader):
        if step >= epoch_step_train:
            break
        images, boxes = batch
        with torch.no_grad():
            images, boxes = images.cuda(), boxes.cuda()
        optimizer.zero_grad()  # 清零梯度
        outputs = model_train(images)  # 前向传播
        loss_step, loss_item = loss(outputs, boxes)  # loss_item(cls, box, dfl)
        loss_step.backward()  # 反向传播
        optimizer.step()
        if ema:
            ema.update(model=model_train)
        train_loss += loss_item.sum().item()
        train_cls += loss_item[0].item()
        train_box += loss_item[1].item()
        train_dfl += loss_item[2].item()
        progress_bar.set_postfix(ordered_dict={
            'loss': train_loss / (step + 1), 'lr': get_lr(optimizer=optimizer),
            'cls': train_cls / (step + 1), 'box': train_box / (step + 1), 'dfl': train_dfl / (step + 1)
        })
        progress_bar.update(1)
    progress_bar.close()
    print('Finished the Epoch Train.')

    print('Start Model Validation......', flush=True)
    model_train_eval = ema.ema if ema else model_train.eval()
    progress_bar = tqdm(desc=f'Epoch {epoch + 1}/{cfg.epoch}', total=epoch_step_val, mininterval=0.3, postfix=dict)
    for step, batch in enumerate(val_loader):
        if step >= epoch_step_val:
            break
        images, boxes = batch
        with torch.no_grad():
            images, boxes = images.cuda(), boxes.cuda()
            optimizer.zero_grad()  # 清零梯度
            eval_outputs, val_outputs = model_train_eval(images)  # 前向传播
            _, loss_item = loss(val_outputs, boxes)  # loss_item(cls, box, dfl)
        val_loss += loss_item.sum().item()
        val_cls += loss_item[0].item()
        val_box += loss_item[1].item()
        val_dfl += loss_item[2].item()
        progress_bar.set_postfix(ordered_dict={
            'val_loss': val_loss / (step + 1),
            'cls': val_cls / (step + 1), 'box': val_box / (step + 1), 'dfl': val_dfl / (step + 1)
        })
        progress_bar.update(1)
        if cfg.eval_flag and (epoch + 1) % cfg.eval_period == 0:
            outputs = non_max_suppression(outputs=eval_outputs, conf_thres=cfg.conf_thres, iou_thres=cfg.iou_thres)
            eval_log.log_eval_txt(pred=outputs, gt=boxes, step=step, batch_size=cfg.batch_size)
    progress_bar.close()
    print('Finished Model Validation.')

    loss_log.log_loss(epoch=epoch + 1, train_loss=train_loss / epoch_step_train, val_loss=val_loss / epoch_step_val)
    print('Epoch: ' + str(epoch + 1) + '/' + str(cfg.epoch))
    print('Train Loss: %.5f || Val Loss: %.5f' % (train_loss / epoch_step_train, val_loss / epoch_step_val))
    if cfg.eval_flag and (epoch + 1) % cfg.eval_period == 0:
        print("Calculate map......")
        eval_log.log_eval_map(epoch=epoch + 1)

    save_state_dict = ema.ema.state_dict() if ema else model.state_dict()  # 保存权值
    if (epoch + 1) % cfg.save_period == 0 or epoch + 1 == cfg.epoch:
        torch.save(save_state_dict, os.path.join(cfg.save_dir, 'epoch%03d-train_loss%.3f-val_loss%.3f.pt' % (
            epoch + 1, train_loss / epoch_step_train, val_loss / epoch_step_val
        )))
    if val_loss / epoch_step_val <= min(loss_log.val_losses):
        print('Save best model to best_epoch_weights.pth')
        torch.save(save_state_dict, os.path.join(cfg.save_dir, 'best_epoch_weights.pt'))
    torch.save(save_state_dict, os.path.join(cfg.save_dir, 'last_epoch_weights.pt'))


def main(cfg):
    class_names, num_classes = get_classes(cfg.classes_path)  # 获取类别
    train_datas, val_datas = read_txt(path=cfg.train_path), read_txt(path=cfg.val_path)  # 读取数据集对应的txt
    num_train, num_val = len(train_datas), len(val_datas)
    show_info(
        init_epoch=cfg.init_epoch, epoch=cfg.epoch,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        model_path=cfg.model_path, classes_path=cfg.classes_path,
        train_path=cfg.train_path, val_path=cfg.val_path,
        input_shape=cfg.input_shape, stride=cfg.stride,
        num_train=num_train, num_val=num_val,
        class_names=class_names, num_classes=num_classes,
        save_dir=cfg.save_dir, save_period=cfg.save_period,
        eval_flag=cfg.eval_flag, eval_period=cfg.eval_period,
        mosaic=cfg.mosaic, mosaic_prob=cfg.mosaic_prob,
        mixup=cfg.mixup, mixup_prob=cfg.mixup_prob,
        special_aug_ratio=cfg.special_aug_ratio,
        init_lr=cfg.init_lr, min_lr=cfg.min_lr,
        momentum=cfg.momentum, weight_decay=cfg.weight_decay,
        optimizer_type=cfg.optimizer_type, lr_decay_type=cfg.lr_decay_type,
        cls_gain=cfg.cls_gain, box_gain=cfg.box_gain, dfl_gain=cfg.dfl_gain,
        conf_thres=cfg.conf_thres, iou_thres=cfg.iou_thres
    )

    epoch_step_train, epoch_step_val = num_train // cfg.batch_size, num_val // cfg.batch_size
    if epoch_step_train == 0 or epoch_step_val == 0:  # 判断每一个世代的长度
        raise ValueError('数据集过小，无法继续进行训练，请扩充数据集。')
    wanted_step, total_step = 5e4 if cfg.optimizer_type == 'sgd' else 1.5e4, num_train // cfg.batch_size * cfg.epoch
    if total_step < wanted_step:
        wanted_epoch = wanted_step // (num_train // cfg.batch_size) + 1
        print('\n\033[1;33;44m[Warning]使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m'
              % (cfg.optimizer_type, wanted_step))
        print('\033[1;33;44m[Warning]本次运行总训练数据量为%d，batch_size为%d，共训练%d个Epoch，总训练步长为%d。\033[0m'
              % (num_train, cfg.batch_size, cfg.epoch, total_step))
        print('\033[1;33;44m[Warning]由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m'
              % (total_step, wanted_step, wanted_epoch))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置训练设备
    model = YOLO(phi=cfg.phi, num_classes=num_classes, stride=cfg.stride)  # 创建模型
    weight_init(model=model)
    if cfg.model_path != '':
        print('Load weight %s' % cfg.model_path)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(cfg.model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v  # 根据预训练权重的Key和模型的Key进行加载
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(state_dict=model_dict)
        print('Successful Load Key: ', str(load_key)[: 200], '……\nSuccessful Load Key Num: ', len(load_key))
        print('Fail To Load Key: ', str(no_load_key)[: 200], '……\nFail To Load Key num: ', len(no_load_key))
        print('\033[1;33;44m温馨提示: head部分没有载入是正常现象，backbone部分没有载入是错误的。\033[0m')

    train_dataset = YoloDataset(
        data=train_datas, input_shape=cfg.input_shape, num_classes=num_classes, epoch_length=cfg.epoch,
        mosaic=cfg.mosaic, mosaic_prob=cfg.mosaic_prob, mixup=cfg.mixup, mixup_prob=cfg.mixup_prob,
        special_aug_ratio=cfg.special_aug_ratio, train=True
    )
    val_dataset = YoloDataset(
        data=val_datas, input_shape=cfg.input_shape, num_classes=num_classes, epoch_length=cfg.epoch,
        mosaic=False, mosaic_prob=0.0, mixup=False, mixup_prob=0.0, special_aug_ratio=0.0, train=False
    )
    train_loader = YoloDataLoader(
        dataset=train_dataset, batch_size=cfg.batch_size, shuttle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = YoloDataLoader(
        dataset=val_dataset, batch_size=cfg.batch_size, shuttle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )

    loss = Loss(model=model, cls_gain=cfg.cls_gain, box_gain=cfg.box_gain, dfl_gain=cfg.dfl_gain)  # 获得损失函数
    nbs = 64  # 根据当前batch_size，自适应调整学习率
    lr_limit_max = 1e-3 if cfg.optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if cfg.optimizer_type == 'adam' else 5e-4
    init_lr_fit = min(max(cfg.batch_size / nbs * cfg.init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(cfg.batch_size / nbs * cfg.min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    lr_scheduler_func = get_lr_scheduler(
        lr_decay_type=cfg.lr_decay_type, init_lr_fit=init_lr_fit, min_lr_fit=min_lr_fit, total_epoch=cfg.epoch
    )  # 获得学习率下降的公式

    param_group0, param_group1, param_group2 = [], [], []
    for k, v in model.named_modules():
        if isinstance(v, nn.BatchNorm2d) or 'bn' in k:
            param_group0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            param_group1.append(v.weight)
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            param_group2.append(v.bias)
    optimizer = {
        'adam': optim.Adam(params=param_group0, lr=init_lr_fit, betas=(cfg.momentum, 0.999)),
        'sgd': optim.SGD(params=param_group0, lr=init_lr_fit, momentum=cfg.momentum, nesterov=True)
    }[cfg.optimizer_type]  # 根据optimizer_type选择优化器
    optimizer.add_param_group({'params': param_group1, 'weight_decay': cfg.weight_decay})
    optimizer.add_param_group({'params': param_group2})

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(cfg.save_dir, 'loss_' + str(time_str))
    loss_log = LossLog(log_dir=log_dir, model=model, input_shape=cfg.input_shape)  # 记录Loss
    eval_log = EvalLog(class_names=class_names, num_classes=num_classes, log_dir=log_dir)  # 记录eval的map曲线

    model_train = nn.DataParallel(module=model)
    cudnn.benchmark = True
    model_train = model_train.cuda()
    ema = EMA(model=model_train)
    if ema:
        ema.updates = epoch_step_train * cfg.init_epoch

    for epoch in range(cfg.init_epoch, cfg.epoch):  # 开始模型训练
        train_dataset.epoch_now, val_dataset.epoch_now = epoch, epoch
        set_optimizer_lr(optimizer=optimizer, lr_scheduler_func=lr_scheduler_func, epoch=epoch)
        train(
            cfg=cfg, model_train=model_train, model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, ema=ema, loss=loss, loss_log=loss_log, eval_log=eval_log,
            epoch=epoch, epoch_step_train=epoch_step_train, epoch_step_val=epoch_step_val
        )

    loss_log.write.close()
    shutil.rmtree(os.path.join(log_dir, "detection-results/"))
    shutil.rmtree(os.path.join(log_dir, "ground-truth/"))
    shutil.rmtree(os.path.join(log_dir, "coco_eval/"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_epoch', type=int, default=0, help='初始训练世代数')
    parser.add_argument('--epoch', type=int, default=100, help='总训练世代数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='加载数据的线程数')
    parser.add_argument('--model_path', type=str, default='', help='预训练权重')
    parser.add_argument('--classes_path', type=str, default='model_data/voc_classes.txt', help='类别文件')
    parser.add_argument('--train_path', type=str, default='dataset/VOC/train.txt', help='训练集')
    parser.add_argument('--val_path', type=str, default='dataset/VOC/val.txt', help='验证集')
    parser.add_argument('--input_shape', type=Tuple, default=(640, 640), help='输入图像大小')
    parser.add_argument('--stride', type=Tuple, default=(8, 16, 32), help='下采样步长')
    parser.add_argument('--save_dir', type=str, default='logs', help='权值与日志文件保存的文件夹')
    parser.add_argument('--save_period', type=int, default=10, help='多少个epoch保存一次权值')
    parser.add_argument('--eval_flag', type=bool, default=True, help='是否在训练时进行评估')
    parser.add_argument('--eval_period', type=int, default=1, help='多少个epoch评估一次')
    parser.add_argument('--mosaic', type=bool, default=True, help='是否执行mosaic数据增强')
    parser.add_argument('--mosaic_prob', type=float, default=0.5, help='执行mosaic数据增强的概率')
    parser.add_argument('--mixup', type=bool, default=True, help='是否执行mixup数据增强')
    parser.add_argument('--mixup_prob', type=float, default=0.5, help='执行mixup数据增强的概率')
    parser.add_argument('--special_aug_ratio', type=float, default=0.7, help='执行特殊数据增强的比率')
    parser.add_argument('--init_lr', type=float, default=1e-3, help='初始/最大学习率')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='最小学习率')
    parser.add_argument('--momentum', type=float, default=0.937, help='动量')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='权重衰减')
    parser.add_argument('--optimizer_type', type=str, default='adam', help='优化器类型: adam、sgd')
    parser.add_argument('--lr_decay_type', type=str, default='cos', help='学习率调整策略: step、cos')
    parser.add_argument('--cls_gain', type=float, default=0.5, help='cls loss gain')
    parser.add_argument('--box_gain', type=float, default=7.5, help='box loss gain')
    parser.add_argument('--dfl_gain', type=float, default=1.5, help='dfl loss gain')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='非极大抑制的类别置信度阈值')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='非极大抑制的iou置信度阈值')
    parser.add_argument('--phi', type=str, default='n', help='版本: n, s, l, m, x')
    args = parser.parse_args()
    main(cfg=args)
