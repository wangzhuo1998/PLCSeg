import numpy as np
from glob import glob
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
import random
import time
import itertools
import matplotlib.pyplot as plt
import torch
import csv
import torch.nn as nn
# from metrics import StreamSegMetrics
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *
from torch.autograd import Variable
from IPython.display import clear_output
from model.vitcross_seg_modelingmask import VisionTransformer as ViT_seg
from model.vitcross_seg_modelingmask import CONFIGS as CONFIGS_ViT_seg
from model_base import print_options
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataloader import get_filelist, TrainDataset, ValDataset
from pynvml import *


def seed_torch(seed=911):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


# 在代码最前面调用
seed_torch(911)  # 你可以更改这个种子数

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))
print("Device :", nvmlDeviceGetName(handle))

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = 7
config_vit.n_skip = 3
config_vit.patches.grid = (int(256 / 16), int(256 / 16))
net = ViT_seg(config_vit, img_size=256, num_classes=7).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path))
params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)


class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        """
        计算多分类 Dice Loss
        :param pred: 形状为 [batch, num_classes, height, width]，未经过 softmax 的 logits
        :param target: 形状为 [batch, height, width]，整数标签 (0 ~ num_classes-1)
        :return: Dice Loss
        """
        pred = F.softmax(pred, dim=1)  # 先对 logits 进行 softmax，获取每个类别的概率
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()  # 转换为 one-hot 形式

        intersection = torch.sum(pred * target_one_hot, dim=(2, 3))  # 计算交集
        denominator = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))  # 计算分母

        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)  # 计算 Dice 系数
        dice_loss = 1 - dice_score.mean()  # 取所有类别的均值作为最终损失

        return dice_loss


##===================================================##
##********** Configure training settings ************##
##===================================================##
parser = argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')
# parser.add_argument('--batch_sz', type=int, default=20, help='batch size used for training')
parser.add_argument('--input_data_folder', type=str, default='/home3/wz/data/M3M-CR/train')
parser.add_argument('--input_testdata_folder', type=str, default='/home3/wz/data/M3M-CR/test')
# parser.add_argument('--train_list_filepath', type=str, default='/home3/wz/data/M3M-CR/csv/train.csv')
parser.add_argument('--train_list_filepath', type=str, default='/home3/wz/data/M3M-CR/csv/one_train_sample.csv')
parser.add_argument('--val_list_filepath', type=str, default='/home3/wz/data/M3M-CR/csv/val.csv')
# parser.add_argument('--test_list_filepath', type=str, default='/home3/wz/data/M3M-CR/csv/test.csv')
parser.add_argument('--test_list_filepath', type=str, default='/home3/wz/data/M3M-CR/csv/one_test_sample.csv')
parser.add_argument('--is_load_SAR', type=bool, default=True)
parser.add_argument('--is_upsample_SAR', type=bool, default=True)  # only useful when is_load_SAR = True
parser.add_argument('--is_load_landcover', type=bool, default=True)
parser.add_argument('--is_upsample_landcover', type=bool,
                    default=True)  # only useful when is_load_landcover = True
parser.add_argument('--lc_level', type=str, default='1')  # only useful when is_load_landcover = True
parser.add_argument('--is_load_cloudmask', type=bool, default=True)
parser.add_argument('--load_size', type=int, default=300)
parser.add_argument('--crop_size', type=int, default=256)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--save_model_dir', type=str,
                    default='')
parser.add_argument('--log_dir', type=str,
                    default='',
                    help='Path to save logs and TensorBoard files')
parser.add_argument('--max_epochs', type=int, default=50)
parser.add_argument('--continue_train_checkpoint', type=str, default="")

opts = parser.parse_args()
print_options(opts)

train_filelist = get_filelist(opts.train_list_filepath)
train_data = TrainDataset(opts, train_filelist)
test_filelist = get_filelist(opts.test_list_filepath)
test_data = ValDataset(opts, test_filelist)
print("Train set: %d, Val set: %d" % (len(train_data), len(test_data)))

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=opts.batch_sz, shuffle=True,
                                           num_workers=4, drop_last=True)

test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=opts.batch_sz, shuffle=False)


print("BATCH_SIZE: ", opts.batch_sz)


base_lr = 0.01
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params': [value], 'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params': [value], 'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45],
                                           gamma=0.1)  #[25, 35, 45]：表示 在 epoch = 25, 35, 45 时降低学习率。gamma=0.1：表示 学习率衰减 10 倍（即 lr = lr * 0.1）。
loss_SS_fn = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
loss_dice = MultiClassDiceLoss(num_classes=7)
alpha = 0.4

writer = SummaryWriter(log_dir=os.path.join(opts.log_dir, "train_logs"))

# 创建 CSV 文件
train_loss_file = os.path.join(opts.log_dir, "train_loss.csv")
val_loss_file = os.path.join(opts.log_dir, "val_loss.csv")

# 写入表头
with open(train_loss_file, mode='w', newline='') as f:
    writer1 = csv.writer(f)
    writer1.writerow(["Epoch", "Step", "Train_SS_Loss", "Train_Dice_Loss", "Train_total_Loss"])
with open(val_loss_file, mode='w', newline='') as f:
    writer4 = csv.writer(f)
    writer4.writerow(["Epoch", "val_SS_Loss", "val_Dice_Loss", "Val_total_Loss", "Mean IoU"])

# 检查是否存在已保存的模型
if os.path.exists(opts.continue_train_checkpoint):
    checkpoint = torch.load(opts.continue_train_checkpoint)  # 加载模型
    net.load_state_dict(checkpoint['network'])  # 加载网络参数
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch'] + 1  # 继续训练的起始epoch
    scheduler.load_state_dict(checkpoint['lr_scheduler'])  # 加载学习率调度器
    print(f"成功加载模型，继续从 epoch {start_epoch} 开始训练！")
else:
    start_epoch = 1  # 如果没有模型，则从头开始训练
    print("未找到已保存的模型，从头开始训练。")


def test(net, e):
    all_preds = []
    all_gts = []
    cloudmasks = []
    val_loss = 0
    val_SS_loss = 0
    val_Dice_loss = 0

    # Switch the network to inference mode
    with torch.no_grad():
        _iter1 = 0
        for datas in tqdm(test_dataloader, total=len(test_dataloader), desc="测试进度", leave=True):
            # unique_values = torch.unique(datas['landcover_data'])
            # print(unique_values)

            cloudfree_img = Variable(datas['cloudy_data'].cuda())  # 1 4 300 300
            sar_img = Variable(datas['SAR_data'].cuda())
            landcover_img = Variable(datas['landcover_data'].cuda())
            cloudmask = Variable(datas['cloudmask_data'].cuda())

            pred = net(cloudfree_img, sar_img)
            val_SS = loss_SS_fn(pred, landcover_img.long())
            val_Dice = loss_dice(pred, landcover_img.long())
            batch_val_loss = alpha * val_SS + (1 - alpha) * val_Dice
            # batch_val_loss = loss_SS_fn(pred, landcover_img.long())
            val_loss += batch_val_loss
            val_SS_loss += val_SS
            val_Dice_loss += val_Dice

            pred = np.argmax(pred.detach().cpu().numpy()[0], axis=0)
            landcover_img = landcover_img.data.cpu().numpy()[0]
            cloudmask = cloudmask.data.cpu().numpy()[0]
            all_preds.append(pred)
            all_gts.append(landcover_img)
            cloudmasks.append(cloudmask)
            clear_output()
            _iter1 += 1
        # 计算验证集的平均损失
        avg_val_loss = val_loss / _iter1
        avg_val_SS_loss = val_SS_loss / _iter1
        avg_val_Dice_loss = val_Dice_loss / _iter1
        print(f'Validation Loss: {avg_val_loss.item()}')
        # 将验证损失记录到 TensorBoard
        # self.writer.add_scalars('Loss', {'val_loss': avg_val_loss}, epoch+1)
        writer.add_scalar('Val_Loss/val_loss', avg_val_loss.item(), e)
        writer.add_scalar('Val_Loss/val_SS_loss', avg_val_SS_loss.item(), e)
        writer.add_scalar('Val_Loss/val_Dice_loss', avg_val_Dice_loss.item(), e)

    accuracy_cloud, accuracy_no_cloud, accuracy_all, mIoU_cloud, mIoU_no_cloud, mIoU_all = metrics(
        np.concatenate([p.ravel() for p in all_preds]),
        np.concatenate([p.ravel() for p in all_gts]).ravel(),
        np.concatenate([m.ravel() for m in cloudmasks])  # 传入掩膜
    )
    writer.add_scalar('Metric/Mean_IoU', mIoU_all, e)
    # 保存验证损失到 CSV
    with open(val_loss_file, mode='a', newline='') as f:
        writer3 = csv.writer(f)
        writer3.writerow([e, avg_val_SS_loss.item(), avg_val_Dice_loss.item(), avg_val_loss.item(), mIoU_all])
    return accuracy_cloud, accuracy_no_cloud, accuracy_all, mIoU_cloud, mIoU_no_cloud, mIoU_all


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    iter_ = 0
    miou_best = 0

    for e in range(start_epoch, epochs + 1):
        train_loss = 0
        train_SS_loss = 0
        train_Dice_loss = 0
        num_batch = 0
        if scheduler is not None:
            scheduler.step()
        net.train()
        # for batch_idx, (data, dsm, target) in enumerate(train_loader):
        for datas in tqdm(train_loader, total=len(train_loader), desc="训练进度", leave=True):
            iter_ += 1
            data, dsm, target = Variable(datas['cloudy_data'].cuda()), Variable(datas['SAR_data'].cuda()), Variable(
                datas['landcover_data'].cuda())
            optimizer.zero_grad()
            output = net(data, dsm)
            SS_loss = loss_SS_fn(output, target.long())
            Dice_loss = loss_dice(output, target.long())
            batch_loss = alpha * SS_loss + (1 - alpha) * Dice_loss
            # batch_loss = loss_SS_fn(output, target.long())
            train_loss = train_loss + batch_loss
            train_SS_loss = train_SS_loss + SS_loss
            train_Dice_loss = train_Dice_loss + Dice_loss
            num_batch = num_batch + 1
            batch_loss.backward()
            optimizer.step()

            if iter_ % 500 == 0:
                # if iter_ % 1 == 0:
                clear_output()
                pred = np.argmax(output.detach().cpu().numpy()[0], axis=0)
                gt = target.detach().cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, num_batch, len(train_loader),
                    100. * num_batch / len(train_loader), batch_loss.item(), accuracy(pred, gt)))  #OA

            del (data, target, batch_loss)

        train_loss /= num_batch
        train_SS_loss /= num_batch
        train_Dice_loss /= num_batch
        print('epoch', e, 'steps', iter_, 'ave_loss', train_loss.item())
        # 将训练损失记录到 TensorBoard
        writer.add_scalar('Train_Loss/train_loss', train_loss.item(), e)
        writer.add_scalar('Train_Loss/train_SS_loss', train_SS_loss.item(), e)
        writer.add_scalar('Train_Loss/train_Dice_loss', train_Dice_loss.item(), e)

        # 保存训练损失到 CSV
        with open(train_loss_file, mode='a', newline='') as f:
            writer2 = csv.writer(f)
            writer2.writerow([e, iter_, train_SS_loss.item(), train_Dice_loss.item(), train_loss.item()])

        if e % opts.save_freq == 0:
            # if iter_ % 500 == 0:
            # if iter_ % 1 == 0:
            net.eval()
            accuracy_cloud, accuracy_no_cloud, accuracy_all, mIoU_cloud, mIoU_no_cloud, mIoU_all = test(net, e)
            net.train()
            save_network(net, optimizer, e, accuracy_cloud, accuracy_no_cloud, accuracy_all, mIoU_cloud, mIoU_no_cloud, mIoU_all, scheduler, opts.save_model_dir)
            if mIoU_all > miou_best:
                miou_best = mIoU_all
    print('miou_best: ', miou_best)


def save_network(network, optimizer, epoch, accuracy_cloud, accuracy_no_cloud, accuracy_all, mIoU_cloud, mIoU_no_cloud, mIoU_all, lr_scheduler, save_dir):
    checkpoint = {
        "network": network.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        "lr_scheduler": lr_scheduler.state_dict()
    }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_filename = '%s_%s_%s_%s_%s_%s_%s_net.pth' % (str(epoch), str(mIoU_all), str(accuracy_all), str(mIoU_cloud), str(accuracy_cloud), str(mIoU_no_cloud), str(accuracy_no_cloud))
    save_path = os.path.join(save_dir, save_filename)
    torch.save(checkpoint, save_path)


#####   train   ####
time_start = time.time()
train(net, optimizer, opts.max_epochs, scheduler)
time_end = time.time()
print('Total Time Cost: ', time_end - time_start)


