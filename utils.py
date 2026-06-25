import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import itertools
from torchvision.utils import make_grid
from PIL import Image
from skimage import io
import os

# Parameters
## SwinFusion

WINDOW_SIZE = (256, 256) # Patch size

LABELS = ["Farmland", "City", "Village", "Water", "Forest", "Road", "Others"] # Label names
N_CLASSES = 7# Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing


def save_img(tensor, name):
    tensor = tensor.cpu() .permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')


def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        target = target.long()
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def metrics(predictions, gts, mask, label_values=LABELS):
    # 获取有云区域和无云区域的预测和真实值
    cloud_region = mask == 1  # 有云区域
    no_cloud_region = mask == 0  # 无云区域

    # 有云区域的预测和真实值
    cloud_preds = predictions[cloud_region]
    cloud_gts = gts[cloud_region]

    # 无云区域的预测和真实值
    no_cloud_preds = predictions[no_cloud_region]
    no_cloud_gts = gts[no_cloud_region]

    # 整张图像的预测和真实值
    all_preds = predictions
    all_gts = gts

    # 分别计算有云、无云区域和整张图像的混淆矩阵
    cm_cloud = confusion_matrix(cloud_gts, cloud_preds, labels=range(len(label_values)))
    cm_no_cloud = confusion_matrix(no_cloud_gts, no_cloud_preds, labels=range(len(label_values)))
    cm_all = confusion_matrix(all_gts, all_preds, labels=range(len(label_values)))

    print("Confusion matrix for cloud region:")
    print(cm_cloud)
    print("Confusion matrix for no cloud region:")
    print(cm_no_cloud)
    print("Confusion matrix for the entire image:")
    print(cm_all)

    # 计算全局准确率
    total_cloud = sum(sum(cm_cloud))
    total_no_cloud = sum(sum(cm_no_cloud))
    total_all = sum(sum(cm_all))

    accuracy_cloud = sum([cm_cloud[x][x] for x in range(len(cm_cloud))]) * 100 / float(total_cloud)
    accuracy_no_cloud = sum([cm_no_cloud[x][x] for x in range(len(cm_no_cloud))]) * 100 / float(total_no_cloud)
    accuracy_all = sum([cm_all[x][x] for x in range(len(cm_all))]) * 100 / float(total_all)

    print("%d pixels processed in cloud region" % (total_cloud))
    print("%d pixels processed in no cloud region" % (total_no_cloud))
    print("%d pixels processed in entire image" % (total_all))

    print("Cloud region accuracy : %.2f" % (accuracy_cloud))
    print("No cloud region accuracy : %.2f" % (accuracy_no_cloud))
    print("Total accuracy (all regions): %.2f" % (accuracy_all))

    # 计算每个区域的类别准确率
    Acc_cloud = np.diag(cm_cloud) / cm_cloud.sum(axis=1)
    Acc_no_cloud = np.diag(cm_no_cloud) / cm_no_cloud.sum(axis=1)
    Acc_all = np.diag(cm_all) / cm_all.sum(axis=1)

    for l_id, score in enumerate(Acc_cloud):
        print("%s (cloud region): %.4f" % (label_values[l_id], score))
    for l_id, score in enumerate(Acc_no_cloud):
        print("%s (no cloud region): %.4f" % (label_values[l_id], score))
    for l_id, score in enumerate(Acc_all):
        print("%s (entire image): %.4f" % (label_values[l_id], score))

    # 计算有云区域、无云区域和整张图像的平均准确率
    mean_accuracy_cloud = np.nanmean(Acc_cloud[:6])
    mean_accuracy_no_cloud = np.nanmean(Acc_no_cloud[:6])
    mean_accuracy_all = np.nanmean(Acc_all[:6])

    print("Mean accuracy (cloud region): %.4f" % (mean_accuracy_cloud))
    print("Mean accuracy (no cloud region): %.4f" % (mean_accuracy_no_cloud))
    print("Mean accuracy (entire image): %.4f" % (mean_accuracy_all))
    print("---")

    # 计算每个区域的 F1 分数
    F1Score_cloud = np.zeros(len(label_values))
    F1Score_no_cloud = np.zeros(len(label_values))
    F1Score_all = np.zeros(len(label_values))

    for i in range(len(label_values)):
        try:
            F1Score_cloud[i] = 2. * cm_cloud[i, i] / (np.sum(cm_cloud[i, :]) + np.sum(cm_cloud[:, i]))
            F1Score_no_cloud[i] = 2. * cm_no_cloud[i, i] / (np.sum(cm_no_cloud[i, :]) + np.sum(cm_no_cloud[:, i]))
            F1Score_all[i] = 2. * cm_all[i, i] / (np.sum(cm_all[i, :]) + np.sum(cm_all[:, i]))
        except:
            pass

    print("F1Score (cloud region):")
    for l_id, score in enumerate(F1Score_cloud):
        print("%s: %.4f" % (label_values[l_id], score))
    print("F1Score (no cloud region):")
    for l_id, score in enumerate(F1Score_no_cloud):
        print("%s: %.4f" % (label_values[l_id], score))
    print("F1Score (entire image):")
    for l_id, score in enumerate(F1Score_all):
        print("%s: %.4f" % (label_values[l_id], score))

    print('Mean F1Score (cloud region): %.4f' % (np.nanmean(F1Score_cloud[:6])))
    print('Mean F1Score (no cloud region): %.4f' % (np.nanmean(F1Score_no_cloud[:6])))
    print('Mean F1Score (entire image): %.4f' % (np.nanmean(F1Score_all[:6])))
    print("---")

    # 计算Kappa系数
    total_cloud = np.sum(cm_cloud)
    total_no_cloud = np.sum(cm_no_cloud)
    total_all = np.sum(cm_all)

    pa_cloud = np.trace(cm_cloud) / float(total_cloud)
    pe_cloud = np.sum(np.sum(cm_cloud, axis=0) * np.sum(cm_cloud, axis=1)) / float(total_cloud * total_cloud)
    kappa_cloud = (pa_cloud - pe_cloud) / (1 - pe_cloud)

    pa_no_cloud = np.trace(cm_no_cloud) / float(total_no_cloud)
    pe_no_cloud = np.sum(np.sum(cm_no_cloud, axis=0) * np.sum(cm_no_cloud, axis=1)) / float(
        total_no_cloud * total_no_cloud)
    kappa_no_cloud = (pa_no_cloud - pe_no_cloud) / (1 - pe_no_cloud)

    pa_all = np.trace(cm_all) / float(total_all)
    pe_all = np.sum(np.sum(cm_all, axis=0) * np.sum(cm_all, axis=1)) / float(total_all * total_all)
    kappa_all = (pa_all - pe_all) / (1 - pe_all)

    print("Kappa (cloud region): %.4f" % (kappa_cloud))
    print("Kappa (no cloud region): %.4f" % (kappa_no_cloud))
    print("Kappa (entire image): %.4f" % (kappa_all))

    # 计算MIoU系数
    epsilon = 1e-7  # 防止除以零
    MIoU_cloud = np.diag(cm_cloud) / (np.sum(cm_cloud, axis=1) + np.sum(cm_cloud, axis=0) - np.diag(cm_cloud) + epsilon)
    MIoU_no_cloud = np.diag(cm_no_cloud) / (
                np.sum(cm_no_cloud, axis=1) + np.sum(cm_no_cloud, axis=0) - np.diag(cm_no_cloud) + epsilon)
    MIoU_all = np.diag(cm_all) / (np.sum(cm_all, axis=1) + np.sum(cm_all, axis=0) - np.diag(cm_all) + epsilon)

    print("MIoU (cloud region):", MIoU_cloud)
    print("MIoU (no cloud region):", MIoU_no_cloud)
    print("MIoU (entire image):", MIoU_all)

    MIoU_cloud_mean = np.nanmean(MIoU_cloud[:6])
    MIoU_no_cloud_mean = np.nanmean(MIoU_no_cloud[:6])
    MIoU_all_mean = np.nanmean(MIoU_all[:6])

    print('Mean MIoU (cloud region): %.4f' % (MIoU_cloud_mean))
    print('Mean MIoU (no cloud region): %.4f' % (MIoU_no_cloud_mean))
    print('Mean MIoU (entire image): %.4f' % (MIoU_all_mean))
    print("---")

    return mean_accuracy_cloud, mean_accuracy_no_cloud, mean_accuracy_all, MIoU_cloud_mean, MIoU_no_cloud_mean, MIoU_all_mean
