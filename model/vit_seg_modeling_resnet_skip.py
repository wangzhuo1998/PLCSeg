import math
from CloudRemoval import CloudRemoval, DataFusion
from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()

        self.layer_one = nn.Linear(channel, 1)
        self.activation_one = nn.ReLU()
        self.layer_two = nn.Linear(1, channel)
        self.activation_two = nn.ReLU()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sigmoid = nn.Hardsigmoid()


    def forward(self, input_l, cross):
        b, c, h, w = input_l.shape

        x = input_l
        avg_pool = self.avg_pool(x)
        avg_pool = avg_pool.reshape(b, 1, 1, c).contiguous()
        avg_pool = self.layer_one(avg_pool)
        avg_pool = self.activation_one(avg_pool)
        avg_pool = self.layer_two(avg_pool)
        avg_pool = self.activation_two(avg_pool)

        max_pool = self.max_pool(x)
        max_pool = max_pool.reshape(b, 1, 1, c).contiguous()
        max_pool = self.layer_one(max_pool)
        max_pool = self.activation_one(max_pool)
        max_pool = self.layer_two(max_pool)
        max_pool = self.activation_two(max_pool)

        weight = self.sigmoid(max_pool + avg_pool)
        weight = weight.permute(0, 3, 1, 2).contiguous()

        avg_pool_cross = self.avg_pool(cross)
        avg_pool_cross = avg_pool_cross.reshape(b, 1, 1, c).contiguous()
        avg_pool_cross = self.layer_one(avg_pool_cross)
        avg_pool_cross = self.activation_one(avg_pool_cross)
        avg_pool_cross = self.layer_two(avg_pool_cross)
        avg_pool_cross = self.activation_two(avg_pool_cross)

        max_pool_cross = self.max_pool(cross)
        max_pool_cross = max_pool_cross.reshape(b, 1, 1, c).contiguous()
        max_pool_cross = self.layer_one(max_pool_cross)
        max_pool_cross = self.activation_one(max_pool_cross)
        max_pool_cross = self.layer_two(max_pool_cross)
        max_pool_cross = self.activation_two(max_pool_cross)

        weight_cross = self.sigmoid(max_pool_cross + avg_pool_cross)
        weight_cross = weight_cross.permute(0, 3, 1, 2).contiguous()

        y = input_l * weight * weight_cross

        return y

#C2FFM
class C2FFM(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(C2FFM, self).__init__()

        self.se_cloudy = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_decloud = SqueezeAndExcitation(channels_in,
                                             activation=activation)
        self.se_sar = SqueezeAndExcitation(channels_in,
                                             activation=activation)
        self.cross = SqueezeAndExcitation(channels_in,
                                          activation=activation)

    def forward(self, cloudy, decloud, sar):
        cross = cloudy + decloud + sar#1 64 128 128
        cloudy = self.se_cloudy(cloudy, cross)#1 64 128 128
        decloud = self.se_decloud(decloud, cross)
        sar = self.se_sar(sar, cross)

        out = cloudy + decloud + sar
        return out

def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(4, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2] + 1)],
            ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


class FuseResNetV2(nn.Module):
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.activation = nn.ReLU(inplace=True)

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(4, width, kernel_size=7, stride=2, bias=False, padding=3)),
            # ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.rootd = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(2, width, kernel_size=7, stride=2, bias=False, padding=3)),
            # ('conv', StdConv2d(1, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.se_layer0 =C2FFM(
            64, activation=self.activation)
        self.se_layer1 = C2FFM(
            256,
            activation=self.activation)
        self.se_layer2 = C2FFM(
            512,
            activation=self.activation)
        self.se_layer3 = C2FFM(
            1024,
            activation=self.activation)

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2] + 1)],
            ))),
        ]))

        self.bodyd = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2] + 1)],
            ))),
        ]))

        self.net_cloudfree_G = CloudRemoval(opt_channel=4, sar_channel=2).cuda()
        self.net_cloudfree_G = nn.DataParallel(self.net_cloudfree_G)
        # 加载预训练的去云模型权重
        checkpoint = torch.load('/home8/wz/code_file/ALL_MT_CRfirst+SSloss/cloudremovel/checkpoints/best_psnr_net.pth', weights_only=True)
        self.net_cloudfree_G.load_state_dict(checkpoint['network'])
        self.net_cloudfree_G.eval()
        for _, param in self.net_cloudfree_G.named_parameters():
            param.requires_grad = False


    def forward(self, x, y):
        global fusion_stage4
        fusion_features = []
        cloudremoval_result = self.net_cloudfree_G(x, y)
        SE = True
        b, c, in_size, _ = x.size()
        x = self.root(x)
        CR_x = self.root(cloudremoval_result)
        y = self.rootd(y)
        if SE:
            fusion_stage1 = self.se_layer0(x, CR_x, y)
            fusion_features.append(fusion_stage1)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        CR_x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(CR_x)
        y = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(y)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            CR_x = self.body[i](CR_x)
            y = self.bodyd[i](y)

            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x

            if CR_x.size()[2] != right_size:
                pad = right_size - CR_x.size()[2]
                assert pad < 3 and pad > 0, "CR_x {} should {}".format(CR_x.size(), right_size)
                CR_feat = torch.zeros((b, CR_x.size()[1], right_size, right_size), device=CR_x.device)
                CR_feat[:, :, 0:CR_x.size()[2], 0:CR_x.size()[3]] = CR_x[:]
            else:
                CR_feat = CR_x

            if y.size()[2] != right_size:
                pad = right_size - y.size()[2]
                assert pad < 3 and pad > 0, "y {} should {}".format(y.size(), right_size)
                y_feat = torch.zeros((b, y.size()[1], right_size, right_size), device=y.device)
                y_feat[:, :, 0:y.size()[2], 0:y.size()[3]] = y[:]
            else:
                y_feat = y

            if SE:
                if i == 0:
                    fusion_stage2 = self.se_layer1(feat, CR_feat, y_feat)
                    fusion_features.append(fusion_stage2)
                if i == 1:
                    fusion_stage3 = self.se_layer2(x, CR_x, y_feat)
                    fusion_features.append(fusion_stage3)


        x = self.body[-1](x)
        CR_x = self.body[-1](CR_x)
        y = self.bodyd[-1](y)
        if SE:
            fusion_stage4 = self.se_layer3(x, CR_x, y)

        return fusion_stage4, y, fusion_features[::-1]
