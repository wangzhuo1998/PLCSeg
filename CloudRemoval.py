import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, feature_size, reduction=256):
        super().__init__()
        self.layer_one = nn.Linear(feature_size, feature_size // reduction)
        self.activation_one = nn.ReLU()
        self.layer_two = nn.Linear(feature_size // reduction, feature_size)
        self.activation_two = nn.ReLU()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sigmoid = nn.Hardsigmoid()

    def forward(self, input_l):
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

        return input_l * weight

class DataFusion(nn.Module):
    def __init__(self, input_shape, feature_size):
        super().__init__()
        self.conv = nn.Conv2d(input_shape, feature_size, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.attention = Attention(feature_size)
        self.counter = 0

    def forward(self, input_l):
        self.counter += 1
        x = self.conv(input_l)
        x = self.act(x)
        x = self.attention(x)
        return x

class DehazeNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.init_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, dilation=1)
        self.init_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, dilation=1)


        self.branch1_conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, dilation=1)
        self.branch1_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, dilation=1)


        self.branch2_conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, dilation=1)
        self.branch2_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2, dilation=2)


        self.branch3_conv1 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=2, dilation=2)
        self.branch3_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=2, dilation=2)


        self.agg_conv1 = nn.Conv2d(in_channels=(64 + 128 + 128), out_channels=128, kernel_size=3, padding=1, dilation=1)
        self.agg_conv2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, padding=0, dilation=1)  # 输出ω

        self.bias = 1

    def forward(self, x):

        features = F.relu(self.init_conv1(x))
        features = F.relu(self.init_conv2(features))


        branch1 = F.relu(self.branch1_conv1(features))
        branch1 = F.relu(self.branch1_conv2(branch1))


        branch2 = F.relu(self.branch2_conv1(features))
        branch2 = F.relu(self.branch2_conv2(branch2))


        branch3 = F.relu(self.branch3_conv1(features))
        branch3 = F.relu(self.branch3_conv2(branch3))


        concatenated = torch.cat([branch1, branch2, branch3], dim=1)


        agg = F.relu(self.agg_conv1(concatenated))
        omega = self.agg_conv2(agg)


        omega_expanded = omega.expand_as(x)
        dehaze_image = omega_expanded * x - omega_expanded + self.bias

        return dehaze_image


class CloudRemoval(nn.Module):
    def __init__(self, opt_channel, sar_channel, feature_size=256):
        super().__init__()
        self.opt_channel = opt_channel
        self.data_fusion = DataFusion(opt_channel + sar_channel, 4)
        self.cloud_removal = DehazeNet(4)

    def forward(self, input_opt, input_sar, output_shape=None):
        x = torch.cat([input_opt, input_sar], dim=1)  #2 6 160 160
        fusion_feature = self.data_fusion(x)  #2 4 160 160
        cloudremoval_result = self.cloud_removal(fusion_feature)  #2 4 160 160
        return cloudremoval_result


if __name__ == '__main__':
    opt = torch.randn(2, 4, 160, 160)
    sar = torch.randn(2, 2, 160, 160)
    model = CloudRemoval(4, 2)
    print(model)
    y = model(opt, sar)  #2 7 160 160
    y = y[0]
    print(y.shape)  # (2, 3, 128, 128)

