import torch
from torch import nn
import torch.nn.functional as F


class GGCA(nn.Module):
    def __init__(self, channel, h=None, w=None, reduction=16, num_groups=4):
        super().__init__()
        assert channel % num_groups == 0
        self.num_groups = num_groups
        self.group_channels = channel // num_groups

        hidden = max(8, self.group_channels // reduction)

        self.shared_conv = nn.Sequential(
            nn.Conv2d(self.group_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.group_channels, kernel_size=1, bias=False),
        )

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x_group = x.view(b, self.num_groups, self.group_channels, h, w)
        x_reshape = x_group.view(b * self.num_groups, self.group_channels, h, w)

        x_h_avg = F.adaptive_avg_pool2d(x_reshape, (h, 1))
        x_h_max = F.adaptive_max_pool2d(x_reshape, (h, 1))
        x_w_avg = F.adaptive_avg_pool2d(x_reshape, (1, w))
        x_w_max = F.adaptive_max_pool2d(x_reshape, (1, w))

        y_h = self.shared_conv(x_h_avg) + self.shared_conv(x_h_max)
        y_w = self.shared_conv(x_w_avg) + self.shared_conv(x_w_max)

        att_h = self.sigmoid_h(y_h).view(b, self.num_groups, self.group_channels, h, 1)
        att_w = self.sigmoid_w(y_w).view(b, self.num_groups, self.group_channels, 1, w)

        out = x_group * att_h * att_w
        out = out.view(b, c, h, w)

        return out