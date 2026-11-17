# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from attention import GGCA


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias
#这是一个双塔模型，目的是为了减少内存，使得两层模型可以共享权重和偏置

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = {2: 39, 4: 35, 6: 31}[num_layers]
        self.fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        conv = torch.relu(self.convs[1](conv))
        self.outputs['conv2'] = conv

        conv = torch.relu(self.convs[2](conv))
        self.outputs['conv3'] = conv

        conv = torch.relu(self.convs[3](conv))
        self.outputs['conv4'] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward_spatial(self, obs):
        obs = obs / 255.
        conv = torch.relu(self.convs[0](obs))
        conv = torch.relu(self.convs[1](conv))
        conv = torch.relu(self.convs[2](conv))
        conv = torch.relu(self.convs[3](conv))
        return conv

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        out = self.ln(h_fc)
        self.outputs['ln'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class PixelEncoderCarla(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super(PixelEncoder, self).__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(obs_shape[0], 64, 5, stride=2))
        self.convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.convs.append(nn.Conv2d(256, 256, 3, stride=2))

        out_dims = 16  # 1 cameras
        # out_dims = 56  # 3 cameras
        # out_dims = 100  # 5 cameras
        self.fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()


class PixelEncoderCarlaAttention(PixelEncoderCarla):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super().__init__(obs_shape, feature_dim, num_layers, num_filters, stride)
        self.attention = GGCA(channel=256, h=4, w=4)

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        conv = torch.relu(self.convs[1](conv))
        self.outputs['conv2'] = conv

        conv = torch.relu(self.convs[2](conv))
        self.outputs['conv3'] = conv

        conv = torch.relu(self.convs[3](conv))
        self.outputs['conv4'] = conv

        conv = self.attention(conv)
        self.outputs['attention'] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward_spatial(self, obs):
        obs = obs / 255.
        conv = torch.relu(self.convs[0](obs))
        conv = torch.relu(self.convs[1](conv))
        conv = torch.relu(self.convs[2](conv))
        conv = torch.relu(self.convs[3](conv))
        return conv

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        out = self.ln(h_fc)
        self.outputs['ln'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class PixelEncoderCarlaAttention(PixelEncoderCarla):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super().__init__(obs_shape, feature_dim, num_layers, num_filters, stride)
        self.attention = GGCA(channel=256, h=4, w=4)

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        conv = torch.relu(self.convs[1](conv))
        self.outputs['conv2'] = conv

        conv = torch.relu(self.convs[2](conv))
        self.outputs['conv3'] = conv

        conv = torch.relu(self.convs[3](conv))
        self.outputs['conv4'] = conv

        conv = self.attention(conv)
        self.outputs['attention'] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward_spatial(self, obs):
        obs = obs / 255.
        conv = torch.relu(self.convs[0](obs))
        conv = torch.relu(self.convs[1](conv))
        conv = torch.relu(self.convs[2](conv))
        conv = torch.relu(self.convs[3](conv))
        return conv

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        out = self.ln(h_fc)
        self.outputs['ln'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class PixelEncoderCarlaSE(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = 4

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(obs_shape[0], 64, 5, stride=2))
        self.convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.convs.append(nn.Conv2d(256, 256, 3, stride=2))

        self.se = nn.ModuleList()
        self.se.append(SEBlock(64))
        self.se.append(SEBlock(128))
        self.se.append(SEBlock(256))
        self.se.append(SEBlock(256))

        out_dims = 16
        self.fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        x = torch.relu(self.convs[0](obs))
        x = self.se[0](x)
        self.outputs['conv1'] = x

        x = torch.relu(self.convs[1](x))
        x = self.se[1](x)
        self.outputs['conv2'] = x

        x = torch.relu(self.convs[2](x))
        x = self.se[2](x)
        self.outputs['conv3'] = x

        x = torch.relu(self.convs[3](x))
        x = self.se[3](x)
        self.outputs['conv4'] = x

        h = x.view(x.size(0), -1)
        return h

    def forward_spatial(self, obs):
        obs = obs / 255.
        x = torch.relu(self.convs[0](obs))
        x = self.se[0](x)
        x = torch.relu(self.convs[1](x))
        x = self.se[1](x)
        x = torch.relu(self.convs[2](x))
        x = self.se[2](x)
        x = torch.relu(self.convs[3](x))
        x = self.se[3](x)
        return x

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()
        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc
        out = self.ln(h_fc)
        self.outputs['ln'] = out
        return out

    def copy_conv_weights_from(self, source):
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])
        for i in range(self.num_layers):
            self.se[i].load_state_dict(source.se[i].state_dict())

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return
        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)
        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)

class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder,
                       'pixelCarla': PixelEncoderCarla,
                       'pixelCarlaAttention': PixelEncoderCarlaAttention,
                       'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, stride
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, stride
    )

_AVAILABLE_ENCODERS['pixelCarlaSE'] = PixelEncoderCarlaSE
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)