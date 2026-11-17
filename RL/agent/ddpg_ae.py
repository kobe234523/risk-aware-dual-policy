# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.encoder import make_encoder
from attention import GGCA

LOG_FREQ = 10000

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

# ------------------------------------- #
# 策略网络
# ------------------------------------- #
class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters, stride):
        super().__init__()
        # 环境可以接受的动作最大值
        self.action_bound = 1
        self.nb_actions = action_shape[0]

        # obs's encoder
        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, stride
        )

        # Attention module
        self.attention = GGCA(channel=256, h=4, w=4)

        self.fc1 = nn.Linear(self.encoder.feature_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.out = nn.Linear(hidden_dim, self.nb_actions)
        self.out.weight.data.normal_(0, 1e-4)

        self.apply(weight_init)


    def forward(self, obs, detach_encoder=False):
        # Strategy 2: Pure CNN Encoder -> Attention -> MLP
        # Get spatial features from encoder
        if hasattr(self.encoder, 'forward_spatial'):
            x = self.encoder.forward_spatial(obs)
            if detach_encoder:
                x = x.detach()
            
            # Apply Attention
            x = self.attention(x)
            
            # Flatten and project (using encoder's layers)
            x = x.view(x.size(0), -1)
            x = self.encoder.fc(x)
            x = self.encoder.ln(x)
        else:
            # Fallback for encoders that don't support forward_spatial (e.g. IdentityEncoder)
            x = self.encoder(obs, detach=detach_encoder)

        x = self.norm1(self.fc1(x))
        x = torch.relu(x)
        x = self.norm2(self.fc2(x))
        x = torch.relu(x)
        out = self.out(x)
        x = out.view(-1, self.nb_actions)

        x = torch.tanh(x)
        return x


# ------------------------------------- #
# 价值网络
# ------------------------------------- #

# 输出选定的Action的Q
class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters, stride):
        super(Critic, self).__init__()
        #
        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, stride
        )

        # Attention module
        self.attention = GGCA(channel=256, h=4, w=4)

        self.fc1 = nn.Linear(self.encoder.feature_dim + action_shape[0], hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.out = nn.Linear(hidden_dim, 1)
        self.apply(weight_init)


    def forward(self, obs, a, detach_encoder=False):
        # Strategy 2: Pure CNN Encoder -> Attention -> MLP
        if hasattr(self.encoder, 'forward_spatial'):
            x = self.encoder.forward_spatial(obs)
            if detach_encoder:
                x = x.detach()
            
            # Apply Attention
            x = self.attention(x)
            
            # Flatten and project
            x = x.view(x.size(0), -1)
            x = self.encoder.fc(x)
            x = self.encoder.ln(x)
        else:
            x = self.encoder(obs, detach=detach_encoder)

        x = torch.cat([x, a], 1)
        x = self.norm1(self.fc1(x))
        x = torch.relu(x)
        x = self.norm2(self.fc2(x))
        x = torch.relu(x)

        out = self.out(x)

        x = out.view(-1, 1)

        return x



