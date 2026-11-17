# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils
from agent.ddpg_ae import Actor, Critic, LOG_FREQ
from agent.transition_model import make_transition_model

class DeepMDPAgent_DDPG(object):
    """Baseline algorithm with transition model and various decoder types."""

    def __init__(
            self,
            obs_shape,
            action_shape,
            device,
            hidden_dim=256,
            discount=0.99,
            init_temperature=0.01,
            alpha_lr=1e-3,
            alpha_beta=0.9,
            actor_lr=1e-3,
            actor_beta=0.9,
            actor_log_std_min=-10,
            actor_log_std_max=2,
            actor_update_freq=2,
            encoder_stride=2,
            critic_lr=1e-3,
            critic_beta=0.9,
            critic_tau=0.005,
            critic_target_update_freq=2,
            encoder_type='pixel',
            encoder_feature_dim=50,
            encoder_lr=1e-3,
            encoder_tau=0.005,
            decoder_type='pixel',
            decoder_lr=1e-3,
            decoder_update_freq=1,
            decoder_weight_lambda=0.0,
            transition_model_type='deterministic',
            num_layers=4,
            num_filters=32
    ):
        self.reconstruction = False
        if decoder_type == 'reconstruction':
            decoder_type = 'pixel'
            self.reconstruction = True
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_type = decoder_type
        self.best_reward = -1e10
        # 属性分配
        self.gamma = 0.99  # 折扣因子
        self.sigma = 0.1  # 高斯噪声的标准差，均值设为0
        self.tau =  0.005  # 目标网络的软更新参数

        #
        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim,
            num_layers, num_filters, encoder_stride
        ).to(device)
        self.actor_target = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim,
            num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride
        ).to(device)
        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.transition_model = make_transition_model(
            transition_model_type, encoder_feature_dim, action_shape
        ).to(device)

        self.reward_decoder = nn.Sequential(
            nn.Linear(encoder_feature_dim + action_shape[0], 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)).to(device)

        decoder_params = list(self.transition_model.parameters()) + list(self.reward_decoder.parameters())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        #
        self.decoder = None
        self.decoder_optimizer = torch.optim.Adam(
            decoder_params,
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )

        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=encoder_lr
        )

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    # 软更新, 意思是每次learn的时候更新部分参数
    def soft_update(self, net, target_net):
        # 获取训练网络和目标网络需要更新的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 训练网络的参数更新要综合考虑目标网络和训练网络
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            action = self.actor(obs)
            return action.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            action = self.actor(obs)
            return action.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, rewards, next_obs, not_done, L, step):
        # 价值目标网络获取下一时刻的每个动作价值[b,n_states]-->[b,n_actors]
        next_q_values = self.actor_target(next_obs).detach()
        # 策略目标网络获取下一时刻状态选出的动作价值 [b,n_states+n_actions]-->[b,1]
        next_val = self.critic_target.forward(next_obs, next_q_values).detach()

        # 当前时刻的动作价值的目标值 [b,1]
        q_targets = rewards + self.gamma * next_val * not_done

        # print("actions:", actions.shape)
        # print("states:", states.shape)
        # print("rewards:", rewards.shape)
        # print("dones:", dones.shape)

        # 当前时刻动作价值的预测值 [b,n_states+n_actions]-->[b,1]
        q_values = self.critic(obs, action, detach_encoder=False)

        # 预测值和目标值之间的均方差损失
        critic_loss = torch.mean(F.mse_loss(q_values, q_targets))
        L.log('train_critic/loss', critic_loss, step)

        # 价值网络梯度
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        # 当前状态的每个动作的价值 [b, n_actions]
        actor_q_values = self.actor(obs, detach_encoder=True)
        L.log('train_actor/actor_q_values', actor_q_values.mean(), step)
        # 当前状态选出的动作价值 [b,1]
        score = self.critic(obs, actor_q_values, detach_encoder=True)
        # 计算损失
        actor_loss = -torch.sum(score)
        L.log('train_actor/actor_loss', actor_loss.mean(), step)
        # 策略网络梯度
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_transition_reward_model(self, obs, action, next_obs, reward, L, step):
        h = self.critic.encoder(obs)
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        next_h = self.critic.encoder(next_obs)
        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        L.log('train_ae/transition_loss', loss, step)

        pred_next_reward = self.reward_decoder(torch.cat([h, action], dim=1))
        reward_loss = F.mse_loss(pred_next_reward, reward)
        total_loss = loss + reward_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def update_decoder(self, obs, action, target_obs, L, step):  # uses transition model
        # image might be stacked, just grab the first 3 (rgb)!
        assert target_obs.dim() == 4
        target_obs = target_obs[:, :3, :, :]

        h = self.critic.encoder(obs)
        if not self.reconstruction:
            next_h = self.transition_model.sample_prediction(torch.cat([h, action], dim=1))
            if target_obs.dim() == 4:
                # preprocess images to be in [-0.5, 0.5] range
                target_obs = utils.preprocess_obs(target_obs)
            rec_obs = self.decoder(next_h)
            loss = F.mse_loss(target_obs, rec_obs)
        else:
            rec_obs = self.decoder(h)
            loss = F.mse_loss(obs, rec_obs)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        L.log('train_ae/ae_loss', loss, step)

        self.decoder.log(L, step, log_freq=LOG_FREQ)

    def update(self, replay_buffer, L, step):
        obs, action, _, reward, next_obs, not_done = replay_buffer.sample()

        L.log('train/batch_reward', reward.mean(), step)

        self.update_actor(obs, L, step)
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        self.update_transition_reward_model(obs, action, next_obs, reward, L, step)

        if step % self.critic_target_update_freq == 0:
            # 软更新策略网络的参数
            self.soft_update(self.actor, self.actor_target)
            # 软更新价值网络的参数
            self.soft_update(self.critic, self.critic_target)

        if self.decoder is not None and step % self.decoder_update_freq == 0:  # decoder_type is pixel
            self.update_decoder(obs, action, next_obs, L, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        if self.decoder is not None:
            torch.save(
                self.decoder.state_dict(),
                '%s/decoder_%s.pt' % (model_dir, step)
            )

    def save_best(self, model_dir, episode_reward):
        save_best = False
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            save_best = True
        if save_best:
            torch.save(
                self.actor.state_dict(), '%s/actor_best.pt' % (model_dir)
            )
            torch.save(
                self.critic.state_dict(), '%s/critic_best.pt' % (model_dir)
            )
            if self.decoder is not None:
                torch.save(
                    self.decoder.state_dict(),
                    '%s/decoder_best.pt' % (model_dir)
                )
            print('----------------save best model------------------')

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        if self.decoder is not None:
            self.decoder.load_state_dict(
                torch.load('%s/decoder_%s.pt' % (model_dir, step))
            )

    def load_best(self, model_dir):
        self.actor.load_state_dict(
            torch.load('%s/actor_best.pt' % (model_dir)), strict=False
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_best.pt' % (model_dir)), strict=False
        )
        self.critic_target.load_state_dict(
            torch.load('%s/critic_best.pt' % (model_dir)), strict=False
        )