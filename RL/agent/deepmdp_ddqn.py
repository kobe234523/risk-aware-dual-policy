# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils
from agent.ddqn_ae import DuelingDQnetwork, LOG_FREQ
from agent.transition_model import make_transition_model


class DeepMDPAgent_DDQN(object):
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
        self.gamma = 0.99
        self.lr = 0.001
        self.epsilon = 1.0
        self.epsilon_end = 0.05
        self.action_space = [i for i in range(action_shape[0])]
        self.q_network_eval = DuelingDQnetwork(obs_shape, action_shape, hidden_dim, encoder_type, encoder_feature_dim,
                                               num_layers, num_filters, encoder_stride, device, self.lr).to(device)
        self.q_network_target = DuelingDQnetwork(obs_shape, action_shape, hidden_dim, encoder_type, encoder_feature_dim,
                                                 num_layers, num_filters, encoder_stride, device, self.lr).to(device)
        # tie encoders between actor and critic
        self.q_network_target.encoder.copy_conv_weights_from(self.q_network_eval.encoder)

        #
        self.transition_model = make_transition_model(
            transition_model_type, encoder_feature_dim, [1]
        ).to(device)

        self.reward_decoder = nn.Sequential(
            nn.Linear(encoder_feature_dim + 1, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)).to(device)

        decoder_params = list(self.transition_model.parameters()) + list(self.reward_decoder.parameters())

        #
        self.decoder = None
        self.decoder_optimizer = torch.optim.Adam(
            decoder_params,
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )

        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(
            self.q_network_eval.encoder.parameters(), lr=encoder_lr
        )

        self.train()

    def train(self, training=True):
        self.training = training
        self.q_network_eval.train(training)
        self.q_network_target.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    def select_action(self, obs):  # 评估用
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            advantage = self.q_network_eval.forward(obs)
            action = torch.argmax(advantage).item()
        return action

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            if np.random.random() > self.epsilon:
                advantage = self.q_network_eval.forward(obs)
                action = torch.argmax(advantage).item()
            else:
                action = np.random.choice(self.action_space)
            return action

    def decrese_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= 0.0001  # 10000
        else:
            self.epsilon = self.epsilon_end

    def update_transition_reward_model(self, obs, action, next_obs, reward, L, step):
        h = self.q_network_eval.encoder(obs)
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        next_h = self.q_network_eval.encoder(next_obs)
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

    def update(self, replay_buffer, L, step):
        observation, action, _, reward, new_observation, not_done = replay_buffer.sample()

        # can be no used
        # self.update_transition_reward_model(observation, action, new_observation, reward, L, step)

        batch_idx = np.arange(replay_buffer.get_batch_size())
        action = action.to(torch.long).squeeze()
        reward = reward.squeeze()
        done = (1 - not_done).to(torch.bool).squeeze()

        with torch.no_grad():
            q_ = self.q_network_eval.forward(new_observation)
            # print("q_:", q_)
            next_actions = torch.argmax(q_, dim=-1)
            # print("next_actions:", next_actions)
            q_ = self.q_network_target.forward(new_observation)
            q_[done] = 0.0
            target = reward + self.gamma * q_[batch_idx, next_actions]

        q = self.q_network_eval.forward(observation)[batch_idx, action]
        loss = F.mse_loss(q, target.detach())
        L.log('train/loss', loss.mean(), step)

        self.q_network_eval.optimizer.zero_grad()
        loss.backward()
        self.q_network_eval.optimizer.step()

        #
        if step % 5 == 0:
            tau = 0.005
            for q_target_params, q_eval_params in zip(self.q_network_target.parameters(), self.q_network_eval.parameters()):
                q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)
        #
        self.decrese_epsilon()

    def save(self, model_dir, step):
        torch.save(
            self.q_network_eval.state_dict(), '%s/q_network_eval_%s.pt' % (model_dir, step)
        )

    def save_best(self, model_dir, episode_reward):
        save_best = False
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            save_best = True
        if save_best:
            torch.save(
                self.q_network_eval.state_dict(), '%s/q_network_eval_best.pt' % (model_dir)
            )
            print('----------------save best model------------------')

    def load(self, model_dir, step):
        self.q_network_eval.load_state_dict(
            torch.load('%s/q_network_eval_%s.pt' % (model_dir, step))
        )

    def load_best(self, model_dir):
        self.q_network_eval.load_state_dict(
            torch.load('%s/q_network_eval_best.pt' % (model_dir)), strict=False
        )
