# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import utils
from agent.td3_ae import Actor, Critic, weight_init
from agent.transition_model import make_transition_model
from agent.decoder import make_decoder


class DeepMDPAgent_TD3(object):
    """Baseline algorithm with transition model and various decoder types."""

    def __init__(
            self,
            obs_shape,
            action_shape,
            device,
            hidden_dim=256,
            actor_lr=1e-3,
            actor_beta=0.9,
            actor_log_std_min=-10,
            actor_log_std_max=2,
            encoder_stride=2,
            critic_lr=1e-3,
            critic_beta=0.9,
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
        self.encoder_tau = encoder_tau
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_type = decoder_type
        self.best_reward = -1e10

        # Whether the Env has dead state. True for Env like BipedalWalkerHardcore-v3, CartPole-v0. False for Env like Pendulum-v0
        self.env_with_Dead = True
        self.action_dim = action_shape[0]
        self.max_action = 1  # carla是1
        self.gamma = 0.99
        self.policy_noise = 0.2 * self.max_action
        self.noise_clip = 0.5 * self.max_action
        self.tau = 0.005
        self.delay_counter = -1
        self.delay_freq = 1

        #
        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, encoder_stride, device
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )
        self.actor_target = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, encoder_stride, device
        ).to(device)
        #
        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride, device
        ).to(device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )
        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride, device
        ).to(device)

        #
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
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

        self.decoder = None
        if decoder_type == 'pixel':
            # create decoder
            self.decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(device)
            self.decoder.apply(weight_init)
            decoder_params += list(self.decoder.parameters())

        self.decoder_optimizer = torch.optim.Adam(
            decoder_params,
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )

        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=encoder_lr
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            a = self.actor(
                obs, detach_encoder=True
            )
        return a.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            a = self.actor(
                obs, detach_encoder=True
            )
        return a.cpu().data.numpy().flatten()

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

    def update(self, replay_buffer, L, step):
        self.delay_counter += 1
        s, a, _, r, s_prime, not_done = replay_buffer.sample()
        #
        self.update_transition_reward_model(s, a, s_prime, r, L, step)
        #
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            smoothed_target_a = (
                    # Noisy on target action
                    self.actor_target(s_prime) + noise
                    # self.actor_target(s_prime, detach_encoder=True) + noise  # wait
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(s_prime, smoothed_target_a)
            # target_Q1, target_Q2 = self.critic_target(s_prime, smoothed_target_a, detach_encoder=True)  # wait
            target_Q = torch.min(target_Q1, target_Q2)

            '''DEAD OR NOT'''
            if self.env_with_Dead:
                target_Q = r + not_done * self.gamma * target_Q  # env with dead
            else:
                target_Q = r + self.gamma * target_Q  # env without dead

        # Get current Q estimates
        # current_Q1, current_Q2 = self.critic(s, a, detach_encoder=False)
        current_Q1, current_Q2 = self.critic(s, a)

        # Compute critic loss
        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the q_critic
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.delay_counter == self.delay_freq:
            # Update Actor
            # a_loss = -self.critic.get_Q1(s, self.actor(s, detach_encoder=True)).mean()  # wait
            a_loss = -self.critic.get_Q1(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.delay_counter = -1

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
