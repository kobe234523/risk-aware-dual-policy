import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.sac_ae import Actor

class ImitationAgent(object):
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
        num_layers=4,
        num_filters=32,
        encoder_stride=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        **kwargs
    ):
        self.device = device
        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, encoder_stride
        ).to(device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )
        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()
    
    def sample_action(self, obs):
        # For compatibility with other agents
        return self.select_action(obs)

    def update(self, replay_buffer, logger, step):
        obs, action, _, _, _, _ = replay_buffer.sample()

        mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
        loss = F.mse_loss(mu, action)

        logger.log('train/bc_loss', loss, step)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
    
    def save_best(self, model_dir, episode_reward):
        # Simplified save best
        torch.save(
            self.actor.state_dict(), '%s/actor_best.pt' % (model_dir)
        )
        return True
    
    def load_best(self, model_dir):
        self.actor.load_state_dict(
            torch.load('%s/actor_best.pt' % (model_dir))
        )