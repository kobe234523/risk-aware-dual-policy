# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys

import numpy as np
import torch
import argparse
import os
import time
import json

from utils import utils
from utils.logger import Logger
from utils.video import VideoRecorder

from agent.deepmdp_ppo import DeepMDPAgent_PPO
from carla_env.carla_env import CarlaEnv

# Import TCP
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TCP-main-attention')))
from TCP.model import TCP
from TCP.config import GlobalConfig
from torchvision import transforms as T
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='carla', choices=['carla'])
    parser.add_argument('--scenarios', default='highway', choices=['highway', 'ghost_static'])
    parser.add_argument('--tcp_ckpt', default='../TCP-main-attention/model_best.ckpt', type=str)
    parser.add_argument('--tcp_use_ggca', default=False, action='store_true')
    parser.add_argument('--tcp_ggca_groups', default=4, type=int)
    parser.add_argument('--weather', default='ClearNoon', type=str, 
                        choices=['ClearNoon', 'ClearSunset', 'CloudyNoon', 'CloudySunset', 
                                 'WetNoon', 'WetSunset', 'MidRainyNoon', 'MidRainSunset', 
                                 'WetCloudyNoon', 'WetCloudySunset', 'HardRainNoon', 'HardRainSunset', 
                                 'SoftRainNoon', 'SoftRainSunset'])
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=3, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--img_source', default=None, type=str, choices=['color', 'noise', 'images', 'video', 'none'])
    # train
    parser.add_argument('--agent', default='ppo', type=str, choices=['ppo'])
    parser.add_argument('--init_steps', default=100, type=int)  #
    parser.add_argument('--num_train_steps', default=30000, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
    parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
    parser.add_argument('--load_encoder', default=False, type=bool)
    parser.add_argument('--encoder_path', default="", type=str)
    parser.add_argument('--model_path', default="", type=str)

    #
    parser.add_argument('--eval_freq', default=1000, type=int)  # TODO: master had 10000
    parser.add_argument('--num_eval_episodes', default=5, type=int)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--action_std_decay_freq', default=2000, type=int)
    parser.add_argument('--action_std_decay_rate', default=0.05, type=int)
    parser.add_argument('--min_action_std', default=0.05, type=int)
    parser.add_argument('--action_std_init', default=1.0, type=int)

    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixelCarla', type=str, choices=['pixel', 'pixelCarla', 'pixelCarlaSE', 'pixelResNet18', 'identity'])
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--encoder_stride', default=1, type=int)
    parser.add_argument('--decoder_type', default='identity', type=str, choices=['pixel', 'identity'])
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_weight_lambda', default=0.0000001, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='./log', type=str)
    parser.add_argument('--save_tb', default=True, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=True, action='store_true')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--transition_model_type', default='', type=str,
                        choices=['', 'deterministic', 'probabilistic', 'ensemble'])
    parser.add_argument('--port', default=2002, type=int)
    parser.add_argument('--trafficManagerPort', default=2050, type=int)
    args = parser.parse_args()
    return args


def evaluate(env, agent, tcp_net, tcp_config, video, num_episodes, args, do_carla_metrics=False, res_dir="", model_dir=""):
    reason_each_episode_ended = []
    distance_driven_each_episode = []
    crash_intensity = 0.
    steer = 0.
    brake = 0.
    count = 0
    agent.load_best(model_dir)
    for i in range(num_episodes):
        dist_driven_this_episode = 0.

        obs = env.reset()
        video.init(enabled=True)
        done = False
        episode_reward = 0
        while not done:
            action_traj, action_ctrl = get_tcp_actions(tcp_net, env, tcp_config)

            rl_weight_raw = agent.select_action(obs)
            alpha = float((rl_weight_raw[0] + 1.0) * 0.5)
            alpha = float(np.clip(alpha, 0.0, 1.0))

            final_action = alpha * action_traj + (1.0 - alpha) * action_ctrl
            final_action = np.clip(final_action, -1.0, 1.0)

            obs, reward, done, info = env.step(final_action)
            if do_carla_metrics:
                if info is not None:
                    dist_driven_this_episode += info.get('distance', 0)
                    crash_intensity += info.get('crash_intensity', 0)
                    steer += abs(info.get('steer', 0))
                    brake += info.get('brake', 0)
                    count += 1
                else:
                    print("Warning: 'info' is None, skipping metrics update.")

            video.record(env)
            episode_reward += reward

        if do_carla_metrics:
            distance_driven_each_episode.append(dist_driven_this_episode)
            eval_log_filepath = os.path.join(res_dir, "eval_log.txt")
            eval_log_txt_formatter = "{step},{distance_driven_each_episode},{episode_reward}\n"
            to_write = eval_log_txt_formatter.format(step=i,
                                                     distance_driven_each_episode=dist_driven_this_episode,
                                                     episode_reward=episode_reward)

            with open(eval_log_filepath, "a") as f:
                f.write(to_write)

        video.save('%d.mp4' % i)

    if do_carla_metrics:
        print('METRICS--------------------------')
        print("distance_driven_each_episode: {}".format(distance_driven_each_episode))
        print('crash_intensity: {}'.format(crash_intensity / num_episodes))
        print('steer: {}'.format(steer / count))
        print('brake: {}'.format(brake / count))
        print('---------------------------------')
        sys.exit()


def make_agent(obs_shape, action_shape, args, device):
    # take deepmdp as base
    agent = DeepMDPAgent_PPO(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        hidden_dim=args.hidden_dim,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_max=args.actor_log_std_max,
        encoder_stride=args.encoder_stride,
        encoder_type=args.encoder_type,
        encoder_feature_dim=args.encoder_feature_dim,
        encoder_lr=args.encoder_lr,
        encoder_tau=args.encoder_tau,
        decoder_type=args.decoder_type,
        decoder_lr=args.decoder_lr,
        decoder_update_freq=args.decoder_update_freq,
        decoder_weight_lambda=args.decoder_weight_lambda,
        transition_model_type=args.transition_model_type,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
        action_std_init=args.action_std_init
    )
    # resume
    if args.load_encoder:
        # model_dict = agent.old_policy.encoder.state_dict()
        # encoder_dict = torch.load(args.encoder_path)
        # encoder_dict = {k[8:]: v for k, v in encoder_dict.items() if 'encoder.' in k}  # hack to remove encoder. string
        # agent.old_policy.encoder.load_state_dict(encoder_dict)
        agent.load_best(args.model_path)

    return agent


def load_tcp(ckpt_path, args):
    config = GlobalConfig(
        use_ggca=bool(args.tcp_use_ggca),
        ggca_groups=int(args.tcp_ggca_groups),
    )
    net = TCP(config)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt:
        ckpt = ckpt["state_dict"]
    new_state_dict = OrderedDict()
    for key, value in ckpt.items():
        new_key = key.replace("model.","")
        new_state_dict[new_key] = value
    net.load_state_dict(new_state_dict, strict=False)
    net.cuda()
    net.eval()
    return net, config

def get_tcp_actions(net, env, config):
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    rgb, speed, target_point, command = base_env.get_tcp_inputs()
    rgb = np.ascontiguousarray(rgb)

    _im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    rgb_tensor = _im_transform(rgb).unsqueeze(0).cuda().float()

    cmd_one_hot = [0] * 6
    cmd_idx = command - 1
    if 0 <= cmd_idx < 6:
        cmd_one_hot[cmd_idx] = 1

    speed_tensor = torch.FloatTensor([speed / 12.0]).view(1, 1).cuda()
    tp_tensor = torch.FloatTensor([target_point]).cuda()
    cmd_tensor = torch.FloatTensor([cmd_one_hot]).cuda()
    state = torch.cat([speed_tensor, tp_tensor, cmd_tensor], 1)

    with torch.no_grad():
        pred = net(rgb_tensor, state, tp_tensor)

        steer_ctrl, throttle_ctrl, brake_ctrl, _ = net.process_action(pred, command, speed_tensor, tp_tensor)
        steer_traj, throttle_traj, brake_traj, _ = net.control_pid(pred['pred_wp'], speed_tensor, tp_tensor)

    steer_ctrl = float(np.asarray(steer_ctrl).item())
    throttle_ctrl = float(np.asarray(throttle_ctrl).item())
    brake_ctrl = float(np.asarray(brake_ctrl).item())
    steer_traj = float(np.asarray(steer_traj).item())
    throttle_traj = float(np.asarray(throttle_traj).item())
    brake_traj = float(np.asarray(brake_traj).item())

    throttle_brake_ctrl = throttle_ctrl if throttle_ctrl > 0.0 else -brake_ctrl
    throttle_brake_traj = throttle_traj if throttle_traj > 0.0 else -brake_traj

    action_ctrl = np.array([steer_ctrl, throttle_brake_ctrl], dtype=np.float32)
    action_traj = np.array([steer_traj, throttle_brake_traj], dtype=np.float32)

    return action_traj, action_ctrl

def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    env = CarlaEnv(
        render_display=args.render,  # for local debugging only
        display_text=args.render,  # for local debugging only
        changing_weather_speed=0.1,  # [0, +inf)
        rl_image_size=args.image_size,
        max_episode_steps=2000,
        frame_skip=args.action_repeat,
        port=args.port,
        trafficManagerPort=args.trafficManagerPort,
        scenarios=args.scenarios,
        algorithm=args.agent,
        weather=args.weather
    )
    # TODO: implement env.seed(args.seed) ?
    eval_env = env

    # stack several consecutive frames together
    if args.encoder_type.startswith('pixel'):
        env = utils.FrameStack(env, k=args.frame_stack)
        eval_env = utils.FrameStack(eval_env, k=args.frame_stack)

    # add time
    work_dir = args.work_dir + '/' + args.scenarios + "_" + args.agent + "_seed_{}_".format(args.seed) + time.strftime(
        "%Y-%m-%d-%H-%M-%S")
    utils.make_dir(work_dir)
    video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(work_dir, 'buffer'))
    res_dir = utils.make_dir(os.path.join(work_dir, 'res_dir'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=(1,),
        args=args,
        device=device
    )

    L = Logger(work_dir, use_tb=args.save_tb)

    tcp_net, tcp_config = load_tcp(args.tcp_ckpt, args)

    # train and evaluate
    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(args.num_train_steps):
        if done:
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            if args.save_model:
                # agent.save(model_dir, step)
                # print('----------------save model-----------------')
                agent.save_best(model_dir, episode_reward)

            # evaluate agent periodically
            # if episode % args.eval_freq == 0 and episode != 0:
            #     print('----------------start eval-----------------')
            #     L.log('eval/episode', episode, step)
            #     evaluate(eval_env, agent, video, args.num_eval_episodes, L, step, do_carla_metrics=True,
            #              res_dir=res_dir)
            #     # if args.save_buffer:
            #     #     replay_buffer.save(buffer_dir)

            L.log('train/episode_reward', episode_reward, step)

            train_log_filepath = os.path.join(res_dir, "train_log.txt")
            train_log_txt_formatter = "{step}:{episode_reward}\n"
            to_write = train_log_txt_formatter.format(step=step,
                                                      episode_reward=episode_reward)
            with open(train_log_filepath, "a") as f:
                f.write(to_write)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            reward = 0

            L.log('train/episode', episode, step)

        action_traj, action_ctrl = get_tcp_actions(tcp_net, env, tcp_config)

        rl_weight_raw, log_prob = agent.sample_action(obs)
        alpha = float((rl_weight_raw[0] + 1.0) * 0.5)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        final_action = alpha * action_traj + (1.0 - alpha) * action_ctrl
        final_action = np.clip(final_action, -1.0, 1.0)

        next_obs, reward, done, _ = env.step(final_action)
        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )

        episode_reward += reward
        agent.add_data(obs, next_obs, reward, done_bool, rl_weight_raw, log_prob)

        # run training update，
        if step != 0 and done is True:
            agent.update(L, step)

        # update current obs
        obs = next_obs
        episode_step += 1

    print('----------------start eval-----------------')
    evaluate(env, agent, tcp_net, tcp_config, video, args.num_eval_episodes, args, do_carla_metrics=True, res_dir=res_dir,
             model_dir=model_dir)


if __name__ == '__main__':
    main()
