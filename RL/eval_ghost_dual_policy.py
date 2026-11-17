import argparse
import csv
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
from PIL import Image

from utils import utils
from utils.video import VideoRecorder
from carla_env.carla_env import CarlaEnv
from train_carla_off_policy import make_agent as make_off_policy_agent
from train_carla_on_policy import get_tcp_actions, load_tcp
from risk_learning.model import encoder_signature, load_head
from risk_learning.reference import ReferenceConfig, reference_from_env
from risk_learning.tcp_adapter import tcp_actions_and_features
from risk_learning.data import save_episode
from risk_learning.runtime import RISK_METHODS, validate_options, compute_online_risk


WEATHERS = [
    "ClearNoon", "ClearSunset", "CloudyNoon", "CloudySunset",
    "WetNoon", "WetSunset", "MidRainyNoon", "MidRainSunset",
    "WetCloudyNoon", "WetCloudySunset", "HardRainNoon",
    "HardRainSunset", "SoftRainNoon", "SoftRainSunset",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified ghost-probe evaluation for TCP, SAC, and risk-aware dual policy."
    )

    parser.add_argument(
        "--method",
        default="ours",
        choices=[
            "tcp",
            "tcp_rs",
            "sac",
            "rule",
            "fixed_switch",
            "risk_switch",
            "fixed_fusion",
            "risk_fusion",
            "ours",
            "ours_rs",
        ],
        help="Policy to evaluate. ours is TCP-GGCA + SAC risk switch.",
    )
    parser.add_argument("--scenarios", default="ghost_static", choices=["ghost_static", "parking"])
    parser.add_argument("--weather", default="ClearNoon", choices=WEATHERS)
    parser.add_argument("--min_walkers", default=1, type=int)
    parser.add_argument("--max_walkers", default=3, type=int)
    parser.add_argument("--walker_trigger_ttc_min", default=1.8, type=float)
    parser.add_argument("--walker_trigger_ttc_max", default=3.0, type=float)
    parser.add_argument("--walker_cross_probability", default=1.0, type=float)
    parser.add_argument("--walker_speed_min", default=1.2, type=float)
    parser.add_argument("--walker_speed_max", default=1.8, type=float)
    parser.add_argument("--walker_direction", default="top_to_bottom", choices=["top_to_bottom", "bottom_to_top", "random"])
    parser.add_argument("--occlusion_level", default="medium", choices=["weak", "medium", "strong"])
    parser.add_argument("--initial_speed", default=None, type=float)
    parser.add_argument("--num_eval_episodes", default=20, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--work_dir", default="./log/ghost_dual_eval", type=str)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument(
        "--save_front_frames",
        action="store_true",
        help="Save 900x256 TCP front-view RGB frames for paper visualization.",
    )
    parser.add_argument(
        "--front_frame_every",
        default=1,
        type=int,
        help="Save one front-view frame every N policy steps when --save_front_frames is enabled.",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--live_status", action="store_true", help="Print a one-line live HUD in the terminal.")
    parser.add_argument("--status_every", default=1, type=int, help="Refresh terminal HUD every N policy steps.")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--trafficManagerPort", default=19191, type=int)

    parser.add_argument("--image_size", default=84, type=int)
    parser.add_argument("--action_repeat", default=3, type=int)
    parser.add_argument("--frame_stack", default=3, type=int)
    parser.add_argument("--max_episode_steps", default=1000, type=int)

    parser.add_argument("--tcp_ckpt", default="../TCP-main-attention/model_best.ckpt", type=str)
    parser.add_argument("--tcp_use_ggca", action="store_true")
    parser.add_argument("--tcp_ggca_groups", default=4, type=int)
    parser.add_argument("--tcp_action", default="mix", choices=["traj", "ctrl", "mix"])
    parser.add_argument("--tcp_mix_alpha", default=0.5, type=float)
    parser.add_argument("--tcp_steer_sign", default=1.0, type=float,
                        help="Multiply TCP steering by this value. Use -1 if the TCP steering sign is inverted.")
    parser.add_argument("--tcp_steer_clip", default=1.0, type=float,
                        help="Clip absolute TCP steering before fusion/switching.")

    parser.add_argument("--rl_model_dir", default="", type=str)
    parser.add_argument("--agent", default="sac", choices=["sac", "ddpg", "td3", "ddqn"])
    parser.add_argument("--encoder_type", default="pixelCarla", choices=["pixel", "pixelCarla", "pixelCarlaSE", "identity"])
    parser.add_argument("--encoder_feature_dim", default=50, type=int)
    parser.add_argument("--encoder_lr", default=1e-3, type=float)
    parser.add_argument("--encoder_tau", default=0.05, type=float)
    parser.add_argument("--encoder_stride", default=1, type=int)
    parser.add_argument("--decoder_type", default="identity", choices=["pixel", "identity"])
    parser.add_argument("--decoder_lr", default=1e-3, type=float)
    parser.add_argument("--decoder_update_freq", default=1, type=int)
    parser.add_argument("--decoder_weight_lambda", default=1e-7, type=float)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--num_filters", default=32, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--init_temperature", default=0.1, type=float)
    parser.add_argument("--alpha_lr", default=1e-4, type=float)
    parser.add_argument("--alpha_beta", default=0.5, type=float)
    parser.add_argument("--actor_lr", default=1e-3, type=float)
    parser.add_argument("--actor_beta", default=0.9, type=float)
    parser.add_argument("--actor_log_std_min", default=-10, type=float)
    parser.add_argument("--actor_log_std_max", default=2, type=float)
    parser.add_argument("--actor_update_freq", default=2, type=int)
    parser.add_argument("--critic_lr", default=1e-3, type=float)
    parser.add_argument("--critic_beta", default=0.9, type=float)
    parser.add_argument("--critic_tau", default=0.01, type=float)
    parser.add_argument("--critic_target_update_freq", default=2, type=int)
    parser.add_argument("--transition_model_type", default="", choices=["", "deterministic", "probabilistic", "ensemble"])
    parser.add_argument("--load_encoder", default=False, type=bool)
    parser.add_argument("--encoder_path", default="", type=str)
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--bc_actor_path", default="", type=str)

    parser.add_argument("--delta", default=0.5, type=float, help="Risk threshold for switch/fusion.")
    parser.add_argument("--fixed_switch_frame", default=120, type=int)
    parser.add_argument("--fixed_fusion_alpha", default=0.5, type=float, help="Final action = alpha * RL + (1-alpha) * TCP.")
    parser.add_argument("--risk_distance", default=15.0, type=float)
    parser.add_argument("--risk_lateral", default=2.0, type=float)
    parser.add_argument("--ttc_threshold", default=3.0, type=float)
    parser.add_argument("--speed_norm", default=10.0, type=float)
    parser.add_argument("--risk_source", default="walker", choices=["walker", "conflict"])
    parser.add_argument("--rule_target_speed", default=25.0, type=float,
                        help="Target speed in km/h for --method rule.")
    parser.add_argument("--rule_hazard_distance", default=20.0, type=float,
                        help="Longitudinal pedestrian hazard distance in meters for --method rule.")
    parser.add_argument("--rule_hazard_lateral", default=2.5, type=float,
                        help="Lateral pedestrian hazard range in meters for --method rule.")
    parser.add_argument("--rule_throttle", default=0.55, type=float,
                        help="Cruising throttle command for --method rule.")
    parser.add_argument("--rule_avoid_steer", default=0.0, type=float,
                        help="Optional small steering-away command for --method rule. Use 0.0 for braking-only expert.")

    parser.add_argument("--risk_estimator", default="learned", choices=["learned", "legacy", "reference"],
                        help="learned: observable-only trained head; legacy/reference are explicitly privileged baselines")
    parser.add_argument("--risk_head_ckpt", default="", help="Trusted risk_head.pt produced by train_risk_head.py")
    parser.add_argument("--collect_risk_data", action="store_true", help="Save frozen features and privileged training references")
    parser.add_argument("--risk_scene_group", default="", help="Stable route/layout group; do not vary it by seed or weather")
    parser.add_argument("--reference_distance", default=15.0, type=float)
    parser.add_argument("--reference_lateral", default=3.0, type=float)
    parser.add_argument("--reference_time", default=3.0, type=float)
    parser.add_argument("--reference_alpha", default=0.5, type=float)
    parser.add_argument("--reference_epsilon", default=0.1, type=float)
    return parser.parse_args()


def get_base_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def split_action(action):
    steer = float(action[0])
    throttle_brake = float(action[1])
    throttle = max(0.0, throttle_brake)
    brake = max(0.0, -throttle_brake)
    return steer, throttle, brake


def make_tcp_action(action_traj, action_ctrl, args):
    if args.tcp_action == "traj":
        return action_traj
    if args.tcp_action == "ctrl":
        return action_ctrl
    alpha = float(np.clip(args.tcp_mix_alpha, 0.0, 1.0))
    return (1.0 - alpha) * action_traj + alpha * action_ctrl


def actor_location_xy(actor):
    loc = actor.get_location()
    return np.array([float(loc.x), float(loc.y)], dtype=np.float32)


def compute_risk(env, args):
    base_env = get_base_env(env)
    info = {
        "rho": 0.0,
        "ttc": -1.0,
        "risk_longitudinal": 0.0,
        "risk_lateral": 0.0,
        "risk_distance": -1.0,
        "risk_source": args.risk_source,
    }

    vehicle = getattr(base_env, "vehicle", None)
    if vehicle is None:
        return info

    try:
        ego_xy = actor_location_xy(vehicle)
        transform = vehicle.get_transform()
        fwd = transform.get_forward_vector()
        fwd_xy = np.array([float(fwd.x), float(fwd.y)], dtype=np.float32)
        fwd_norm = np.linalg.norm(fwd_xy)
        if fwd_norm > 1e-6:
            fwd_xy = fwd_xy / fwd_norm
        speed_vec = vehicle.get_velocity()
        speed = float(math.sqrt(speed_vec.x ** 2 + speed_vec.y ** 2))

        target_xy = None
        if args.risk_source == "walker":
            walker = getattr(base_env, "walker_probe", None)
            if walker is not None:
                target_xy = actor_location_xy(walker)

        if target_xy is None:
            if getattr(base_env, "scenarios", "") == "ghost_static":
                target_xy = np.array([-76.393974, float(ego_xy[1])], dtype=np.float32)
            elif getattr(base_env, "goal_location", None) is not None:
                goal = base_env.goal_location
                target_xy = np.array([float(goal.x), float(goal.y)], dtype=np.float32)

        if target_xy is None:
            return info

        rel = target_xy - ego_xy
        long_dist = float(np.dot(rel, fwd_xy))
        dist = float(np.linalg.norm(rel))
        lat_sq = max(0.0, dist ** 2 - long_dist ** 2)
        lat_dist = float(math.sqrt(lat_sq))

        ttc = long_dist / max(speed, 1e-3) if long_dist > 0.0 else float("inf")
        dist_risk = np.clip((args.risk_distance - max(long_dist, 0.0)) / args.risk_distance, 0.0, 1.0)
        lat_risk = np.clip((args.risk_lateral - lat_dist) / args.risk_lateral, 0.0, 1.0)
        ttc_risk = np.clip((args.ttc_threshold - min(ttc, args.ttc_threshold)) / args.ttc_threshold, 0.0, 1.0)
        speed_risk = np.clip(speed / args.speed_norm, 0.0, 1.0)
        zone_risk = 1.0 if 0.0 < long_dist < args.risk_distance and lat_dist < args.risk_lateral else 0.0

        rho = (
            0.30 * dist_risk
            + 0.25 * lat_risk
            + 0.25 * ttc_risk
            + 0.10 * speed_risk
            + 0.10 * zone_risk
        )

        info.update({
            "rho": float(np.clip(rho, 0.0, 1.0)),
            "ttc": -1.0 if math.isinf(ttc) else float(ttc),
            "risk_longitudinal": long_dist,
            "risk_lateral": lat_dist,
            "risk_distance": dist,
        })
    except Exception as exc:
        info["risk_error"] = repr(exc)

    return info


def find_walker_actors(base_env):
    """Return live pedestrian actors exposed by the CARLA environment."""
    walkers = []
    for attr in ("walker_probe", "walker"):
        walker = getattr(base_env, attr, None)
        if walker is not None and getattr(walker, "is_alive", True):
            walkers.append(walker)

    for attr in ("walkers", "walker_actors"):
        value = getattr(base_env, attr, None)
        if value is None:
            continue
        try:
            iterable = list(value)
        except TypeError:
            iterable = [value]
        for actor in iterable:
            if actor is not None and getattr(actor, "is_alive", True):
                walkers.append(actor)

    for actor in getattr(base_env, "actor_list", []) or []:
        try:
            if "walker" in actor.type_id and getattr(actor, "is_alive", True):
                walkers.append(actor)
        except Exception:
            continue

    unique = []
    seen = set()
    for walker in walkers:
        actor_id = getattr(walker, "id", id(walker))
        if actor_id not in seen:
            unique.append(walker)
            seen.add(actor_id)
    return unique


def make_rule_action(env, args):
    """
    Simple rule-based expert for ghost-pedestrian visualization.

    The expert keeps a target speed in normal driving and applies strong braking
    when a pedestrian is in front of the ego vehicle within a longitudinal and
    lateral safety envelope. It is intended for paper frame capture, not as the
    proposed learning policy.
    """
    base_env = get_base_env(env)
    vehicle = getattr(base_env, "vehicle", None)
    if vehicle is None:
        return np.zeros(2, dtype=np.float32)

    steer = 0.0
    accel = 0.0
    hazard_detected = False
    closest_lat_dist = 0.0

    try:
        veh_loc = vehicle.get_location()
        transform = vehicle.get_transform()
        fwd = transform.get_forward_vector()
        right = transform.get_right_vector()

        for walker in find_walker_actors(base_env):
            walk_loc = walker.get_location()
            vec_x = float(walk_loc.x - veh_loc.x)
            vec_y = float(walk_loc.y - veh_loc.y)
            vec_z = float(walk_loc.z - veh_loc.z)
            long_dist = vec_x * float(fwd.x) + vec_y * float(fwd.y) + vec_z * float(fwd.z)
            lat_dist = vec_x * float(right.x) + vec_y * float(right.y) + vec_z * float(right.z)

            if (
                0.0 < long_dist < float(args.rule_hazard_distance)
                and abs(lat_dist) < float(args.rule_hazard_lateral)
            ):
                hazard_detected = True
                closest_lat_dist = lat_dist
                break

        if hazard_detected:
            avoid_steer = abs(float(args.rule_avoid_steer))
            if avoid_steer > 0.0:
                steer = -avoid_steer if closest_lat_dist >= 0.0 else avoid_steer
            accel = -1.0
        else:
            velocity = vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            if speed_kmh < float(args.rule_target_speed):
                accel = float(args.rule_throttle)
            else:
                accel = 0.0
    except Exception:
        steer = 0.0
        accel = 0.0

    return np.array([
        np.clip(steer, -1.0, 1.0),
        np.clip(accel, -1.0, 1.0),
    ], dtype=np.float32)


def choose_action(method, tcp_action, rl_action, rule_action, rho, frame_idx, args):
    takeover = False
    alpha = 0.0

    if method in ("tcp", "tcp_rs"):
        final_action = tcp_action
        selected = "tcp"
    elif method == "sac":
        final_action = rl_action
        selected = "rl"
        takeover = True
        alpha = 1.0
    elif method == "rule":
        final_action = rule_action
        selected = "rule"
    elif method == "fixed_switch":
        takeover = frame_idx >= args.fixed_switch_frame
        final_action = rl_action if takeover else tcp_action
        selected = "rl" if takeover else "tcp"
        alpha = 1.0 if takeover else 0.0
    elif method in ("risk_switch", "ours", "ours_rs"):
        takeover = rho >= args.delta
        final_action = rl_action if takeover else tcp_action
        selected = "rl" if takeover else "tcp"
        alpha = 1.0 if takeover else 0.0
    elif method == "fixed_fusion":
        alpha = float(np.clip(args.fixed_fusion_alpha, 0.0, 1.0))
        final_action = alpha * rl_action + (1.0 - alpha) * tcp_action
        selected = "fusion"
    elif method == "risk_fusion":
        alpha = float(np.clip((rho - args.delta) / max(1e-6, 1.0 - args.delta), 0.0, 1.0))
        final_action = alpha * rl_action + (1.0 - alpha) * tcp_action
        selected = "fusion"
        takeover = alpha > 0.0
    else:
        raise ValueError(f"Unsupported method: {method}")

    return np.clip(final_action, -1.0, 1.0).astype(np.float32), selected, takeover, alpha


def should_load_tcp(method):
    return method in {"tcp", "tcp_rs", "fixed_switch", "risk_switch", "fixed_fusion", "risk_fusion", "ours", "ours_rs"}


def should_load_rl(method):
    return method in {"sac", "fixed_switch", "risk_switch", "fixed_fusion", "risk_fusion", "ours", "ours_rs"}


def create_env(args):
    env = CarlaEnv(
        render_display=args.render,
        display_text=args.render,
        changing_weather_speed=0.1,
        rl_image_size=args.image_size,
        max_episode_steps=args.max_episode_steps,
        frame_skip=args.action_repeat,
        port=args.port,
        trafficManagerPort=args.trafficManagerPort,
        scenarios=args.scenarios,
        algorithm=args.agent,
        weather=args.weather,
        min_walkers=args.min_walkers,
        max_walkers=args.max_walkers,
        walker_trigger_ttc_min=args.walker_trigger_ttc_min,
        walker_trigger_ttc_max=args.walker_trigger_ttc_max,
        walker_cross_probability=args.walker_cross_probability,
        walker_speed_min=args.walker_speed_min,
        walker_speed_max=args.walker_speed_max,
        walker_direction=args.walker_direction,
        occlusion_level=args.occlusion_level,
        initial_speed=args.initial_speed,
    )
    if args.encoder_type.startswith("pixel"):
        env = utils.FrameStack(env, k=args.frame_stack)
    return env


def create_rl_agent(env, args, device):
    agent = make_off_policy_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device,
    )
    if not args.rl_model_dir:
        raise ValueError("--rl_model_dir is required for methods that use the RL policy.")
    agent.load_best(args.rl_model_dir)
    return agent


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def save_front_frame(front_root, episode, frame, rgb, metadata):
    """Save one TCP front-camera frame and its metadata for paper figures."""
    episode_dir = os.path.join(front_root, f"episode_{episode:03d}")
    rgb_dir = os.path.join(episode_dir, "rgb")
    meta_dir = os.path.join(episode_dir, "meta")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    image_path = os.path.abspath(os.path.join(rgb_dir, f"{frame:04d}.png"))
    meta_path = os.path.abspath(os.path.join(meta_dir, f"{frame:04d}.json"))
    Image.fromarray(np.asarray(rgb).astype(np.uint8)).save(image_path)
    # Bind the exact saved image to its pre-action inputs and issued command.
    metadata["front_rgb_path"] = image_path
    metadata["front_meta_path"] = meta_path
    with open(image_path, "rb") as image_file:
        metadata["front_rgb_sha256"] = hashlib.sha256(image_file.read()).hexdigest()
    write_json(meta_path, metadata)
    return image_path, meta_path


def update_window_caption(args, episode, frame, selected, rho, speed, dist_to_walker):
    if not args.render:
        return
    try:
        import pygame

        pygame.display.set_caption(
            "Ghost Eval | "
            f"{args.method} | ep {episode + 1}/{args.num_eval_episodes} "
            f"frame {frame} | policy {selected} | rho {rho:.2f} | "
            f"speed {speed:.2f} | ped {dist_to_walker:.1f}m"
        )
    except Exception:
        pass


def print_live_status(args, episode, frame, selected, takeover, alpha, risk, info,
                      tcp_action, rl_action, final_action, reward):
    if not args.live_status:
        return
    if args.status_every <= 0 or frame % args.status_every != 0:
        return

    tcp_steer, tcp_throttle, tcp_brake = split_action(tcp_action)
    rl_steer, rl_throttle, rl_brake = split_action(rl_action)
    final_steer, final_throttle, final_brake = split_action(final_action)
    status = (
        f"\r[ep {episode + 1}/{args.num_eval_episodes} f {frame:04d}] "
        f"policy={selected:<6} takeover={int(bool(takeover))} alpha={alpha:.2f} "
        f"rho={risk.get('rho', 0.0):.2f}/{args.delta:.2f} "
        f"ttc={risk.get('ttc', -1.0):5.2f} "
        f"ped={float(info.get('dist_to_walker', -1.0)):5.1f}m "
        f"speed={float(info.get('speed', 0.0)):4.1f} "
        f"reward={float(reward):6.2f} | "
        f"TCP(s={tcp_steer:+.2f},t={tcp_throttle:.2f},b={tcp_brake:.2f}) "
        f"RL(s={rl_steer:+.2f},t={rl_throttle:.2f},b={rl_brake:.2f}) "
        f"OUT(s={final_steer:+.2f},t={final_throttle:.2f},b={final_brake:.2f})"
    )
    print(status[:240].ljust(240), end="", flush=True)


def evaluate(args):
    validate_options(args)  # Fail before creating CARLA if learned weights/group identity are missing.
    reference_config = ReferenceConfig(args.reference_distance, args.reference_lateral,
                                       args.reference_time, args.reference_alpha, args.reference_epsilon)
    needs_learned = args.method in RISK_METHODS and args.risk_estimator == "learned"
    needs_features = needs_learned or args.collect_risk_data
    needs_reference = args.collect_risk_data or (args.method in RISK_METHODS and args.risk_estimator == "reference")
    utils.set_seed_everywhere(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.method in ("ours", "ours_rs"):
        args.tcp_use_ggca = True

    run_name = (
        f"{args.method}_{args.scenarios}_{args.weather}_seed_{args.seed}_"
        f"{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    work_dir = utils.make_dir(os.path.join(args.work_dir, run_name))
    video_dir = utils.make_dir(os.path.join(work_dir, "video"))
    front_frame_root = os.path.join(work_dir, "front_frames")
    if args.save_front_frames:
        os.makedirs(front_frame_root, exist_ok=True)
    log_path = os.path.join(work_dir, "frames.csv")
    summary_path = os.path.join(work_dir, "summary.json")
    write_json(os.path.join(work_dir, "args.json"), vars(args))

    tcp_net, tcp_config = (None, None)
    if should_load_tcp(args.method) or needs_features:
        tcp_net, tcp_config = load_tcp(args.tcp_ckpt, args)
    signature = encoder_signature(tcp_net) if needs_features else None
    risk_head, risk_checkpoint = (None, None)
    if needs_learned:
        risk_head, risk_checkpoint = load_head(args.risk_head_ckpt, signature, device)
    write_json(os.path.join(work_dir, "risk_config.json"), {
        "risk_backend": args.risk_estimator if args.method in RISK_METHODS else "none",
        "uses_privileged_risk": args.method in RISK_METHODS and args.risk_estimator != "learned",
        "encoder_signature": signature,
        "collection_reference": reference_config.__dict__ if args.collect_risk_data else None,
        "checkpoint_reference": risk_checkpoint.get("reference_config") if risk_checkpoint else None,
        "scene_group": args.risk_scene_group,
    })
    env = create_env(args)
    base_env = get_base_env(env)
    video = VideoRecorder(video_dir if args.save_video else None)

    rl_agent = None
    if should_load_rl(args.method):
        rl_agent = create_rl_agent(env, args, device)

    fieldnames = [
        "episode", "frame", "method", "selected_policy", "takeover", "alpha",
        "rho", "delta", "ttc", "risk_longitudinal", "risk_lateral", "risk_distance",
        "speed", "reward", "done", "success", "collision", "crash_intensity",
        "dist_to_walker", "distance", "acceleration", "yaw_diff",
        "tcp_steer", "tcp_throttle", "tcp_brake",
        "rl_steer", "rl_throttle", "rl_brake",
        "final_steer", "final_throttle", "final_brake",
        "front_rgb_path", "front_meta_path", "front_rgb_sha256",
        "weather", "input_world_frame", "next_world_frame",
        "reason_each_episode_ended",
        "input_speed", "next_speed", "input_sim_time", "next_sim_time",
        "risk_backend", "uses_privileged_risk", "risk_reference", "reference_error",
        "executed_steer", "executed_throttle", "executed_brake",
    ]

    episodes = []
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        try:
            for episode in range(args.num_eval_episodes):
                obs = env.reset()
                video.init(enabled=args.save_video)
                done = False
                frame = 0
                episode_reward = 0.0
                episode_distance = 0.0
                episode_takeovers = 0
                episode_collision = False
                episode_success = False
                previous_acc = None
                jerks = []
                min_dist_to_walker = None
                min_ttc = None
                feature_rows, reference_rows, reference_frames = [], [], []
                reference_errors = 0
                first_takeover_frame = None
                first_takeover_time = None
                previous_policy = None
                switch_count = 0
                episode_time_origin = None

                while not done:
                    # Snapshot only observable policy inputs before env.step().
                    tcp_rgb, input_speed, input_target, input_command = base_env.get_tcp_inputs()
                    input_snapshot = base_env.world.get_snapshot()
                    input_sim_time = float(input_snapshot.timestamp.elapsed_seconds)
                    if episode_time_origin is None:
                        episode_time_origin = input_sim_time
                    input_time = input_sim_time - episode_time_origin
                    features = None

                    action_traj = np.zeros(2, dtype=np.float32)
                    action_ctrl = np.zeros(2, dtype=np.float32)
                    tcp_action = np.zeros(2, dtype=np.float32)
                    if tcp_net is not None:
                        if needs_features:
                            action_traj, action_ctrl, features = tcp_actions_and_features(
                                tcp_net, tcp_rgb, input_speed, input_target, input_command)
                        else:
                            action_traj, action_ctrl = get_tcp_actions(tcp_net, env, tcp_config)
                        tcp_action = make_tcp_action(action_traj, action_ctrl, args)
                        tcp_action[0] *= float(args.tcp_steer_sign)
                        steer_clip = abs(float(args.tcp_steer_clip))
                        tcp_action[0] = np.clip(tcp_action[0], -steer_clip, steer_clip)
                        if args.method in ("tcp_rs", "ours_rs"):
                            tcp_action[0] = 0.0

                    reference_value, reference_error = None, ""
                    if needs_reference:
                        try:
                            reference_value = reference_from_env(base_env, reference_config)
                        except (ValueError, RuntimeError, AttributeError) as exc:
                            reference_error = str(exc)
                            reference_errors += 1
                            if args.method in RISK_METHODS and args.risk_estimator == "reference":
                                raise  # Invalid oracle input must not silently become zero risk.
                    if args.collect_risk_data and reference_value is not None:
                        feature_rows.append(features.copy())
                        reference_rows.append(reference_value)
                        reference_frames.append(frame)
                    risk = compute_online_risk(
                        args.risk_estimator if args.method in RISK_METHODS else "none",
                        head=risk_head, features=features, privileged_reference=reference_value,
                        legacy=lambda: compute_risk(env, args))

                    rl_action = np.zeros(2, dtype=np.float32)
                    if rl_agent is not None:
                        with utils.eval_mode(rl_agent):
                            if args.agent == "ddqn":
                                action_idx = rl_agent.select_action(obs)
                                action_space = np.array([-0.50, -0.30, -0.10, 0.0, 0.10, 0.30, 0.50])
                                rl_action = np.array([action_space[action_idx], 0.5], dtype=np.float32)
                            else:
                                rl_action = rl_agent.select_action(obs).astype(np.float32)

                    rule_action = np.zeros(2, dtype=np.float32)
                    if args.method == "rule":
                        rule_action = make_rule_action(env, args)

                    final_action, selected, takeover, alpha = choose_action(
                        args.method, tcp_action, rl_action, rule_action, risk["rho"], frame, args
                    )

                    front_rgb = None
                    if args.save_front_frames and args.front_frame_every > 0 and frame % args.front_frame_every == 0:
                        try:
                            front_rgb = np.asarray(tcp_rgb).copy()
                        except Exception:
                            front_rgb = None

                    obs, reward, done, info = env.step(final_action)
                    info = info or {}
                    next_snapshot = base_env.world.get_snapshot()
                    next_sim_time = float(next_snapshot.timestamp.elapsed_seconds)
                    if previous_policy is not None and previous_policy != selected:
                        switch_count += 1
                    previous_policy = selected
                    if takeover and first_takeover_frame is None:
                        first_takeover_frame, first_takeover_time = frame, input_time
                    video.record(base_env)
                    update_window_caption(
                        args=args,
                        episode=episode,
                        frame=frame,
                        selected=selected,
                        rho=float(risk.get("rho", 0.0)),
                        speed=float(info.get("speed", 0.0)),
                        dist_to_walker=float(info.get("dist_to_walker", -1.0)),
                    )

                    episode_reward += float(reward)
                    episode_distance += float(info.get("distance", 0.0))
                    episode_takeovers += int(bool(takeover))
                    episode_collision = episode_collision or bool(info.get("collision", False)) or float(info.get("crash_intensity", 0.0)) > 0.0
                    episode_success = episode_success or bool(info.get("success", False))

                    dist_to_walker = float(info.get("dist_to_walker", -1.0))
                    if dist_to_walker >= 0.0:
                        min_dist_to_walker = dist_to_walker if min_dist_to_walker is None else min(min_dist_to_walker, dist_to_walker)
                    ttc = float(risk.get("ttc", -1.0))
                    if ttc >= 0.0:
                        min_ttc = ttc if min_ttc is None else min(min_ttc, ttc)

                    acc = float(info.get("acceleration", 0.0))
                    if previous_acc is not None:
                        dt = 0.05 * args.action_repeat
                        jerks.append(abs((acc - previous_acc) / max(dt, 1e-6)))
                    previous_acc = acc

                    tcp_steer, tcp_throttle, tcp_brake = split_action(tcp_action)
                    rl_steer, rl_throttle, rl_brake = split_action(rl_action)
                    final_steer, final_throttle, final_brake = split_action(final_action)
                    frame_row = {
                        "episode": episode,
                        "frame": frame,
                        "method": args.method,
                        "selected_policy": selected,
                        "takeover": int(bool(takeover)),
                        "alpha": alpha,
                        "rho": risk.get("rho", 0.0),
                        "delta": args.delta,
                        "ttc": risk.get("ttc", -1.0),
                        "risk_longitudinal": risk.get("risk_longitudinal", 0.0),
                        "risk_lateral": risk.get("risk_lateral", 0.0),
                        "risk_distance": risk.get("risk_distance", -1.0),
                        "speed": info.get("speed", 0.0),
                        "reward": reward,
                        "done": int(bool(done)),
                        "success": int(bool(info.get("success", False))),
                        "collision": int(bool(info.get("collision", False))),
                        "crash_intensity": info.get("crash_intensity", 0.0),
                        "dist_to_walker": dist_to_walker,
                        "distance": info.get("distance", 0.0),
                        "acceleration": acc,
                        "yaw_diff": info.get("yaw_diff", 0.0),
                        "tcp_steer": tcp_steer,
                        "tcp_throttle": tcp_throttle,
                        "tcp_brake": tcp_brake,
                        "rl_steer": rl_steer,
                        "rl_throttle": rl_throttle,
                        "rl_brake": rl_brake,
                        "final_steer": final_steer,
                        "final_throttle": final_throttle,
                        "final_brake": final_brake,
                        "front_rgb_path": "",
                        "front_meta_path": "",
                        "front_rgb_sha256": "",
                        "weather": args.weather,
                        "input_world_frame": int(input_snapshot.frame),
                        "next_world_frame": int(next_snapshot.frame),
                        "reason_each_episode_ended": info.get("reason_each_episode_ended", ""),
                        "input_speed": input_speed,
                        "next_speed": info.get("speed", 0.0),
                        "input_sim_time": input_time,
                        "next_sim_time": next_sim_time - episode_time_origin,
                        "risk_backend": risk["risk_backend"],
                        "uses_privileged_risk": int(risk["uses_privileged_risk"]),
                        "risk_reference": reference_value if args.collect_risk_data and reference_value is not None else "",
                        "reference_error": reference_error,
                        "executed_steer": info.get("steer", ""),
                        "executed_throttle": info.get("throttle", ""),
                        "executed_brake": info.get("brake", ""),
                    }
                    if front_rgb is not None:
                        front_rgb_path, front_meta_path = save_front_frame(
                            front_frame_root,
                            episode,
                            frame,
                            front_rgb,
                            frame_row,
                        )
                        frame_row["front_rgb_path"] = front_rgb_path
                        frame_row["front_meta_path"] = front_meta_path
                    print_live_status(
                        args=args,
                        episode=episode,
                        frame=frame,
                        selected=selected,
                        takeover=takeover,
                        alpha=alpha,
                        risk=risk,
                        info=info,
                        tcp_action=tcp_action,
                        rl_action=rl_action,
                        final_action=final_action,
                        reward=reward,
                    )

                    writer.writerow(frame_row)
                    frame += 1

                if args.collect_risk_data and feature_rows:
                    save_episode(os.path.join(work_dir, "risk_data", "episode_{:03d}.npz".format(episode)),
                                 feature_rows, reference_rows, reference_frames, {
                        "scene_group": args.risk_scene_group.strip(), "episode": episode,
                        "encoder_signature": signature, "reference_config": reference_config.__dict__,
                        "weather": args.weather, "seed": args.seed, "method": args.method,
                        "tcp_use_ggca": args.tcp_use_ggca, "tcp_ggca_groups": args.tcp_ggca_groups,
                        "valid_samples": len(reference_rows), "invalid_reference_samples": reference_errors,
                    })
                video.save(f"{episode}.mp4")
                if args.live_status:
                    print()
                episodes.append({
                    "episode": episode,
                    "reward": episode_reward,
                    "distance": episode_distance,
                    "frames": frame,
                    "success": episode_success,
                    "collision": episode_collision,
                    "takeover_ratio": episode_takeovers / max(frame, 1),
                    "avg_jerk": float(np.mean(jerks)) if jerks else 0.0,
                    "min_dist_to_walker": -1.0 if min_dist_to_walker is None else min_dist_to_walker,
                    "min_ttc": -1.0 if min_ttc is None else min_ttc,
                    "takeover_occurred": first_takeover_frame is not None,
                    "first_takeover_frame": first_takeover_frame,
                    "first_takeover_time": first_takeover_time,
                    "policy_switch_count": switch_count,
                    "risk_samples_saved": len(reference_rows),
                    "invalid_reference_samples": reference_errors,
                })
                print(
                    f"Episode {episode + 1}/{args.num_eval_episodes}: "
                    f"success={episode_success}, collision={episode_collision}, "
                    f"reward={episode_reward:.2f}, takeover={episodes[-1]['takeover_ratio']:.2f}"
                )
        finally:
            get_base_env(env).finish()

    summary = {
        "method": args.method,
        "risk_backend": args.risk_estimator if args.method in RISK_METHODS else "none",
        "uses_privileged_risk": args.method in RISK_METHODS and args.risk_estimator != "learned",
        "episode_takeover_incidence": float(np.mean([e["takeover_occurred"] for e in episodes])) if episodes else 0.0,
        "avg_policy_switch_count": float(np.mean([e["policy_switch_count"] for e in episodes])) if episodes else 0.0,
        "scenario": args.scenarios,
        "weather": args.weather,
        "seed": args.seed,
        "episodes": episodes,
        "num_episodes": len(episodes),
        "success_rate": float(np.mean([e["success"] for e in episodes])) if episodes else 0.0,
        "collision_rate": float(np.mean([e["collision"] for e in episodes])) if episodes else 0.0,
        "avg_reward": float(np.mean([e["reward"] for e in episodes])) if episodes else 0.0,
        "avg_distance": float(np.mean([e["distance"] for e in episodes])) if episodes else 0.0,
        "avg_takeover_ratio": float(np.mean([e["takeover_ratio"] for e in episodes])) if episodes else 0.0,
        "avg_jerk": float(np.mean([e["avg_jerk"] for e in episodes])) if episodes else 0.0,
        "avg_min_dist_to_walker": float(np.mean([e["min_dist_to_walker"] for e in episodes if e["min_dist_to_walker"] >= 0.0]))
        if any(e["min_dist_to_walker"] >= 0.0 for e in episodes) else -1.0,
        "avg_min_ttc": float(np.mean([e["min_ttc"] for e in episodes if e["min_ttc"] >= 0.0]))
        if any(e["min_ttc"] >= 0.0 for e in episodes) else -1.0,
        "frame_log": log_path,
    }
    write_json(summary_path, summary)

    print("\nFINAL SUMMARY")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))
    print(f"\nSaved frame log: {log_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    evaluate(parse_args())
