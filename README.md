# Risk-Aware Dual-Policy Decision Framework

Research code for “Risk-Aware Dual-Policy Decision Framework for Autonomous Driving in Occluded Pedestrian Scenarios”. The framework combines TCP-GGCA for routine driving, SAC for emergency avoidance, and a learned risk estimate to switch between them.

| Path | Contents |
| --- | --- |
| `RL/eval_ghost_dual_policy.py` | Closed-loop evaluation, baseline modes, feature collection and switching |
| `RL/risk_learning/` | Continuous risk reference, frozen TCP features, risk head and online inference |
| `RL/train_risk_head.py` | Offline training and independent testing of the risk head |
| `RL/train_carla_off_policy.py` and `RL/agent/` | SAC training and policy components |
| `RL/carla_env/` | CARLA environment, occluded pedestrian scenarios, and dense reward |
| `TCP-main-attention/TCP/` | TCP model, GGCA visual encoder and supervised training code |

## Environment

The offline tests were checked with Python 3.7, PyTorch 1.7.0, and torchvision 0.8.1. Install matching PyTorch and torchvision builds for your platform and CUDA version, then run `python -m pip install -r requirements.txt`. Closed-loop evaluation also requires a CARLA server, its matching Python API and CUDA for the existing TCP loader. The local Python environment contained the CARLA 0.9.13 API. The required server and maps should be installed separately.

## Training and evaluation

Run the following from `RL/`. Provide your own checkpoints and repeat risk data collection across at least three distinct scene groups before training.

```sh
python eval_ghost_dual_policy.py --method tcp --tcp_use_ggca --tcp_ckpt ../checkpoints/tcp_ggca.ckpt --collect_risk_data --risk_scene_group routeA_layout01 --work_dir ./log/risk_collection
python train_risk_head.py train --data_dir ./log/risk_collection --output_dir ./log/risk_head --device cuda
python train_risk_head.py test --checkpoint ./log/risk_head/risk_head.pt --manifest ./log/risk_head/split_manifest.json --output ./log/risk_head/test_metrics.json --device cuda
python eval_ghost_dual_policy.py --method ours --risk_estimator learned --tcp_ckpt ../checkpoints/tcp_ggca.ckpt --risk_head_ckpt ./log/risk_head/risk_head.pt --rl_model_dir ../checkpoints/sac --num_eval_episodes 50 --seed 1
```

Use the same TCP-GGCA encoder for feature collection and evaluation. To train the SAC policy, use `python train_carla_off_policy.py --agent sac --scenarios ghost_static` and consult its `--help` for training options. The evaluation directory contains `frames.csv` and `summary.json`. Use `--weather`, `--seed`, and `--delta` to configure evaluation runs. See [the Chinese risk-learning guide](RL/README_risk_learning_CN.md) for the label, split, and logging details.

TCP supervised training uses packed CARLA data. From `TCP-main-attention/`, set `TCP_DATA_ROOT` to the dataset path and run `python -m TCP.train --use_ggca`. This entry also needs imgaug 0.4.0 and the PyTorch Lightning 1.2.5 API used by the original training code. Set `TCP_ROACH_CHECKPOINT` when using `--load_rl_weight`.

Model weights, datasets and run outputs are supplied separately. The CARLA experiments and exact reproduction of the paper's reported numbers have not been verified in this release.

## Offline tests

From the repository root, run `python -m unittest discover -s RL/tests -v`. These tests run without a CARLA server.

## Upstream code

- [DriveOcclusionSim (DOS)](https://github.com/opendilab/DOS) and [TCP](https://github.com/OpenPerceptionX/TCP) provide the upstream benchmark and TCP model. Their Apache 2.0 license files are retained.
- Several files under `RL/` retain Facebook copyright notices from [Deep Bisimulation for Control](https://github.com/facebookresearch/deep_bisim4control). Its upstream [license](https://github.com/facebookresearch/deep_bisim4control/blob/main/LICENSE) is CC BY-NC 4.0.
- The retained CARLA-derived files carry their original Computer Vision Center notices. See the source headers and [CARLA's license](https://github.com/carla-simulator/carla/blob/0.9.13/LICENSE).
