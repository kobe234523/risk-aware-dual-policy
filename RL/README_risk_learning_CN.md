# 连续风险监督与轻量风险网络

本次增加连续风险标签、冻结视觉特征采集、MLP训练与测试，以及在线学习型风险接管。代码兼容项目现有的Python 3.7和PyTorch 1.7环境。

## 模块和数据流

- `risk_learning/reference.py`：只在显式采集或`reference`对照模式下读取仿真行人几何信息。多行人取最大参考值，车辆后方目标贡献为0，异常状态不作为安全标签。
- `risk_learning/model.py`：风险头为521→256→128→1，ReLU、ReLU、Sigmoid，共166657个参数。512维来自TCP视觉编码器返回的GGCA后空间特征的平均池化，另有车速1维、目标点2维和导航one-hot 6维。
- `risk_learning/tcp_adapter.py`：只执行一次TCP前向，同时取候选动作和池化特征。保持TCP本身的速度/12预处理，风险分支单独采用速度/10和目标点/30。相机输入为256×900 RGB。
- `risk_learning/data.py`：每回合保存NPZ特征分片，按完整场景组划分数据，不随机拆分帧。元数据记录参考参数、编码器指纹、天气和种子。
- `train_risk_head.py`：不依赖CARLA，只训练缓存特征上的风险头；训练阶段不评估测试集。独立test命令使用原始划分清单和校验和。
- `eval_ghost_dual_policy.py`：新风险推理、特征采集以及对齐的控制日志入口。

## 默认行为变更

`ours`、`ours_rs`、`risk_switch`和`risk_fusion`默认使用`--risk_estimator learned`，必须提供真正训练好的`--risk_head_ckpt`。缺少权重、编码器不匹配、无效特征或无效预测会报错，不会退回规则风险或随机初始化风险头。

旧几何计算仍可通过`--risk_estimator legacy`显式调用。新连续参考值也可通过`--risk_estimator reference`作为特权信息对照。两种对照均在日志中标记`uses_privileged_risk=true`，不能作为纯可观测输入方法。

不需要风险门控的TCP、SAC、固定切换等模式不再默认读取行人真值来计算无用风险。环境内部仍会读取仿真状态生成训练奖励和离线评价指标；这一点与在线风险头输入不同。

## 连续参考标签

对每个行人计算纵向距离d、横向距离l，以及T=d/max(v,epsilon)（d>0）；T是到达时间近似量，不是严格TTC。

`q = I(d>0) * exp(-l²/(2L²)) * [alpha*clip(1-d/D,0,1)+(1-alpha)*clip(1-T/T0,0,1)]`

`y = max(q)`；没有行人时y=0。默认D=15m、L=3m、T0=3s、alpha=0.5、epsilon=0.1m/s。它是连续取值的风险参考，不是校准碰撞概率。L是标签的软尺度，区别于SAC奖励中的1.5m硬边界。

## 使用顺序

以下命令在RL目录运行，路径必须替换成自己的真实文件。这里仅给出使用方式，没有执行CARLA采集或正式训练。

### 1 采集冻结特征与参考标签

```powershell
python eval_ghost_dual_policy.py --method tcp --tcp_use_ggca --tcp_ckpt "实际TCP-GGCA权重.ckpt" --collect_risk_data --risk_scene_group "routeA_occluderLayout01" --work_dir "./log/risk_collection" --num_eval_episodes 20 --weather ClearNoon --seed 1
```

可以用既有SAC采集更丰富的状态：改为`--method sac`并提供`--rl_model_dir`；仍需加载同一TCP编码器以提取特征。采集模式无需风险头权重。

`risk_scene_group`必须反映真实路线和遮挡/行人布局；同一场景的不同天气、种子和回合使用同一个组名，不能为了满足划分要求随意更改组名。至少需要3个真实独立场景组，建议准备足够组数再按70/15/15划分。代码只保证相同组名不跨集合，场景分组的物理含义须由采集者保证。编码器权重或参考公式不同的数据会被拒绝混用。

每个正常结束的回合生成一个`risk_data/episode_XXX.npz`。无效标签样本被剔除，并记录剔除数量；无效在线特征不会继续控制。可加`--save_front_frames`保留原图供审查，但风险头训练只使用冻结后的特征。

### 2 单独训练风险头

```powershell
python train_risk_head.py train --data_dir "./log/risk_collection" --output_dir "./log/risk_head_run01" --device cuda
```

默认Adam、MSE、学习率1e-4、weight_decay=1e-5、batch=256、最多30轮、验证MSE连续5轮无改善则早停。只训练风险头，完全不更新TCP/SAC。输出`risk_head.pt`、`split_manifest.json`、训练历史和摘要。训练记录目录不可覆盖，以防不慎改变数据划分与检查点。

### 3 独立测试

```powershell
python train_risk_head.py test --checkpoint "./log/risk_head_run01/risk_head.pt" --manifest "./log/risk_head_run01/split_manifest.json" --output "./log/risk_head_run01/test_metrics.json" --device cuda
```

仅计算连续参考的MSE/MAE，不宣称碰撞概率校准。不得利用测试结果挑选模型和阈值。

### 4 接入双策略

```powershell
python eval_ghost_dual_policy.py --method ours --risk_estimator learned --risk_head_ckpt "./log/risk_head_run01/risk_head.pt" --tcp_ckpt "实际TCP-GGCA权重.ckpt" --rl_model_dir "实际SAC模型目录" --save_front_frames --work_dir "./log/learned_risk_eval"
```

风险头保存实际编码器状态的SHA256指纹。即使文件名相同，编码器权重不同也会拒绝加载，避免训练与推理特征分布不一致。如果旧TCP加载器没有真正加载GGCA权重，随机初始化会造成指纹不一致，不能绕过该校验；应先核对原TCP检查点。

## 日志

保留原字段`speed`的执行后含义，并增加：

- `input_speed`、`input_sim_time`：作出当前风险判断/动作决策时的速度与相对仿真时间。
- `next_speed`、`next_sim_time`：执行当前动作后的速度和时间。
- `risk_backend`、`uses_privileged_risk`：明确风险来源。
- `risk_reference`：仅显式采集时写入，不作为learned模式的预测输入。
- `final_*`：选择器输出命令；`executed_*`：环境实际采用的控制值（若环境返回）。
- 每回合记录首次接管时间、是否发生过接管、策略切换次数。原`takeover_ratio`仍指控制步占比，新增`episode_takeover_incidence`才指发生接管的回合占比。

Figure 10应使用同一回合的原图、输入时间、风险预测与控制值对齐，避免把动作后的速度当成决策前速度。

## 已完成与未完成

已实现并用离线单元/模拟集成测试检查标签边界、多行人、归一化、编码器指纹、场景组隔离、训练检查点往返，以及learned控制路径不调用特权风险计算。

本次没有运行真实CARLA、没有正式收集/训练风险模型，也没有产生论文实验结果或新的已训练权重。实际评测仍需要项目原有CARLA和TCP/SAC依赖；原TCP加载器仍按既有实现使用CUDA。
