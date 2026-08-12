# Piper Puzzle UVA-DiT / WAM 真机部署

这个目录提供 Puzzle 0809-B、0811-C 与当前 0812-E checkpoint 在松灵 Piper 真机上的
推理 server、机器人 client、轨迹录制和 replay 工具。它从
`UVA_dit/realmachine_deploy_v2/piper` 提取，作为 WAM 的 Piper 部署适配层。

## 仓库边界与目录

完整模型实现仍由外部 `UVA_dit` checkout 提供，尤其是
`realmachine_deploy_v2/tianyi/infer_server_pi07_ar.py` 及其模型模块。真实推理前
必须设置：

```bash
export UVA_DIT_ROOT=/path/to/UVA_dit
```

权重、norm stats、UniDiT base 和 Qwen3.5 也保持在仓库外。仅运行 WebSocket mock
协议测试时不需要 `UVA_DIT_ROOT`。

本目录结构为：

```text
server/   推理 server、启动器和测试
client/   机器人 client、WebSocket 协议、状态历史、位姿和视频录制依赖
```

server 与 client 共同保持 Piper 的 ROS topic、相机传输和动作 response 格式兼容。

## 部署目录

开发服务器需要同时存在本仓库和完整 `UVA_dit` checkout，例如：

```text
/bh/zbh_self/projects/piper-openpi-real-robot
/bh/zbh_self/projects/UVA_dit
```

仓库源码按模型保存在 `models/uva_dit/client/`。当前机器人运行环境沿用兼容部署
路径：

```text
/home/agilex/piper-openpi-real-robot/scripts
/home/agilex/piper-openpi-real-robot/configs
```

从仓库根目录更新机器人 client：

```bash
ssh agilex@10.13.11.215 \
  'mkdir -p /home/agilex/piper-openpi-real-robot/scripts /home/agilex/piper-openpi-real-robot/configs /home/agilex/piper-openpi-real-robot/recordings'

scp models/uva_dit/client/*.py \
  agilex@10.13.11.215:/home/agilex/piper-openpi-real-robot/scripts/
```

这里复制整个 client 目录中的 Python 文件，确保主 client 的相机、state history、
start pose、轨迹记录和 WebSocket 协议依赖同时更新。启动真机前还必须单独确认
机器人已有 `configs/piper_uva_dit_puzzle_b_left_observation_pose.json`；该现场配置
当前不在本仓库中。

## 训练/推理契约

已支持的训练契约为：

```text
scripts/train/acclerate/refine_debug/0809/
train_puzzle_0809_B_from_A4k_new0805_train406_eval20_p2p_ar2_rightactionloss_rightgripsampler_lr1e6_12k_wandb.sh

scripts/train/acclerate/refine_debug/0811/
train_puzzle_0811_C_from_Abest_new0805_train406_eval20_p2p_noar_rightactionloss_rightgripsampler_lr1e6_12k_wandb.sh
```

server 会在启动时从 `config.json` 强制校验：

- `joint_pos`，`puppet → puppet`；
- 模型头 32D，物理动作 14D，尾部 18D 必须是归一化零；
- 物理布局为 `[left_arm6, right_arm6, left_gripper, right_gripper]`；
- action condition 为 causal history16；
- 图像 history 为 checkpoint 的 `pi07_history_frames`：B 为 2（image AR2），C 为 0（image no-AR）；三路输入始终为头部、左腕、右腕；
- B/C 的 loss mask 都必须是右臂+右夹爪 `6,7,8,9,10,11,13`；
- 数据集时序为 30 FPS。

Piper client 的格式不同：

```text
client left_arm  = [left_arm6, left_gripper]
client right_arm = [right_arm6, right_gripper]

model physical14 = [left_arm6, right_arm6, left_gripper, right_gripper]
```

该映射在 server 边界完成；不能直接把模型前 7D 和后 7D 当作左右臂。

## 调度方式

默认每次模型生成 32 步，但 server 只给旧 Piper client 返回前 4 步：

```text
动作间隔       1 / 30 s
每个请求返回   4 步
完整模型 chunk 32 步 / 8 个请求消费
重新去噪       chunk 用完后
```

部署时可设 `EXECUTE_STEPS=24`：每次模型预测 32 步，仅下发前 24 步；下一次请求会使用当前图像和 state 重新去噪，余下 8 步不会执行。B 的 image AR2 与 C 的 image no-AR 均由 checkpoint 配置自动选择；二者都保留 causal action history16。

Piper client 现在在每个 waypoint 发布后读取 `/puppet/joint_left` 和
`/puppet/joint_right`，把真实状态转换为 `[left_arm6, right_arm6,
left_gripper, right_gripper]` 的 14D 行，并在下一次请求中发送最近 16 行
`state_history`（顺序为 oldest → newest）。server 在推理前调用 Pi07 的
`update_state_history`，真实 measured history 优先于任何 fallback；C 实时
推理缺少或发送错误 history 时会拒绝请求，不会把预测 waypoint 静默当作历史。

若需要在机器人本地保存可 replay 的实际执行轨迹，在 client 启动命令中加：

```bash
--record-trajectory \
--record-dir /home/agilex/piper-openpi-real-robot/recordings
```

结果保存在 `recordings/<session>/trajectory.hdf5`。HDF5 的 `/action` 只包含
实际发布的绝对 joint waypoint，使用 Piper wire order
`[L1..L6,Lg,R1..R6,Rg]`；未执行的模型预测尾部不会写入。`/state` 是每个
waypoint 发布后的实测 state，`/initial_state` 是启动位姿准备完成后、第一条
policy action 执行前的 state。

## 安全默认值

Puzzle B/C 都只监督右侧输出。为避免未监督左侧动作造成风险，server 默认：

- 下发的左臂六轴与左夹爪保持当前实测 Piper 状态；
- 右臂六轴与右夹爪采用模型输出；
- `INACTIVE_ACTION_MODE=model` 才允许原始模型控制左侧；
- `MAX_JOINT_STEP` 可限制每个右臂 waypoint 相对前一 waypoint 的最大弧度变化；默认 `0.0` 表示不额外改变模型轨迹。

可选 `RECORD_DIR` 会记录每个 request 的原始模型 14D 输出、最终下发的 14D 输出和 timing；不会把大 JPEG payload 重复写进 JSONL。

若要保存可用于后续 replay 的“模型原始推理轨迹”，在 server 启动时设置：

```bash
UVA_DIT_ROOT=/path/to/UVA_dit \
RAW_ACTION_HDF5=/bh/zbh_ckp/datasets/piper_uva_dit_raw_actions/raw_model_actions_<run>.hdf5 \
RAW_ACTION_STEPS=8 \
bash models/uva_dit/server/run_piper_puzzle_policy_server.sh
```

`run_piper_puzzle_c_noar_policy_server.sh` 默认已经打开这项记录，并把文件放在
`/bh/zbh_ckp/datasets/piper_uva_dit_raw_actions/`。HDF5 的 `/action` 是模型
denormalize 后、server 的左臂 hold 和右臂限幅之前的原始 physical14 输出，布局为
`[left_arm6, right_arm6, left_gripper, right_gripper]`；每次成功响应只保存前 8
行。`/action_piper_wire` 是同一批原始 action 的 Piper 7+7 映射，供后续 replay
使用；它不是 client 实际执行后的轨迹。`/current_state`、`/initial_state`、
`/dt`、`/request_index` 和 `/waypoint_index` 用于对齐推理请求与回放时间。

## 原始模型 action 的右臂 replay

server 的 raw-action HDF5 不能直接把 14D 整行发送给 Piper。模型布局中的右臂
是 `action[6:12] + [action[13]]`，对应 Piper 右侧 7D
`[R1..R6,Rg]`。机器人端使用独立 replay 脚本，只创建右臂 publisher，不会发布
左臂 topic：

```bash
cd /home/agilex/piper-openpi-real-robot
python3 models/uva_dit/client/replay_raw_action_hdf5.py \
  --hdf5 /path/to/raw_model_actions_xxx.hdf5 \
  --control-hz 20
```

上面命令只校验并预览。确认文件和首个 waypoint 正确后，才加 `--execute`：

```bash
python3 models/uva_dit/client/replay_raw_action_hdf5.py \
  --hdf5 /path/to/raw_model_actions_xxx.hdf5 \
  --control-hz 20 \
  --execute
```

`--control-hz` 是实际发布频率；它不使用 HDF5 中 server 记录的 wall-clock 时间，
因此可以按需要改成 `10`、`20` 或 `30`。脚本默认不 enable、不 home、不 hold，
可显式追加 `--enable-on-start`、`--hold-on-exit` 或 `--disable-on-exit`。

## 实际执行轨迹的双臂 replay

client 通过 `--record-trajectory` 保存的 HDF5 与上面的 server raw-action HDF5
不是同一种格式。它记录真正发布过的双臂绝对 joint waypoint、实测 state 和
每步时间。先使用默认 dry-run 检查文件：

```bash
python3 models/uva_dit/client/replay_trajectory_hdf5.py \
  --hdf5 /path/to/recordings/<session>/trajectory.hdf5
```

确认 `initial_state`、首尾 action、行数与时长后，才允许真机发布：

```bash
python3 models/uva_dit/client/replay_trajectory_hdf5.py \
  --hdf5 /path/to/recordings/<session>/trajectory.hdf5 \
  --execute
```

`--execute` 会先读取双臂实测 state，平滑对齐到文件的 `initial_state`，再按记录的
`dt` 发布双臂 waypoint。默认结束后不 hold、不 disable；需要时显式加
`--hold-on-exit` 或 `--disable-on-exit`。首次回放可用 `--max-steps` 限制范围，
但仍必须完成现场净空、急停和起始位姿检查。

## 环境

推理环境位于：

```text
/bh/zbh_self/envs/uva-dit-piper
```

它从本机 CUDA 12.1 PyTorch 环境克隆后，再安装
`models/uva_dit/requirements-piper.txt`。这里将 NumPy 固定在
`>=1.26,<2.0`，并固定 `transformers==5.9.0`；当前模型的
`Qwen3_5ForConditionalGeneration` 不兼容旧版 Transformers。

若本机默认 PyPI 镜像未提供 NumPy 1.x，使用下面这个一次性的官方源安装命令（代理只对该命令生效）：

```bash
cd /path/to/piper-openpi-real-robot
PIP_CACHE_DIR=/bh/zbh_self/cache/pip \
http_proxy=http://192.168.32.28:18000 https_proxy=http://192.168.32.28:18000 \
/bh/zbh_self/envs/uva-dit-piper/bin/python -m pip install \
  --index-url https://pypi.org/simple \
  -r models/uva_dit/requirements-piper.txt
```

checkpoint、UniDiT base、Qwen3.5 权重、Hugging Face cache 和记录输出均应放在 `/bh/zbh_ckp`，不要放进 Git 仓库或 `/root`。

环境完成后先确认：

```bash
/bh/zbh_self/envs/uva-dit-piper/bin/python - <<'PY'
import torch, cv2, websockets, safetensors, transformers
print(torch.__version__, torch.cuda.is_available())
PY
```

不依赖 checkpoint 的测试：

```bash
/bh/zbh_self/envs/uva-dit-piper/bin/python \
  models/uva_dit/server/test_piper_puzzle_policy_server.py
/bh/zbh_self/envs/uva-dit-piper/bin/python \
  models/uva_dit/server/test_piper_puzzle_model_adapter.py
```

## 启动前检查

checkpoint 到位后，至少需要：

```text
RUN_DIR/config.json
RUN_DIR/checkpoint_*/transformer/
RUN_DIR/checkpoint_*/qwen3vl_proj.pt
配套的 Puzzle train406 norm stats .pt
UniDiT base: vae/ 和 transformer/
Qwen3.5-9B 目录
```

只读契约检查：

```bash
cd /path/to/piper-openpi-real-robot

CONFIG_ONLY=1 \
UVA_DIT_ROOT=/path/to/UVA_dit \
RUN_DIR=/bh/zbh_ckp/checkpoints/<puzzle-run> \
CKPT_NAME=checkpoint_step_<step> \
NORM_STATS=/bh/zbh_ckp/checkpoints/<puzzle-stats>.pt \
PRETRAINED=/bh/zbh_ckp/models/unidit-base \
QWEN35=/bh/zbh_ckp/models/Qwen3.5-9B \
bash models/uva_dit/server/run_piper_puzzle_policy_server.sh
```

正式启动：

```bash
GPU=0 \
UVA_DIT_ROOT=/path/to/UVA_dit \
RUN_DIR=/bh/zbh_ckp/checkpoints/<puzzle-run> \
CKPT_NAME=checkpoint_step_<step> \
NORM_STATS=/bh/zbh_ckp/checkpoints/<puzzle-stats>.pt \
PRETRAINED=/bh/zbh_ckp/models/unidit-base \
QWEN35=/bh/zbh_ckp/models/Qwen3.5-9B \
MAX_JOINT_STEP=0.10 \
bash models/uva_dit/server/run_piper_puzzle_policy_server.sh
```

以下 Puzzle 0811-C 启动器作为历史兼容入口保留；当前 0812-E 真机运行参数以
下一节“三终端真机链路”为准。0811-C 启动器固定 C 的 checkpoint、配套 norm stats 和
`EXECUTE_STEPS=24`。它每次预测 32 个绝对关节位置 waypoint，只下发前 24
个；随后以新观测和当前 state 重新预测，且 server 会保留最后 16 个已下发
的真实机器人 state 作为 causal action-condition history。Piper client 需要
使用已部署的 state-history 版本：

```bash
cd /path/to/piper-openpi-real-robot
UVA_DIT_ROOT=/path/to/UVA_dit \
bash models/uva_dit/server/run_piper_puzzle_c_noar_policy_server.sh
```

可先加 `CONFIG_ONLY=1` 做只读 artifact/contract 检查。该 C 专用启动器所有
环境变量仍可覆盖，例如 `GPU=1` 或 `PORT=7082`。

server 默认监听 `ws://0.0.0.0:7081`。checkpoint 不在此机器时，先把大文件放到 `/bh/zbh_ckp/checkpoints` 或 `/bh/zbh_ckp/models` 后再启动。

## 三终端真机链路

下面是 Puzzle 0812-E checkpoint 的当前真机执行流程。终端 3 使用真实 ROS
observation、三路相机并发布机械臂 action，不是 dry-run。执行前必须完成现场
净空、急停、起始位姿、checkpoint 和 action 检查。

机器人目录中需要已经部署 `scripts/websocket_policy_client.py` 及其同目录依赖，
并存在 `configs/piper_uva_dit_puzzle_b_left_observation_pose.json`。

### 终端 1：启动 WAM Policy Server

在开发服务器执行：

```bash
cd /bh/zbh_self/projects/UVA_dit

GPU=0 \
HOST=127.0.0.1 \
PORT=7081 \
RUN_DIR=/bh/media/unify/joyzhang/checkpoints_unidit_ur_usb/puzzle_0812_E_from_0811Cbest_circle0805_0810_new70_old30_p2p_noar_grasp25_place25_righthead_lr5e7_1k_port23456 \
CKPT_NAME=checkpoint_final \
NORM_STATS=/bh/media/unify/joyzhang/checkpoints_unidit_ur_usb/pertask_norm_stats_puzzle_circle_0805_0810_train1650_mask14_p2p_armsgrip_gripfull.pt \
EXECUTE_STEPS=8 \
REPLAN_EVERY_CALL=0 \
ACTION_DT=0.067 \
RAW_ACTION_STEPS=8 \
INACTIVE_ACTION_MODE=hold \
STATE_OOB_MODE=clip \
ACTION_OUTPUT_CLIP=0 \
MAX_JOINT_STEP=0.10 \
ACTION_LATENT_FP32=1 \
ACTION_SMOOTH_WINDOW=11 \
ACTION_SMOOTH_POLY=2 \
ACTION_SMOOTH_GRIPPERS=0 \
bash realmachine_deploy_v2/piper/server/run_piper_puzzle_c_noar_policy_server.sh
```

### 终端 2：建立 SSH Tunnel

在本地电脑执行并保持运行：

```bash
ssh -o ExitOnForwardFailure=yes \
  -p 7156 \
  -N \
  -L 10.13.0.133:17081:127.0.0.1:7081 \
  root@10.40.1.215
```

### 终端 3：机器人真机 Client

在机器人端执行：

```bash
cd /home/agilex/piper-openpi-real-robot

source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

python3 scripts/websocket_policy_client.py \
  --uri ws://10.13.0.133:17081 \
  --instruction '拼图（圆形）' \
  --source ros \
  --executor ros \
  --execute-actions \
  --no-home-on-start \
  --left-observation-pose configs/piper_uva_dit_puzzle_b_left_observation_pose.json \
  --home-right-on-start \
  --transport-image-profile fastwam \
  --compressed-images \
  --require-images \
  --camera-topic head=/camera_f/color/image_raw/compressed \
  --camera-topic left_wrist=/camera_l/color/image_raw/compressed \
  --camera-topic right_wrist=/camera_r/color/image_raw/compressed \
  --publish-hz 100 \
  --record-videos \
  --record-trajectory \
  --record-dir /home/agilex/piper-openpi-real-robot/recordings \
  --record-session-name piper_c_$(date +%Y%m%d_%H%M%S)
```

## 不加载模型的协议测试

先在 server 机器启动 mock server：

```bash
MOCK_POLICY=1 \
bash models/uva_dit/server/run_piper_puzzle_policy_server.sh
```

再在 Piper client 所在机器测试旧协议：

```bash
python models/uva_dit/client/websocket_policy_client.py \
  --uri ws://<server-ip>:7081 \
  --instruction '拼图（圆形）' \
  --source mock --executor mock --max-steps 10
```

正常输出应为每次四个 waypoint，`dt=0.0333`，且每条 `left_arm`、`right_arm` 都是 7D。

## 真机验证顺序

1. 先运行旧 Piper client 的 `--source mock --executor mock`，只验证 WebSocket 协议。
2. 接入真实 ROS observation、真实三路相机，但不加 `--execute-actions`；查看 server JSONL 中 raw/sent 动作和 normalization 警告。
3. 确认左侧 sent action 始终等于当前状态，右侧动作范围合理后，再启用 `--execute-actions`。
4. 首次真实执行保持 `INACTIVE_ACTION_MODE=hold`，并设置保守的 `MAX_JOINT_STEP`；确认初始位姿与 Puzzle train406 分布一致。

不要在 `--allow-instruction-override` 下做首次真机执行。默认只接受训练 prompt：`拼图（圆形）`。
