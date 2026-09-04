# Piper OpenPI Real Robot Workflow

这个仓库保存松灵 Piper 双臂机器人运行 OpenPI checkpoint 的完整真机流程。当前链路是：

- 开发服务器使用一张 A800 加载 OpenPI checkpoint，运行 JSON WebSocket policy server。
- 本地电脑通过 SSH tunnel，把机器人可访问的本地端口转发到服务器。
- 机器人端 WebSocket client 读取左右臂 joint state 和三路相机，接收 action chunk。
- 通信测试默认不执行动作；完整实验才向左右臂发布 joint action。
- client 全程录制三路相机，结束时封装 MP4，并上传到开发服务器。

当前 action space 是 joint，不是 EEF。机器人端通信协议不强绑定 OpenPI，但
`piper_openpi_policy_server.py` 是 OpenPI checkpoint 专用适配器。
FastWAM 使用独立的 `piper_fastwam_policy_server.py`，不替换或修改 OpenPI 链路。

## 目录

```text
.
├── README.md
├── docs
│   ├── agilex_robot_ros_startup.md
│   ├── architecture.md
│   └── setup_robot_init_todo.md
├── reference
│   ├── README.md
│   ├── basic_websocket_flow
│   ├── legacy_http_policy_client
│   └── piper_ros_nodes
├── scripts
│   ├── camera_video_recorder.py
│   ├── piper_fastwam_policy_server.py
│   ├── piper_realrobot_replay_server.py
│   ├── piper_openpi_policy_server.py
│   ├── run_piper_fastwam_server.sh
│   ├── websocket_policy_client.py
│   └── ws_policy_protocol.py
└── tests
    └── test_camera_video_recorder.py
```

## 已验证配置

开发服务器：

```text
SSH: ssh -p 3763 root@10.40.1.215
OpenPI: /bh/zbh_self/projects/openpi-official
GPU: 1 × A800
Config: pi05_realrobot
Checkpoint: /bh/media/ZBH/pi05_realrobot_jax_ft_8xA800_20260829/checkpoints/pi05_realrobot/pi05-realrobot-jax-8xa800-30k-20260830-r7-overlay/99999
```

任务：

```text
Assemble the puzzle pieces on the table to complete the puzzle
```

2026-07-28/29 的真机验证结果：

- 三路相机均为 1280×720、30 FPS，每路短测写入 150 帧。
- 三路均为 0 丢帧，FFmpeg 正常退出，MP4 可由 FFprobe 正常读取。
- 录像目录成功从机器人上传到服务器。
- 单张 A800 成功加载 checkpoint。
- 一次真实 policy 请求成功返回左右臂各 10 个 waypoint。

## 三端通信结构

```text
Robot
  websocket_policy_client.py
  ws://<local-computer-robot-network-ip>:18001
        |
        v
Local computer
  SSH tunnel: 0.0.0.0:18001 -> server 127.0.0.1:7081
        |
        v
Development server
  piper_openpi_policy_server.py
  OpenPI checkpoint inference on one A800
```

`10.13.10.63` 是已验证时本地电脑在机器人网络中的 IP，仅作为下面命令的示例。
网络变化后，应先确认本地电脑当前 IP，并同步修改机器人端的 `--uri`。

## 0. 机器人 ROS 初始化

在启动 policy client 前，必须完成机器人端 ROS 初始化。完整启动顺序、节点、
topic 映射和排障见
[`docs/agilex_robot_ros_startup.md`](docs/agilex_robot_ros_startup.md)。

当前 joint-control 流程已验证的 Piper 模式为：

```bash
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=false
```

`mode:=1` 用于接收上层 joint action。client 读取：

```text
/puppet/joint_left
/puppet/joint_right
```

并在完整实验中发布：

```text
/master/joint_left
/master/joint_right
```

启动 client 前应确认三路压缩相机 topic 都在持续出图：

```text
/camera_f/color/image_raw/compressed
/camera_l/color/image_raw/compressed
/camera_r/color/image_raw/compressed
```

## 1. 部署脚本

服务器端脚本应位于 OpenPI checkout：

```text
/bh/zbh_self/projects/openpi-official/scripts/piper_openpi_policy_server.py
```

从仓库根目录更新服务器脚本：

```bash
scp -P 3763 scripts/piper_openpi_policy_server.py \
  root@10.40.1.215:/bh/zbh_self/projects/openpi-official/scripts/
```

机器人端目录：

```text
/home/agilex/piper-openpi-real-robot/scripts
```

机器人可连接时，从仓库根目录更新 client：

```bash
ssh agilex@10.13.11.215 \
  'mkdir -p /home/agilex/piper-openpi-real-robot/scripts /home/agilex/piper-openpi-real-robot/recordings'

scp \
  scripts/camera_video_recorder.py \
  scripts/websocket_policy_client.py \
  scripts/ws_policy_protocol.py \
  agilex@10.13.11.215:/home/agilex/piper-openpi-real-robot/scripts/
```

不要把服务器端脚本和机器人端脚本混用。

## 2. 三终端无动作通信测试

按终端 1 → 终端 2 → 终端 3 的顺序执行。本节会发送一次真实 observation，
得到一次真实 policy action，并录制三路相机，但不会向机械臂发布 action。

### 终端 1：启动 OpenPI Policy Server

在本地电脑运行：

```bash
ssh -F /dev/null -p 3763 -tt root@10.40.1.215 "cd /bh/zbh_self/projects/openpi-official && \
export CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
PYTHONPATH=/bh/zbh_self/projects/openpi-official/src:/bh/zbh_self/projects/openpi-official/packages/openpi-client/src && \
exec /bh/media/section/iclr_proj/third_party/openpi/.venv/bin/python -u \
scripts/piper_openpi_policy_server.py \
--host 127.0.0.1 \
--port 7081 \
--openpi-root /bh/zbh_self/projects/openpi-official \
--config pi05_realrobot \
--checkpoint /bh/media/ZBH/pi05_realrobot_jax_ft_8xA800_20260829/checkpoints/pi05_realrobot/pi05-realrobot-jax-8xa800-30k-20260830-r7-overlay/99999 \
--horizon 50 \
--action-dt 0.1 \
--max-joint-step 0.05"
```

看到下面这行后，再启动终端 2：

```text
piper OpenPI policy server listening on ws://127.0.0.1:7081
```

### 终端 2：建立 SSH Tunnel

在本地电脑运行并保持该终端开启：

```bash
ssh -F /dev/null -p 3763 -N -g \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=accept-new \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L 0.0.0.0:18001:127.0.0.1:7081 \
  root@10.40.1.215
```

正常情况下该命令不输出内容。它在本地电脑监听 `18001`，并转发到服务器
`127.0.0.1:7081`。

### 终端 3：机器人 Dry Run

在本地电脑运行。`-A` 必须保留：录像上传程序使用转发后的 SSH agent，以
`BatchMode=yes` 连接开发服务器。

```bash
ssh -A -tt agilex@10.13.11.16 "source /opt/ros/noetic/setup.bash && \
source /home/agilex/agilex_ws/devel/setup.bash && \
source /home/agilex/cobot_magic/camera_ws/devel/setup.bash && \
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash && \
exec /home/agilex/miniconda3/envs/xrocs-env/bin/python \
/home/agilex/piper-openpi-real-robot/scripts/websocket_policy_client.py \
--uri ws://10.13.10.63:18001 \
--instruction 'Assemble the puzzle pieces on the table to complete the puzzle' \
--source ros \
--executor mock \
--max-steps 1 \
--camera-topic head=/camera_f/color/image_raw/compressed \
--camera-topic left_wrist=/camera_l/color/image_raw/compressed \
--camera-topic right_wrist=/camera_r/color/image_raw/compressed \
--compressed-images \
--require-images \
--record-videos \
--record-upload-host root@10.40.1.215 \
--record-upload-port 3763 \
--record-upload-dir /bh/media/section/iclr_proj/zbh/piper_openpi_real_robot_videos"
```

该命令使用真实 ROS observation 和真实 checkpoint，但由于：

```text
--executor mock
没有 --execute-actions
```

所以不会发布机器人 action。通信成功时可看到：

```text
connected to policy server
[dry-run] action execution disabled
[record] uploaded recording: ...
```

服务器端同时会输出 `[step 0]`。

## 3. 完整真机实验

先成功完成第 2 节的单步 dry run，确认 action、通信和视频上传均正常，再运行
本节。终端 1 和终端 2 的命令保持不变，只替换终端 3：

```bash
ssh -A -tt agilex@10.13.11.215 "source /opt/ros/noetic/setup.bash && \
source /home/agilex/agilex_ws/devel/setup.bash && \
source /home/agilex/cobot_magic/camera_ws/devel/setup.bash && \
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash && \
exec /home/agilex/miniconda3/envs/xrocs-env/bin/python \
/home/agilex/piper-openpi-real-robot/scripts/websocket_policy_client.py \
--uri ws://10.13.10.63:18001 \
--instruction 'Assemble the puzzle pieces on the table to complete the puzzle' \
--source ros \
--executor ros \
--execute-actions \
--enable-on-start \
--camera-topic head=/camera_f/color/image_raw/compressed \
--camera-topic left_wrist=/camera_l/color/image_raw/compressed \
--camera-topic right_wrist=/camera_r/color/image_raw/compressed \
--compressed-images \
--require-images \
--record-videos \
--record-upload-host root@10.40.1.215 \
--record-upload-port 3763 \
--record-upload-dir /bh/media/section/iclr_proj/zbh/piper_openpi_real_robot_videos"
```

完整实验会：

1. 等待左右臂状态和三路相机就绪。
2. 使能机械臂，并按现有 client 行为平滑回到 zero-joint home。
3. 持续请求 policy，并发布服务器返回的左右臂 waypoint。
4. 同时把全部三路压缩相机 callback 写入独立 MP4。

### 停止与录像上传

完整实验默认启用退出 hold：

1. 第一次按 `Ctrl-C`：停止推理，进入当前位置 hold。
2. 第二次按 `Ctrl-C`：退出 hold。
3. client 关闭三个 FFmpeg writer，写入 `manifest.json`。
4. client 把完整 session 上传到开发服务器。

不要在 hold 阶段直接关闭终端，否则录像可能没有机会完成封装和上传。本命令没有
加入 `--disable-on-exit`，退出 hold 后机器人保持 enabled；按现场安全流程处理
后续失能。

## FastWAM Puzzle 单任务真机部署（与 OpenPI 独立）

这一节部署的是 Puzzle 单任务的 FastWAM 微调结果，不会替换前面的 OpenPI
服务端。开发机固定为：

```text
ssh -p 3763 root@10.40.1.215
FastWAM 工程: /bh/zbh_self/projects/fastwam-official
FastWAM 环境: /bh/zbh_self/envs/fastwam
Piper 服务工程: /bh/zbh_self/projects/piper-openpi-real-robot-20260831
```

服务端只使用这次训练的最终模型和同一份训练数据统计：

```text
checkpoint: /bh/media/ZBH/fastwam-realrobot-ft-2node8gpu-bs1-chunk64-15ep-v2/checkpoints/weights/step_150000.pt
statistics: /bh/media/ZBH/fastwam-realrobot-ft-2node8gpu-bs1-chunk64-15ep-v2/dataset_stats.json
config: /bh/zbh_self/projects/piper-openpi-real-robot-20260831/configs/piper_fastwam_puzzle.yaml
```

FastWAM 的 prompt 固定为 `Assemble the puzzle pieces on the table to complete the puzzle`，
与该 realrobot 单任务 checkpoint 的训练文本一致。它要求
Piper wire state/action 顺序为：

```text
[L1,L2,L3,L4,L5,L6,Lg,R1,R2,R3,R4,R5,R6,Rg]
```

该 realrobot checkpoint 内部保持训练时的 `[L1..L6,Lg,R1..R6,Rg]` 顺序；server
会在安全边界处显式转换，不能在 robot client、ROS topic 或既有 OpenPI 服务端中
手工修改顺序。

### FastWAM 的安全前置条件

`configs/piper_fastwam_puzzle.yaml` 中的 `joint_lower`、`joint_upper` 和
`max_delta` 必须由**本次实际接入的 Piper** 的活动 URDF/driver 配置与现场验证
得出；每项均须按上述 Piper wire 顺序填写 14 个有限数值。仓库不提供、也不能
用通用网络资料替代这些数值。启动物理推理前，服务端会拒绝缺失、非有限或上下界
无效的配置。

FastWAM 每次模型推理固定生成 64 个候选动作。真机采用滚动规划：服务端只返回最前
面的 `execution_horizon: 64` 个点（30 Hz 下约为 2.13 秒），robot client 连续执行这
64 个点，随后采集新的三路图像和关节状态并重新推理下一段。每一个实际返回并可能
发布的 waypoint 都经过绝对关节限位和逐点增量限制。OpenPI 继续绑定 `127.0.0.1:7081`；本次
realrobot FastWAM 服务绑定 `127.0.0.1:7083`，两个服务可以并行保留。

机器人使用 `--compressed-images` 时，默认直接传输相机原始 JPEG。FastWAM 命令必须
额外传入 `--transport-image-profile fastwam`：robot client 会在发送前把 head 缩至
`320x256`、两个 wrist 缩至 `160x128`，再以 JPEG 发送。服务端保持相同的模型输入
拼图尺寸，因此这只降低传输量，不改变模型视觉几何；OpenPI 命令不传此参数，保持原
有传输行为。

边界投影不会放宽任何物理限位：有限的手臂原始预测最多可校正 `0.10 rad`、夹爪最多
可校正 `5 mm` 到真实关节边界，随后仍受每个 30 Hz waypoint `0.01 rad` 的增量上限
约束。校正量不是新的运动范围；所有返回值仍在 `joint_lower` 和 `joint_upper` 内。
每次投影都会记录 warning。若连续三次请求都需要至少 `0.05 rad` 的手臂校正，服务端
停止返回动作；超过上述校正上限的单次预测也会拒绝整条动作，绝不
发布。

### 1. 服务端预检（不监听端口、不发送机器人动作）

先在本地电脑运行。这会在 `3763` 的 A800 上加载 FastWAM 基座、`step_150000.pt`
和 `dataset_stats.json`，但不会打开 WebSocket 服务，也不会与机器人通信：

```bash
ssh -F /dev/null -p 3763 -tt root@10.40.1.215 "
  source /bh/zbh_self/miniforge3/etc/profile.d/conda.sh &&
  conda activate /bh/zbh_self/envs/fastwam &&
  cd /bh/zbh_self/projects/piper-openpi-real-robot-20260831 &&
  CUDA_VISIBLE_DEVICES=0 exec bash scripts/run_piper_fastwam_server.sh --preflight-only
"
```

预检必须确认：使用一张预期的 A800、模型 action horizon 为 64、视频帧数为 17、
checkpoint 和归一化统计均加载成功。任何加载或配置错误都应在此处修复；不要跳过
预检进入真机阶段。

### 2. 启动 FastWAM 服务端

预检成功后，在单独终端以前台方式启动服务端并保留该终端：

```bash
ssh -F /dev/null -p 3763 -tt root@10.40.1.215 "
  source /bh/zbh_self/miniforge3/etc/profile.d/conda.sh &&
  conda activate /bh/zbh_self/envs/fastwam &&
  cd /bh/zbh_self/projects/piper-openpi-real-robot-20260831 &&
  CUDA_VISIBLE_DEVICES=0 exec bash scripts/run_piper_fastwam_server.sh
"
```

`scripts/run_piper_fastwam_server.sh` 只负责激活 FastWAM 环境，并将额外参数原样
传给 `piper_fastwam_policy_server.py`。它绑定 `127.0.0.1:7083`，因此必须通过
下一步 tunnel 才能由机器人访问。

### 3. 建立 SSH tunnel

在本地电脑另一终端运行并保持开启：

```bash
ssh -F /dev/null -p 3763 -N -g \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=accept-new \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L 0.0.0.0:18003:127.0.0.1:7083 \
  root@10.40.1.215
```

其中 `10.13.10.63` 仍只是下面示例中的本地电脑机器人网络 IP；实验前以本机实际
IP 替换。

### 4. 机器人端单步 dry run（不会发布动作）

完成 ROS 初始化后，在本地电脑运行。该命令读取真实三路相机和 joint state，并让
FastWAM 真实推理一次；但 `--executor mock` 且**没有** `--execute-actions`，因此
不会向 `/master/joint_left` 或 `/master/joint_right` 发布任何动作：

```bash
ssh -A -tt agilex@10.13.11.215 "source /opt/ros/noetic/setup.bash && \
source /home/agilex/agilex_ws/devel/setup.bash && \
source /home/agilex/cobot_magic/camera_ws/devel/setup.bash && \
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash && \
exec /home/agilex/miniconda3/envs/xrocs-env/bin/python \
/home/agilex/piper-openpi-real-robot/scripts/websocket_policy_client.py \
--uri ws://10.13.10.63:18003 \
--instruction 'crimp' \
--source ros \
--executor mock \
--max-steps 1 \
--camera-topic head=/camera_f/color/image_raw/compressed \
--camera-topic left_wrist=/camera_l/color/image_raw/compressed \
--camera-topic right_wrist=/camera_r/color/image_raw/compressed \
--compressed-images \
--transport-image-profile fastwam \
--require-images \
--record-videos \
--record-upload-host root@10.40.1.215 \
--record-upload-port 3763 \
--record-upload-dir /bh/zbh_ckp/runs/fastwam/piper_crimp_real_robot"
```

开始任何受控执行前，操作者必须逐项确认：三张图像正确对应 `head`、`left_wrist`、
`right_wrist`；服务端拼图为 `[3,384,320]`；每臂 state 均为“6 个关节 + 夹爪”且
顺序已核对；prompt 为 `crimp`；返回的 64 个 waypoint 全部在已验证的关节范围内；以及
真实端到端延迟低于配置上限。还应保留该次请求的服务端日志和三路录像以供复核。

只有完成以上检查、机器人场景已按 Puzzle/`crimp` 数据布置、操作者站在急停旁，且
已经从一个 waypoint 开始的受控验证成功后，才可在 robot client 命令中将
`--executor mock` 改为 `--executor ros` 并加入 `--execute-actions`。不要为了快速
完成任务跳过单 waypoint 或短 action chunk 阶段，也不要把 FastWAM 用于未训练的
自然语言任务。

本次 Puzzle 策略的训练轨迹不从全零位开始。真机命令必须传入：

```bash
--start-pose-config /home/agilex/piper-openpi-real-robot/configs/piper_fastwam_puzzle_start_pose.json \
```

该配置是训练集里真实记录的 episode 159 起点，而非逐维拼出的统计姿态。最初的
episode 242 medoid 在现场图像下曾把左臂 J2 预测到硬件下限以下；episode 159 将
该关节的起始余量从约 `0.00609 rad` 增加到 `0.01526 rad`，并使用本 checkpoint、
对应训练首帧和三个随机推理重复筛选过前 64 个 waypoint。离线筛选不能替代现场
图像验证。客户端会以不超过 `0.15 rad/s` 的速度平滑移动到该起始位，并等待真实
反馈：机械臂误差不超过 `0.01 rad`、夹爪误差不超过 `0.001 m` 后才连接 FastWAM。
配置错误、移动超时、反馈不到位或 ROS fault 都会阻止推理。使用该参数时不要再传
`--no-home-on-start`；OpenPI 未传该参数时仍保留原有 home 行为。

首次现场检查可使用 `--executor ros --execute-actions --max-steps 0`：它只移动并
验证 episode-159 起始位，不执行模型动作。起始位检查和现场图像推理都通过后，完整
FastWAM 命令使用 `--max-steps 25`。一次动作后，客户端会以最后**实际发布的
waypoint**（而非可能滞后的 ROS 观测）持续 hold。此时按 `Ctrl-C` 只退出 hold
循环，机械臂保持使能和最后目标位姿；异常轨迹或故障时仍应优先使用现场急停。

## realrobot RoboMIND Episode Replay

`scripts/piper_realrobot_replay_server.py` 不加载模型，只从
`/bh/media/pku-data/realrobot` 自动发现 LeRobot/RoboMIND 数据集，读取一个指定
episode 的 30 Hz joint 轨迹，并通过现有 JSON WebSocket 协议返回一次完整 action
chunk。默认使用 `puppet` stream；需要回放 master command stream 时传
`--action-source master`。服务进程会在内存中保留已校验轨迹，可重复响应多个请求，
不需要为了再次 replay 重启；单次完整回放仍建议 client 设置 `--max-steps 1`，避免
重复执行同一条轨迹。

先查看可用 episode，再做不监听端口的预检：

```bash
/bh/zbh_self/envs/fastwam/bin/python \
  scripts/piper_realrobot_replay_server.py --list-episodes

/bh/zbh_self/envs/fastwam/bin/python \
  scripts/piper_realrobot_replay_server.py \
  --episode-index 0 --preflight-only
```

预检通过后启动服务（默认 `127.0.0.1:7082`）：

```bash
/bh/zbh_self/envs/fastwam/bin/python \
  scripts/piper_realrobot_replay_server.py \
  --episode-index 0 --action-source puppet
```

可用现有 client 做无动作协议测试：

```bash
/bh/zbh_self/envs/fastwam/bin/python scripts/websocket_policy_client.py \
  --uri ws://127.0.0.1:7082 \
  --instruction 'realrobot episode 0 replay' \
  --source mock --executor mock --max-steps 1
```

机器人侧连接 tunnel 的命令如下。先在机器人 shell 中确认 ROS 状态 topic 正常：

```bash
source /opt/ros/noetic/setup.bash
source /home/agilex/agilex_ws/devel/setup.bash
source /home/agilex/cobot_magic/camera_ws/devel/setup.bash
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
rostopic hz /puppet/joint_left
rostopic hz /puppet/joint_right
```

然后使用机器人端 client 做 dry-run（把 `10.13.10.63` 换成本地电脑在机器人网络
中的实际 IP）：

```bash
cd /home/agilex/piper-openpi-real-robot
exec /home/agilex/miniconda3/envs/xrocs-env/bin/python \
  scripts/websocket_policy_client.py \
  --uri ws://10.13.10.63:18002 \
  --instruction 'realrobot episode 0 replay' \
  --source ros --executor mock --max-steps 1
```

确认首帧姿态、关节范围和急停流程后，才将 executor 切换为 ROS 执行。机器人必须
已经处于所选 episode 的首帧姿态；`--no-home-on-start` 用于避免 client 先移动到
默认 zero-joint home：

```bash
cd /home/agilex/piper-openpi-real-robot
exec /home/agilex/miniconda3/envs/xrocs-env/bin/python \
  scripts/websocket_policy_client.py \
  --uri ws://10.13.10.63:18002 \
  --instruction 'realrobot episode 0 replay' \
  --source ros --executor ros \
  --execute-actions --enable-on-start \
  --no-home-on-start --max-steps 1
```

服务端只校验有限值、连续 frame index、30 Hz 时间戳和 Piper 绝对关节范围；gripper
边界固定为 `0.0–0.1`，数据中不超过 `0.001` 的 signed zero drift 按容差接受，所有
waypoint 数值不做平滑或裁剪。更大的负值或任何正向越过 `0.1` 的值都会在预检拒绝。
真机执行前必须在机器人端完成 ROS 初始化、确认 episode 首帧与当前姿态匹配，并先
使用 client 默认 dry-run；不要把未验证的录制轨迹直接发送到实体机器人。dry-run
之后可以直接再次运行真实执行命令，无需重启 replay server。

### 5. 停止流程

出现异常物理轨迹、状态顺序疑问、越限、推理超时或 ROS fault 时，首先按现场安全
流程触发可触达的物理急停；不要继续发送 policy 请求。正常结束时，先在 robot
client 终端按 `Ctrl-C`，待其完成 hold/录像收尾后，再在 FastWAM 服务端和 SSH
tunnel 的终端分别按 `Ctrl-C`。不要使用宽泛的 `pkill`，并保留 server 日志与上传
录像以便追溯。

## 4. 视频文件

机器人本地根目录：

```text
/home/agilex/piper-openpi-real-robot/recordings
```

服务器上传根目录：

```text
/bh/media/section/iclr_proj/zbh/piper_openpi_real_robot_videos
```

每次运行创建一个 `domino_YYYYMMDD_HHMMSS` session：

```text
domino_YYYYMMDD_HHMMSS/
├── head.mp4
├── left_wrist.mp4
├── right_wrist.mp4
├── head.ffmpeg.log
├── left_wrist.ffmpeg.log
├── right_wrist.ffmpeg.log
└── manifest.json
```

`manifest.json` 记录时长、FPS、收到/写入/丢弃帧数、FFmpeg 返回码和上传目标。
即使上传失败，机器人本地 session 也不会删除。确认服务器文件完整后，可再人工
清理机器人上的旧录像。

## 5. 文件说明

### `scripts/camera_video_recorder.py`

每路相机使用独立 bounded queue 和 writer thread。ROS callback 只提交 JPEG
payload；队列满时丢弃最旧帧，避免视频写盘阻塞控制与推理链路。FFmpeg 直接把
相机 MJPEG 写入 MP4，不重复编码。

### `scripts/piper_openpi_policy_server.py`

把机器人 JSON 请求转换为 OpenPI observation，包含 state、三路图像和 prompt；
加载 checkpoint 推理，再将 OpenPI action 转换为 client 需要的 joint action
chunk。

### FastWAM action limiting

FastWAM 的有限关节预测先裁剪到已验证的 Piper 绝对关节范围，再按前一个
已接受 waypoint 的 `max_delta` 顺序裁剪。该行为与 Pi05 的顺序限速语义一致：
有限越界值只记录告警，不会中止请求；NaN/Inf、格式错误、模型错误和超时仍会
返回 no-action。

### `scripts/websocket_policy_client.py`

读取左右臂 joint state、三路压缩相机和 Piper arm status；发送 policy 请求，
执行或 dry-run 返回的 action，并管理录像收尾与上传。

### `scripts/ws_policy_protocol.py`

生成 `request_id`，编码 JPEG payload，并构造 WebSocket JSON request。

## 6. Action Space

当前 WebSocket 推理链路使用 joint action：

- client 读取 `sensor_msgs/JointState.position` 作为 state。
- server 输出 `left_arm` / `right_arm` waypoint。
- client 把左右臂 waypoint 发布成 `JointState.position`。

Piper ROS 节点还提供 EEF 接口，例如 `/puppet/end_pose`、
`/puppet/end_pose_euler` 和 `/pos_cmd`，但当前 client/server 没有接入 EEF
action schema。

## 7. 运行前检查

- 确认 checkpoint 路径存在。
- 确认所选 A800 空闲。
- 确认服务器 `127.0.0.1:7081` 没有旧 policy server 占用。
- 确认 tunnel 使用的本地端口没有被其他程序占用。
- 确认机器人能访问本地电脑的机器人网络 IP。
- 确认 Piper 为 `mode:=1`。
- 确认左右臂 joint state 和三路相机 topic 持续有数据。
- 确认机械臂周围净空、急停可触达。
- 每次完整实验前只做一次第 2 节 dry run，避免增加不必要的流程。

## 8. 测试

本地运行：

```bash
python3 -m unittest -v tests.test_camera_video_recorder
python3 -m unittest -v tests.test_piper_realrobot_replay_server
python3 -m py_compile \
  scripts/camera_video_recorder.py \
  scripts/websocket_policy_client.py \
  scripts/piper_openpi_policy_server.py \
  scripts/piper_realrobot_replay_server.py \
  scripts/ws_policy_protocol.py
```

当前测试覆盖：

- 三路 FFmpeg 命令。
- bounded queue 丢帧策略。
- manifest 统计。
- SSH/SCP 上传命令。
- 上传失败时保留本地 session。
- client 录像参数默认值。
- 压缩相机 callback 原始 JPEG payload。

## Reference Scripts

`reference/` 保存早期基础流程和 Piper ROS/CAN 参考节点，只用于理解和迁移，不是
当前三端 OpenPI 流程的必需文件。详见
[`reference/README.md`](reference/README.md)。

## 后续待补

- EEF action 版本设计。
- 上电、急停、夹爪和 home 的完整硬件安全清单。
