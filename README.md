# Piper Real Robot Model Workflows

这个仓库保存松灵 Piper 双臂机器人运行不同模型 checkpoint 的真机部署流程。
目前包含两条相互独立、共享 Piper ROS/WebSocket 约定的链路：

- **OpenPI pi0.5**：原有完整流程，脚本位于 `scripts/`，操作说明在本页。
- **UVA-DiT / WAM**：Puzzle checkpoint 的 server、机器人 client、录制与 replay
  工具位于 [`models/uva_dit/`](models/uva_dit/README.md)。

不要混用不同模型的 server 和 client。模型权重、训练代码、HDF5 轨迹和视频均不
进入本仓库；各模型目录只保存 Piper 部署适配层与可复现的操作细节。

以下是已经验证的 OpenPI pi0.5 链路：

- 开发服务器使用一张 A800 加载 OpenPI checkpoint，运行 JSON WebSocket policy server。
- 本地电脑通过 SSH tunnel，把机器人可访问的本地端口转发到服务器。
- 机器人端 WebSocket client 读取左右臂 joint state 和三路相机，接收 action chunk。
- 通信测试默认不执行动作；完整实验才向左右臂发布 joint action。
- client 全程录制三路相机，结束时封装 MP4，并上传到开发服务器。

当前 action space 是 joint，不是 EEF。机器人端通信协议不强绑定 OpenPI，但
`piper_openpi_policy_server.py` 是 OpenPI checkpoint 专用适配器。

## 目录

```text
.
├── README.md
├── docs
│   ├── agilex_robot_ros_startup.md
│   ├── architecture.md
│   └── setup_robot_init_todo.md
├── models
│   └── uva_dit
│       ├── README.md
│       ├── client
│       ├── server
│       └── requirements-piper.txt
├── reference
│   ├── README.md
│   ├── basic_websocket_flow
│   ├── legacy_http_policy_client
│   └── piper_ros_nodes
├── scripts
│   ├── camera_video_recorder.py
│   ├── piper_openpi_policy_server.py
│   ├── websocket_policy_client.py
│   └── ws_policy_protocol.py
└── tests
    └── test_camera_video_recorder.py
```

## 已验证配置

开发服务器：

```text
SSH: ssh -p 3763 root@10.40.1.215
OpenPI: /bh/media/section/iclr_proj/zbh/openpi
GPU: 1 × A800
Config: pi05_cmap_all_tasks_v21
Checkpoint: /bh/media/section/iclr_proj/zbh/openpi/training_runs/pi05_all_tasks_openpi_v21_8gpu_b512_80k/checkpoints/pi05_cmap_all_tasks_v21/pi05_all_tasks_openpi_v21_pi05_8gpu_b512_80k_correct_20260718/60000
```

任务：

```text
Arrange the dominoes on the table in a horizontal row and then knock them over.
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
/bh/media/section/iclr_proj/zbh/openpi/scripts/piper_openpi_policy_server.py
```

从仓库根目录更新服务器脚本：

```bash
scp -P 3763 scripts/piper_openpi_policy_server.py \
  root@10.40.1.215:/bh/media/section/iclr_proj/zbh/openpi/scripts/
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
ssh -F /dev/null -p 3763 -tt root@10.40.1.215 "cd /bh/media/section/iclr_proj/zbh/openpi && \
export CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
PYTHONPATH=/bh/media/section/iclr_proj/zbh/openpi/src:/bh/media/section/iclr_proj/zbh/openpi/packages/openpi-client/src && \
exec /bh/media/section/iclr_proj/third_party/openpi/.venv/bin/python -u \
scripts/piper_openpi_policy_server.py \
--host 127.0.0.1 \
--port 7081 \
--openpi-root /bh/media/section/iclr_proj/zbh/openpi \
--config pi05_cmap_all_tasks_v21 \
--checkpoint /bh/media/section/iclr_proj/zbh/openpi/training_runs/pi05_all_tasks_openpi_v21_8gpu_b512_80k/checkpoints/pi05_cmap_all_tasks_v21/pi05_all_tasks_openpi_v21_pi05_8gpu_b512_80k_correct_20260718/60000 \
--horizon 10 \
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
ssh -A -tt agilex@10.13.11.215 "source /opt/ros/noetic/setup.bash && \
source /home/agilex/agilex_ws/devel/setup.bash && \
source /home/agilex/cobot_magic/camera_ws/devel/setup.bash && \
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash && \
exec /home/agilex/miniconda3/envs/xrocs-env/bin/python \
/home/agilex/piper-openpi-real-robot/scripts/websocket_policy_client.py \
--uri ws://10.13.10.63:18001 \
--instruction 'Arrange the dominoes on the table in a horizontal row and then knock them over.' \
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
--instruction 'Arrange the dominoes on the table in a horizontal row and then knock them over.' \
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
python3 -m py_compile \
  scripts/camera_video_recorder.py \
  scripts/websocket_policy_client.py \
  scripts/piper_openpi_policy_server.py \
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
