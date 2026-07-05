# Piper OpenPI Real Robot Workflow

这个仓库记录松灵 Piper 机器人真机实验流程。当前版本的主链路是：

- 服务器端加载 OpenPI checkpoint，启动 JSON WebSocket policy server。
- 本地电脑开启 SSH tunnel，把机器人可访问的本地端口转发到服务器 policy server。
- 机器人端运行 WebSocket client，读取 ROS joint/camera topic，接收 action chunk，并发布 joint action。

当前可用流程主要面向 OpenPI 模型推理。机器人端 client 的通信协议本身不强绑定 OpenPI，但本仓库里的 `piper_openpi_policy_server.py` 是 OpenPI checkpoint 专用适配器。

## 目录

```text
.
├── README.md
├── docs
│   ├── architecture.md
│   └── setup_robot_init_todo.md
├── ros_nodes
│   └── piper_start_ms_node.py
└── scripts
    ├── piper_openpi_policy_server.py
    ├── websocket_policy_client.py
    └── ws_policy_protocol.py
```

## 当前状态

- 机器初始化流程还没有整理进来，后续补齐。
- 当前 action space 是 joint，不是 EEF。
- 目前 server 侧加载 OpenPI checkpoint。
- 机器人端通过 WebSocket 连接本地电脑暴露的 tunnel 地址。
- 本地电脑只是通信桥，不运行模型。

## 三端通信结构

```text
Robot
  websocket_policy_client.py
  ws://10.13.12.85:18000
        |
        v
Local computer
  SSH tunnel: 0.0.0.0:18000 -> server 127.0.0.1:7081
        |
        v
Server
  piper_openpi_policy_server.py
  OpenPI checkpoint inference
```

## 0. 机器初始化

待补齐。这里后续记录：

- Piper 上电和急停状态检查。
- ROS master / CAN / 相机启动顺序。
- 左右臂 topic remap 规则。
- 夹爪标定、home、enable 流程。
- 真机安全检查清单。

## 1. 服务器端启动 OpenPI Policy Server

先登录服务器，开一个终端保持运行：

```bash
ssh -p 3164 -o ClearAllForwardings=yes root@10.40.1.219
```

进入 OpenPI 目录：

```bash
cd /media/section/iclr_proj/zbh/openpi
```

启动 policy server：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
.venv/bin/python scripts/piper_openpi_policy_server.py \
  --host 127.0.0.1 \
  --port 7081 \
  --config pi05_s40_3 \
  --checkpoint /media/section/iclr_proj/zbh/openpi/training_runs/pi05_s40_3/checkpoints/pi05_s40_3/s40_3_pi05_8gpu_50k_fixed_indices/49999 \
  --horizon 10 \
  --action-dt 0.1 \
  --max-joint-step 0.05
```

说明：

- `--host 127.0.0.1` 表示只在服务器本机监听，外部通过 SSH tunnel 访问。
- `--port 7081` 是服务器内部 WebSocket 端口。
- `--checkpoint` 按实际实验 checkpoint 修改。
- `--max-joint-step 0.05` 用于限制每个 waypoint 的关节跳变。

## 2. 本地电脑开启 SSH Tunnel

本地电脑另开一个终端，保持 tunnel 运行：

```bash
ssh -F /dev/null -p 3164 -N -g \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=accept-new \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L '0.0.0.0:18000:127.0.0.1:7081' \
  root@10.40.1.219
```

说明：

- 本地电脑监听 `0.0.0.0:18000`。
- 请求会转发到服务器的 `127.0.0.1:7081`。
- 机器人需要能访问本地电脑的 IP，例如下面命令中的 `10.13.12.85`。

## 3. 机器人端启动 WebSocket Client

在机器人终端运行：

```bash
python3 websocket_policy_client.py \
  --uri ws://10.13.12.85:18000 \
  --instruction "Arrange the dominoes on the table in a horizontal row and then knock them over." \
  --source ros \
  --executor ros \
  --execute-actions \
  --waypoint-sleep 1.0 \
  --home-hz 20 \
  --home-min-duration 2.0 \
  --home-max-duration 20.0 \
  --home-joint-speed 0.25 \
  --hold-hz 20 \
  --left-observation-topic /puppet/joint_left \
  --right-observation-topic /puppet/joint_right \
  --left-action-topic /master/joint_left \
  --right-action-topic /master/joint_right \
  --camera-topic head=/camera_f/color/image_raw/compressed \
  --camera-topic left_wrist=/camera_l/color/image_raw/compressed \
  --camera-topic right_wrist=/camera_r/color/image_raw/compressed \
  --compressed-images \
  --require-images \
  --enable-on-start
```

启动后，机器人端会周期性读取当前 joint state 和三路相机图像，通过 WebSocket 请求 policy server。服务器根据当前 OpenPI checkpoint 返回 action chunk，机器人端再发布到左右臂 action topic。

## 文件说明

### `scripts/piper_openpi_policy_server.py`

服务器端入口。它把机器人端 JSON 请求转换为 OpenPI observation，加载本地 OpenPI checkpoint 推理，并把 OpenPI action 转换成机器人端 client 需要的 joint action chunk。

### `scripts/websocket_policy_client.py`

机器人端入口。它从 ROS topic 读取：

- 左右臂 joint state。
- 三路压缩相机图像。
- Piper arm status。

收到 server 返回的 action 后，它向左右臂 joint action topic 发布 `sensor_msgs/JointState`。

### `scripts/ws_policy_protocol.py`

WebSocket JSON 请求协议工具。主要负责：

- 生成 `request_id`。
- 编码图像为 JSON 可传输的 base64 JPEG payload。
- 构造 policy request。

### `ros_nodes/piper_start_ms_node.py`

Piper ROS/CAN 桥接节点。它可以发布 puppet joint/end-pose 状态，也可以订阅 master joint 或 `/pos_cmd` 控制真机。

注意：这个节点默认使用 `/puppet/joint_states` 和 `/master/joint_states`，而当前 WebSocket client 默认使用 `/puppet/joint_left`、`/puppet/joint_right`、`/master/joint_left`、`/master/joint_right`。实际使用时需要启动左右臂两个实例并做 ROS remap，或后续把 topic 改成参数化。

## Action Space

当前 WebSocket 推理链路使用 joint action。

证据：

- 机器人端读取 `sensor_msgs/JointState.position` 作为 state。
- server 输出 `left_arm` / `right_arm` waypoint。
- client 把 `left_arm` / `right_arm` 直接发布成 `JointState.position`。

Piper ROS 节点里有 EEF 相关接口，例如 `/puppet/end_pose`、`/puppet/end_pose_euler` 和 `/pos_cmd`，但当前 WebSocket client/server 还没有接入 EEF action schema。

## 运行前检查

建议每次真机运行前确认：

- 服务器 OpenPI checkpoint 路径存在。
- policy server 已在服务器 `127.0.0.1:7081` 启动。
- 本地电脑 tunnel 正在运行，且机器人能访问本地电脑 IP。
- 机器人端 ROS topic 都有数据。
- 三路相机图像 topic 可用。
- `/enable_flag` 行为符合当前实验预期。
- 初次测试时先去掉 `--execute-actions` 做 dry run。

## 后续待补

- 机器初始化流程。
- 左右臂 ROS launch/remap 示例。
- EEF action 版本设计。
- checkpoint 和任务 instruction 的实验记录模板。
- 常见报错排查。
