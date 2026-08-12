# Piper Real Robot Model Workflows

这个仓库集中保存松灵 Piper 双臂机器人运行不同模型 checkpoint 的真机部署流程。
每个模型拥有独立目录，目录内分别维护自己的 server、client、测试和模型说明；
公共 ROS/网络资料继续放在根目录的 `docs/` 和 `reference/`。

## 模型与目录

```text
.
├── README.md
├── docs/                         公共 ROS、网络和机器人初始化说明
├── models/
│   ├── openpi_pi05/
│   │   ├── README.md             pi0.5 完整部署说明
│   │   ├── server/               OpenPI checkpoint server
│   │   └── client/               Piper ROS client、协议、录像和测试
│   └── uva_dit/
│       ├── README.md             WAM/UVA-DiT 完整部署说明
│       ├── server/               Puzzle checkpoint server、启动器和测试
│       ├── client/               Piper ROS client、录制/replay 和测试
│       └── requirements-piper.txt
└── reference/                    公共底层 WebSocket、ROS/CAN 参考代码
```

| 模型 | Server | Client | 详细说明 |
| --- | --- | --- | --- |
| OpenPI pi0.5 | `models/openpi_pi05/server/` | `models/openpi_pi05/client/` | [`models/openpi_pi05/README.md`](models/openpi_pi05/README.md) |
| WAM / UVA-DiT | `models/uva_dit/server/` | `models/uva_dit/client/` | [`models/uva_dit/README.md`](models/uva_dit/README.md) |

模型权重、训练代码、HDF5 轨迹和视频不进入 Git。不要混用不同模型目录中的
server 与 client。

## 公共三端拓扑

两种模型使用同一套三端网络结构：

```text
机器人端 client
  ws://10.13.10.63:18001
            |
            v
本地电脑 SSH tunnel
  0.0.0.0:18001 -> server 127.0.0.1:7081
            |
            v
开发服务器 model server
```

这里沿用已验证环境：开发服务器 `root@10.40.1.215:3763`，机器人
`agilex@10.13.11.215`，本地电脑机器人网 IP `10.13.10.63`。网络变化时需同步
替换 SSH 地址和机器人命令中的 WebSocket URI。

启动任一模型前，先按照
[`docs/agilex_robot_ros_startup.md`](docs/agilex_robot_ros_startup.md) 初始化 Piper，
并确认左右臂 state 与三路相机 topic 持续发布。

## OpenPI pi0.5 三终端启动

下面是 pi0.5 单步 dry-run。它使用真实相机和真实 checkpoint，但终端 3 使用
`--executor mock`，不会发布机械臂 action。

### pi0.5 终端 1：Policy Server

在本地电脑执行：

```bash
ssh -F /dev/null -p 3763 -tt root@10.40.1.215 "\
cd /bh/zbh_self/projects/piper-openpi-real-robot && \
export CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
PYTHONPATH=/bh/media/section/iclr_proj/zbh/openpi/src:/bh/media/section/iclr_proj/zbh/openpi/packages/openpi-client/src && \
exec /bh/media/section/iclr_proj/third_party/openpi/.venv/bin/python -u \
models/openpi_pi05/server/piper_openpi_policy_server.py \
--host 127.0.0.1 \
--port 7081 \
--openpi-root /bh/media/section/iclr_proj/zbh/openpi \
--config pi05_cmap_all_tasks_v21 \
--checkpoint /bh/media/section/iclr_proj/zbh/openpi/training_runs/pi05_all_tasks_openpi_v21_8gpu_b512_80k/checkpoints/pi05_cmap_all_tasks_v21/pi05_all_tasks_openpi_v21_pi05_8gpu_b512_80k_correct_20260718/60000 \
--horizon 10 \
--action-dt 0.1 \
--max-joint-step 0.05"
```

### pi0.5 终端 2：SSH Tunnel

在本地电脑另开终端并保持运行：

```bash
ssh -F /dev/null -p 3763 -N -g \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=accept-new \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L 0.0.0.0:18001:127.0.0.1:7081 \
  root@10.40.1.215
```

### pi0.5 终端 3：机器人 Dry-run Client

```bash
ssh -A -tt agilex@10.13.11.215 "\
source /opt/ros/noetic/setup.bash && \
source /home/agilex/agilex_ws/devel/setup.bash && \
source /home/agilex/cobot_magic/camera_ws/devel/setup.bash && \
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash && \
exec /home/agilex/miniconda3/envs/xrocs-env/bin/python \
/home/agilex/piper-openpi-real-robot/models/openpi_pi05/client/websocket_policy_client.py \
--uri ws://10.13.10.63:18001 \
--instruction 'Arrange the dominoes on the table in a horizontal row and then knock them over.' \
--source ros \
--executor mock \
--max-steps 1 \
--camera-topic head=/camera_f/color/image_raw/compressed \
--camera-topic left_wrist=/camera_l/color/image_raw/compressed \
--camera-topic right_wrist=/camera_r/color/image_raw/compressed \
--compressed-images \
--require-images"
```

pi0.5 的文件部署、录像参数和真机执行命令见
[`models/openpi_pi05/README.md`](models/openpi_pi05/README.md)。

## WAM / UVA-DiT 三终端启动

下面使用 Puzzle 0811-C 推荐启动器。开发服务器需要同时存在完整 `UVA_dit`
checkout、WAM 推理环境和模型权重；具体路径及 checkpoint 契约见模型 README。

### WAM 终端 1：Policy Server

在本地电脑执行：

```bash
ssh -F /dev/null -p 3763 -tt root@10.40.1.215 "\
cd /bh/zbh_self/projects/piper-openpi-real-robot && \
UVA_DIT_ROOT=/bh/zbh_self/projects/UVA_dit \
GPU=0 \
HOST=127.0.0.1 \
PORT=7081 \
exec bash models/uva_dit/server/run_piper_puzzle_c_noar_policy_server.sh"
```

如不使用启动器内的默认 checkpoint 路径，可在 `exec bash` 前显式设置 `RUN_DIR`、
`CKPT_NAME`、`NORM_STATS`、`PRETRAINED` 和 `QWEN35`。

### WAM 终端 2：SSH Tunnel

在本地电脑另开终端并保持运行：

```bash
ssh -F /dev/null -p 3763 -N -g \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=accept-new \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L 0.0.0.0:18001:127.0.0.1:7081 \
  root@10.40.1.215
```

### WAM 终端 3：机器人 Dry-run Client

```bash
ssh -A -tt agilex@10.13.11.215 "\
source /opt/ros/noetic/setup.bash && \
source /home/agilex/agilex_ws/devel/setup.bash && \
source /home/agilex/cobot_magic/camera_ws/devel/setup.bash && \
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash && \
exec /home/agilex/miniconda3/envs/xrocs-env/bin/python \
/home/agilex/piper-openpi-real-robot/models/uva_dit/client/websocket_policy_client.py \
--uri ws://10.13.10.63:18001 \
--instruction '拼图（圆形）' \
--source ros \
--executor mock \
--max-steps 1 \
--camera-topic head=/camera_f/color/image_raw/compressed \
--camera-topic left_wrist=/camera_l/color/image_raw/compressed \
--camera-topic right_wrist=/camera_r/color/image_raw/compressed \
--compressed-images \
--require-images"
```

WAM 的 checkpoint 配置、安全 hold、state history、轨迹录制和 replay 说明见
[`models/uva_dit/README.md`](models/uva_dit/README.md)。

## 切换到真机执行

先完成对应模型的单步 dry-run，并核对 server 返回的 action、机器人当前位姿、
相机、ROS topic、急停和工作区净空。确认无误后，终端 1 和终端 2 保持不变，
终端 3 将：

```text
--executor mock
--max-steps 1
```

替换为：

```text
--executor ros
--execute-actions
--enable-on-start
```

真机执行前必须继续遵循对应模型 README 中的起始位姿、动作限幅、inactive side、
退出 hold 和 checkpoint 契约，不能只依据这段公共参数直接运行。

## 公共资料

- [`docs/agilex_robot_ros_startup.md`](docs/agilex_robot_ros_startup.md)：机器人 ROS 初始化。
- [`docs/architecture.md`](docs/architecture.md)：三端通信和 action schema。
- [`reference/README.md`](reference/README.md)：底层 WebSocket、ROS/CAN 与旧接口参考。
