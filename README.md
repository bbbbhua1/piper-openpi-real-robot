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

两种模型使用相同的三端角色，但监听地址和 SSH 端口由各模型的实际命令决定：

```text
机器人端 client
  ws://<local-computer-ip>:<forwarded-port>
            |
            v
本地电脑 SSH tunnel
  <local-bind> -> server 127.0.0.1:7081
            |
            v
开发服务器 model server
```

pi0.5 当前使用 SSH `3763` 和 `10.13.10.63:18001`；UVA-DiT 当前使用 SSH
`7156` 和 `10.13.0.133:17081`。网络变化时需同时替换 tunnel bind 地址和机器人
命令中的 WebSocket URI。

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

下面是 Puzzle 0812-E checkpoint 的当前真机执行流程。终端 1 在开发服务器运行，
终端 2 在连接机器人网络的本地电脑运行，终端 3 在机器人端运行。

终端 3 会真实发布机器人 action。执行前必须确认 checkpoint、起始位姿、相机、
ROS topic、急停和工作区净空。机器人目录中需要已经部署 `scripts/` client 及
`configs/piper_uva_dit_puzzle_b_left_observation_pose.json`。

### WAM 终端 1：Policy Server

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

### WAM 终端 2：SSH Tunnel

在本地电脑执行并保持运行：

```bash
ssh -o ExitOnForwardFailure=yes \
  -p 7156 \
  -N \
  -L 10.13.0.133:17081:127.0.0.1:7081 \
  root@10.40.1.215
```

### WAM 终端 3：机器人真机 Client

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

WAM 的 checkpoint 配置、安全 hold、state history、轨迹录制和 replay 说明见
[`models/uva_dit/README.md`](models/uva_dit/README.md)。

## pi0.5 切换到真机执行

先完成 pi0.5 的单步 dry-run，并核对 server 返回的 action、机器人当前位姿、
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

真机执行前必须继续遵循 pi0.5 README 中的起始位姿、动作限幅、
退出 hold 和 checkpoint 契约，不能只依据这段公共参数直接运行。

## 公共资料

- [`docs/agilex_robot_ros_startup.md`](docs/agilex_robot_ros_startup.md)：机器人 ROS 初始化。
- [`docs/architecture.md`](docs/architecture.md)：三端通信和 action schema。
- [`reference/README.md`](reference/README.md)：底层 WebSocket、ROS/CAN 与旧接口参考。
