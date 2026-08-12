# FastWAM Puzzle/crimp Piper 真机部署

这是 FastWAM 在松灵 Piper 双臂真机上的独立工作流。它与
`models/openpi_pi05/`、`models/uva_dit/` 并列；不要混用不同模型目录中的 server、
client 或配置。

FastWAM 的训练任务为 `crimp`，动作空间为关节。模型每轮预测 32 个 waypoint，服务端
只返回前 24 个（30 Hz 下约 0.8 秒）；client 执行后重新获取三路图像和机器人状态。

## 目录

```text
models/fastwam/
├── README.md
├── server/                 推理、两种服务端模式、配置、启动器和测试
├── client/                 Piper ROS client、起始位、录像、协议和测试
└── docs/                   实现历史；不作为部署命令来源
```

## 环境、权重与统计

```text
开发机：ssh -p 7156 root@10.40.1.215
FastWAM source：/bh/zbh_self/projects/fastwam
FastWAM env：/bh/zbh_self/envs/fastwam
server：/bh/zbh_self/projects/piper-openpi-real-robot/models/fastwam/server
client：/home/agilex/piper-openpi-real-robot/models/fastwam/client
base models：/bh/zbh_ckp/models/fastwam
dataset stats：/bh/zbh_ckp/datasets/fastwam/agilex_cobotmagic2_puzzle_20260801_v21_puppet_joint/normalization/dataset_stats.json
checkpoint：/bh/zbh_ckp/runs/fastwam/agilex_cobotmagic2_puzzle_joint_3cam_384_1e-4_baidu32_statecontinue30000_wandb/CogWAM_fastWAM-puzzle-statecontinue32-30000-wandb-retry2/checkpoints/weights/step_005000.pt
```

权重、基座、统计和录像不进入 Git。

## 机器人 ROS 初始化

先完成公共 ROS 初始化：
[`docs/agilex_robot_ros_startup.md`](../../docs/agilex_robot_ros_startup.md)。已验证控制模式：

```bash
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=false
```

client 读取 `/puppet/joint_left`、`/puppet/joint_right`，执行时发布
`/master/joint_left`、`/master/joint_right`。三路压缩相机 topic 必须可用：

```text
/camera_f/color/image_raw/compressed
/camera_l/color/image_raw/compressed
/camera_r/color/image_raw/compressed
```

## 部署

从仓库根目录部署开发机服务端：

```bash
ssh -F /dev/null -p 7156 root@10.40.1.215 \
  'mkdir -p /bh/zbh_self/projects/piper-openpi-real-robot/models/fastwam/server'
scp models/fastwam/server/* \
  root@10.40.1.215:/bh/zbh_self/projects/piper-openpi-real-robot/models/fastwam/server/
```

机器人可访问时部署 client：

```bash
ssh agilex@10.13.0.155 \
  'mkdir -p /home/agilex/piper-openpi-real-robot/models/fastwam/client /home/agilex/piper-openpi-real-robot/recordings'
scp models/fastwam/client/* \
  agilex@10.13.0.155:/home/agilex/piper-openpi-real-robot/models/fastwam/client/
```

## 三终端 dry-run

先完成 ROS 初始化、摆好 Puzzle/crimp 场景，并确认急停可达。一次只能启动一个模型服务端，
因为它们都使用开发机 `127.0.0.1:7081`。

### 终端 1：标准保护 server

```bash
ssh -F /dev/null -tt -p 7156 root@10.40.1.215 \
  'cd /bh/zbh_self/projects/piper-openpi-real-robot/models/fastwam/server && \
   CUDA_VISIBLE_DEVICES=0 exec bash run_piper_fastwam_server.sh'
```

出现 `FastWAM Piper policy server listening on ws://127.0.0.1:7081` 后继续。

### 终端 2：SSH tunnel

```bash
ssh -F /dev/null -N -g -p 7156 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L 0.0.0.0:18001:127.0.0.1:7081 \
  root@10.40.1.215
```

### 终端 3：机器人单步 dry-run

将 `10.13.0.96` 替换为本地电脑在机器人网络中的实际 IP。没有 `--execute-actions`，
所以不会向机械臂发布动作。

```bash
ssh -tt agilex@10.13.0.155 '
source /opt/ros/noetic/setup.bash
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

/home/agilex/miniconda3/envs/xrocs-env/bin/python \
  /home/agilex/piper-openpi-real-robot/models/fastwam/client/websocket_policy_client.py \
  --uri ws://10.13.0.96:18001 \
  --instruction crimp \
  --source ros \
  --executor mock \
  --max-steps 1 \
  --control-hz 30 \
  --compressed-images \
  --transport-image-profile fastwam \
  --require-images \
  --camera-topic head=/camera_f/color/image_raw/compressed \
  --camera-topic left_wrist=/camera_l/color/image_raw/compressed \
  --camera-topic right_wrist=/camera_r/color/image_raw/compressed
'
```

预期输出：`connected to policy server`、动作 JSON 与 `[dry-run] action execution disabled`。
先核对图像、状态、`crimp` 指令、动作和延迟，再进行真机执行。

## 完整真机执行

终端 1、2 保持不变。终端 3 使用以下命令：先平滑移动并验证训练 episode 159 的起始位，
然后最多执行 25 次滚动推理，同时录制三路视频。

```bash
ssh -tt agilex@10.13.0.155 '
source /opt/ros/noetic/setup.bash
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

/home/agilex/miniconda3/envs/xrocs-env/bin/python \
  /home/agilex/piper-openpi-real-robot/models/fastwam/client/websocket_policy_client.py \
  --uri ws://10.13.0.96:18001 \
  --instruction crimp \
  --source ros \
  --executor ros \
  --enable-on-start true \
  --execute-actions \
  --start-pose-config /home/agilex/piper-openpi-real-robot/models/fastwam/client/piper_fastwam_puzzle_start_pose.json \
  --max-steps 25 \
  --control-hz 30 \
  --compressed-images \
  --transport-image-profile fastwam \
  --require-images \
  --camera-topic head=/camera_f/color/image_raw/compressed \
  --camera-topic left_wrist=/camera_l/color/image_raw/compressed \
  --camera-topic right_wrist=/camera_r/color/image_raw/compressed \
  --record-videos \
  --record-dir /home/agilex/piper-openpi-real-robot/recordings \
  --record-session-name fastwam-crimp-step005000-$(date +%Y%m%d_%H%M%S) \
  --record-upload-host root@10.40.1.215 \
  --record-upload-port 7156 \
  --record-upload-dir /bh/zbh_ckp/runs/fastwam/piper_crimp_real_robot
'
```

FastWAM transport profile 会在发送前 resize：head 为 `320×256`，两个 wrist 为 `160×128`，
并保持模型所需的 `[3,384,320]` 拼图几何。

## 两种服务端模式

标准模式是默认模式：有限模型输出按验证过的 Piper 绝对范围投影，并由
`max_delta` 做逐 waypoint 限速。

人工监护原样模式仅用于诊断软限制影响：仍拒绝 NaN/Inf 或格式错误输出，但有限的 24 个
waypoint 不经过服务端关节夹取、投影阈值拒绝或逐点限速。机器人驱动/固件硬保护、ROS
fault、Ctrl-C hold、最大步数和录像仍保留。操作者必须持续现场监护。

使用它时仅替换终端 1：

```bash
ssh -F /dev/null -tt -p 7156 root@10.40.1.215 \
  'cd /bh/zbh_self/projects/piper-openpi-real-robot/models/fastwam/server && \
   CUDA_VISIBLE_DEVICES=0 exec bash run_piper_fastwam_supervised_raw_server.sh'
```

启动日志会出现 `SUPERVISED RAW ACTION MODE enabled`，每次返回动作会出现
`no software clamp or delta limit applied`。

## 录像、停止与 checkpoint 切换

录像先保存在：

```text
/home/agilex/piper-openpi-real-robot/recordings/<session>/
```

内含三路 MP4 和 `manifest.json`。运行结束后才尝试上传到：

```text
/bh/zbh_ckp/runs/fastwam/piper_crimp_real_robot
```

没有可用开发机 SSH key 时上传失败不会删除机器人本地录像。第一次 `Ctrl-C` 停止推理并
进入最后 waypoint hold，第二次 `Ctrl-C` 才退出 hold；出现异常物理轨迹优先使用现场急停。

切换 checkpoint 时，同时修改：

```text
models/fastwam/server/piper_fastwam_puzzle.yaml
models/fastwam/server/piper_fastwam_puzzle_supervised_raw.yaml
```

中的 `checkpoint_path`，部署 server 文件并重启终端 1。client、隧道、起始位配置无需更改。
