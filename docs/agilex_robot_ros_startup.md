# Agilex Robot ROS Startup And Data Collection Notes

这份文档记录松灵 Piper / Agilex 真机的底层 ROS 初始化、相机启动、rosbag 录制、数据上传和数据处理命令。第 1 节是已在 2026-07-28 真机上验证的完整 ROS 初始化流程；后续章节保留历史的数据采集与处理笔记。

## 1. 真机 ROS 初始化

### 1.1 适用范围与准备

- 机器人用户为 `agilex`。本次验证时机器人有两个 IP；同一局域网内可直接使用的地址为 `10.13.11.215`。网络变化后先在机器人上运行 `hostname -I` 确认地址，不要把密码写入脚本或仓库。
- 先完成上电、急停释放、机械臂周围净空等人工安全检查。当前仓库尚未验证完整的上电、夹爪标定与 home 硬件流程。
- 使用 5 个独立终端标签页执行以下命令，并保持每个 `roscore` / `roslaunch` 前台运行。这样需要停止时可以在对应终端按 `Ctrl-C`，不要用宽泛的 `pkill`。
- 启动 policy client 前，必须完成本节的节点与话题检查。

### 1.2 终端 1：启动 ROS master

```bash
source /opt/ros/noetic/setup.bash
roscore
```

看到 `started core service [/rosout]` 后再启动其余终端。若 `roscore` 报已有 master，不要再启动第二个；先运行 `rosnode list` 确认当前状态，或在原来的 `roscore` 终端按 `Ctrl-C` 后重新开始。

### 1.3 终端 2：配置 CAN 并启动 Piper 双臂

```bash
cd /home/agilex/cobot_magic/Piper_ros_private-ros-noetic
sudo bash can_config1.sh
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=false
```

`can_config1.sh` 应识别并启用 `can_left`、`can_right`、`can_ugv` 和 `can_null`。本流程使用 `mode:=1`，用于接收和下发机械臂控制指令；不要在需要执行 OpenPI action 时替换成旧笔记里的 `mode:=0`。保留 `auto_enable:=false`，由上层 client 的 `--enable-on-start` 控制使能时机。

Piper 节点的左右臂 remap 为：

| 用途 | 左臂 | 右臂 |
| --- | --- | --- |
| 当前 joint observation | `/puppet/joint_left` | `/puppet/joint_right` |
| OpenPI joint action | `/master/joint_left` | `/master/joint_right` |
| Piper 状态 | `/puppet/arm_status_left` | `/puppet/arm_status_right` |
| EEF/位置控制接口（当前 OpenPI 链路未使用） | `/puppet/pos_cmd_left` | `/puppet/pos_cmd_right` |

### 1.4 终端 3：启动三路 RealSense 相机

```bash
cd /home/agilex/cobot_magic/camera_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch realsense2_camera multi_camera.launch initial_reset:=true
```

OpenPI 流程使用三路压缩彩色图像：

```text
/camera_f/color/image_raw/compressed
/camera_l/color/image_raw/compressed
/camera_r/color/image_raw/compressed
```

### 1.5 终端 4：启动 Tracer 底盘

```bash
cd /home/agilex/agilex_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch tracer_bringup tracer_robot_base.launch
```

### 1.6 终端 5：启动实时投影节点

```bash
cd /home/agilex/embodied_data/device/app/calibration
source /opt/ros/noetic/setup.bash
python3 realtime_projection_node.py
```

该节点是已验证的整机启动栈的一部分，并订阅左右臂当前 joint state。若只做底层 CAN 或单臂排障，可以不启动它；运行完整真机链路时按本节启动。

### 1.7 启动后验证

先查看节点。Piper 和投影节点名称带有启动时生成的后缀，因此不要比较完整字符串，只确认同类节点存在：

```bash
rosnode list
```

预期至少包括：

```text
/rosout
/camera_f/realsense2_camera
/camera_f/realsense2_camera_manager
/camera_l/realsense2_camera
/camera_l/realsense2_camera_manager
/camera_r/realsense2_camera
/camera_r/realsense2_camera_manager
/tracer_base_node
/realtime_projection_node_<...>
/piper_left_agilex_<...>
/piper_right_agilex_<...>
```

逐个确认节点可响应：

```bash
for node in $(rosnode list); do
  rosnode ping -c 1 "$node"
done
```

确认 CAN 接口处于 `UP`：

```bash
ip -br link | grep -E 'can_(left|right|ugv|null)'
```

确认 Piper observation/action topic 的发布订阅关系。启动 policy client 前，action topic 没有 publisher 是正常的；Piper 必须是 subscriber：

```bash
rostopic info /puppet/joint_left
rostopic info /puppet/joint_right
rostopic info /master/joint_left
rostopic info /master/joint_right
```

确认三路相机实际在出图，而不只是在 ROS master 中注册了节点：

```bash
rostopic hz /camera_f/color/image_raw/compressed
rostopic hz /camera_l/color/image_raw/compressed
rostopic hz /camera_r/color/image_raw/compressed
```

每条命令应持续显示频率。2026-07-28 曾出现相机节点能 `ping`、但日志包含 `/dev/video* Permission denied` 且图像 topic 无数据的情况；遇到这种情况不要启动带 `--require-images` 的 policy client。先检查设备权限、相机连接和 `multi_camera.launch` 的设备配置。

### 1.8 启动 OpenPI client 前的安全顺序

1. 按第 1.7 节确认 joint state、CAN 和三路图像都正常。
2. 先去掉 `--execute-actions` 运行 WebSocket client，确认能连到 policy server、能收到 action，且 action 数值没有明显跳变。
3. 确认机械臂周围净空、急停可触达、`--max-joint-step` 和 `--waypoint-sleep` 符合当前实验后，再加入 `--execute-actions --enable-on-start`。

OpenPI client 的完整命令见仓库根目录 [`README.md`](../README.md)。

### 1.9 停止、重启与日志

正常停止顺序是：先在投影、Piper、相机和 Tracer 的终端分别按 `Ctrl-C`，最后停止 `roscore`。如果 `roscore` 提示已有 master，先查清原进程和原终端，不要直接再开一个 master。

ROS 日志目录可能累积到 1 GB 以上。先检查：

```bash
rosclean check
```

确认不需要保留旧日志后，才执行 `rosclean purge`；该命令会删除 ROS 日志。

## 2. 历史数据采集与处理笔记

conda activate xrocs-env


# 启动客户端

conda activate xrocs-env
python /home/agilex/Dev/collect_agent/app.py --device=agilex_cobotmagic2_dualArm-gripper-3cameras_6 --env=prod

#录制rosbag数据
rosbag record -O "$FULL_PATH" --lz4 \
/camera_f/color/image_raw/compressed \
/camera_f/color/metadata \
/camera_f/depth/image_rect_raw/compressed \
/camera_f/depth/metadata \
/camera_l/color/image_raw/compressed \
/camera_l/color/metadata \
/camera_l/depth/image_rect_raw/compressed \
/camera_l/depth/metadata \
/camera_r/color/image_raw/compressed \
/camera_r/color/metadata \
/camera_r/depth/image_rect_raw/compressed \
/camera_r/depth/metadata \
/master/joint_left \
/master/joint_right \
/puppet/joint_left \
/puppet/joint_right \
/puppet/end_pose_left \
/puppet/end_pose_right


数据上传到数采平台：
conda activate xrocs-env
/home/agilex/embodied_data/device/app/collect_service/start.sh scape -d '/home/agilex/4.18.am' --preload --file_format=rosbag --upload='/home/agilex/Downloads'

#################################################################################################################
# 数据处理

conda activate xrocs-env
python /home/agilex/Dev/data_scape/data_scape/cmd.py -d '/home/agilex/data' --preload --file_format hdf5

python /home/agilex/Dev/data_scape/data_scape/cmd.py -d '/home/agilex/data/2025-09-16' --preload --file_format hdf5 --upload=/home/agilex/Downloads


# 相机参数

code /home/agilex/cobot_magic/camera_ws/src/realsense-ros/realsense2_camera/launch/multi_camera.launch 




=========================================

启动相机之前执行

mkdir -p /home/agilex/Documents/camera_calibration

python xrocs-plus/apps/camera_calibration/tools/load_realsense_intrinsics.py 


cp /media/agilex/data_department9/agilex_station.py /home/agilex/Dev/xrocs1.9/xRocs/xrocs/entity/station/Agilex_v2/agilex_station.py



[2025-09-25 18:41:19] HIGHLIGHT:    /home/agilex/embodied_data/device/app/collect_service/start.sh collect --device='franka_emika_sim-singleArm-gripper-3cameras_1' --env='prod'
[2025-09-25 18:41:19] HIGHLIGHT: 
[2025-09-25 18:41:19] HIGHLIGHT:    
启动数据上传：
conda activate xrocs-env
/home/agilex/embodied_data/device/app/collect_service/start.sh scape -d '/home/agilex/2025-10-09' --preload --file_format=rosbag --upload='/home/agilex/Downloads'


################################################################################
cp /media/agilex/data_department9/DataCollector.py /home/agilex/Dev/xrocs1.9/xRocs/xrocs/apps/data_collection/DataCollector.py



source /home/agilex/embodied_data/.venv/bin/activate && /home/agilex/embodied_data/device/app/data_service/start.sh local_process -d '/media/agilex/agilex_4/2025-11-18' --upload='/home/agilex/embodied_data/tools' --file_format=ros1mcap -s



./embodied_data/device/app/data_collection_server/scripts/start_server.sh
