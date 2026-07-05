# Reference Scripts

这个目录保留 `/Users/hua/Downloads/websocket真机` 中的基础流程脚本。它们不是当前 OpenPI 三端运行链路的必需文件，而是以后换机器、换模型、重新梳理底层 ROS/CAN 流程时的参考材料。

当前主流程仍然是：

1. 服务器端运行 `scripts/piper_openpi_policy_server.py`。
2. 本地电脑开启 SSH tunnel。
3. 机器人端运行 `scripts/websocket_policy_client.py`。

## `basic_websocket_flow/`

基础 WebSocket policy client/server 示例：

```text
basic_websocket_flow/
├── example_remote_policy.py
├── websocket_policy_client.py
├── websocket_policy_server.py
└── ws_policy_protocol.py
```

这几个文件基本是一套：

- `websocket_policy_client.py` 使用 `ws_policy_protocol.build_policy_request()` 发送请求。
- `websocket_policy_server.py` 使用 `decode_policy_request()` 解码请求。
- `websocket_policy_server.py` 默认加载 `example_remote_policy:predict_action`。
- `example_remote_policy.py` 返回 hold-position 风格的 joint action chunk。

注意：这里的 client 是 Astribot 版本，依赖 `core.astribot_api.astribot_client.Astribot`，不是当前 Piper ROS client。

## `piper_ros_nodes/`

Piper ROS/CAN 底层桥接参考节点：

```text
piper_ros_nodes/
├── piper_read_master_node.py
├── piper_start_ms_node.py
└── piper_start_slave_node.py
```

这些脚本主要负责把 Piper SDK/CAN 和 ROS topic 接起来。它们会发布或订阅类似：

```text
/puppet/joint_states
/puppet/arm_status
/puppet/end_pose
/puppet/end_pose_euler
/master/joint_states
/pos_cmd
/enable_flag
```

它们不是当前 OpenPI WebSocket 流程直接 import 或调用的文件。当前新 client 默认使用左右臂拆开的 topic：

```text
/puppet/joint_left
/puppet/joint_right
/master/joint_left
/master/joint_right
```

如果未来要复用这些 Piper ROS 节点，需要通过 ROS remap 或参数化方式对齐 topic。

## `legacy_http_policy_client/`

旧 HTTP policy client：

```text
legacy_http_policy_client/
└── piper_policy_client_node.py
```

它通过 HTTP `POST` 请求 `http://127.0.0.1:8000/infer`，不是 WebSocket 流程。当前目录没有配套的 HTTP server，因此只作为旧接口参考保留。

## 当前判断

- `basic_websocket_flow/` 是一套基础 WebSocket 示例。
- `piper_ros_nodes/` 是 Piper ROS/CAN 桥接参考。
- `legacy_http_policy_client/` 是另一条旧 HTTP 路线。
- 三者相关，但不是一个完全统一、开箱即跑的单一项目。
