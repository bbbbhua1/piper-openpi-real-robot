# Robot Initialization TODO

当前仓库还没有完整机器初始化流程。后续需要补齐以下内容。

## Power And Safety

- 上电顺序。
- 急停状态检查。
- 机械臂使能前检查。
- 夹爪状态检查。
- 安全边界和人员站位。

## ROS And CAN

- ROS master 启动方式。
- 左右臂 CAN 口对应关系。
- Piper bridge 节点启动方式。
- `/enable_flag` 的预期行为。
- 左右臂 topic remap 规则。

## Cameras

- 三路相机启动方式。
- 当前使用 topic：
  - `/camera_f/color/image_raw/compressed`
  - `/camera_l/color/image_raw/compressed`
  - `/camera_r/color/image_raw/compressed`
- 图像方向、颜色空间和延迟检查。

## Dry Run

- 不加 `--execute-actions` 跑通信。
- 确认 server 返回 action。
- 确认 action 数值和当前 joint 不发生明显跳变。

## Real Run

- 加 `--execute-actions` 前确认 home 流程。
- 确认 `--max-joint-step` 设置。
- 确认 `--waypoint-sleep` 设置。
- 运行时保留急停可触达。
