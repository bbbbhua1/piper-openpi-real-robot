# Robot Initialization TODO

ROS、CAN、Piper 双臂、三路相机、Tracer 和实时投影节点的已验证启动流程见
[`agilex_robot_ros_startup.md`](agilex_robot_ros_startup.md)。本文件只保留尚未在本仓库中形成可复现步骤的事项。

## Power And Safety

- 上电顺序。
- 急停状态检查。
- 机械臂使能前检查。
- 夹爪状态检查。
- 安全边界和人员站位。

## Hardware Safety And Calibration

- 机械臂和夹爪的 home、标定与复位流程。
- `/enable_flag` 的精确定义、失能条件和异常恢复流程。
- 与 `auto_enable:=false` 配合的人工使能检查。
- 不同机器人硬件版本的 CAN 转接器端口映射确认。

## Cameras

- 图像方向、颜色空间和延迟检查。
- `/dev/video* Permission denied` 等权限异常的标准修复步骤。

## Dry Run

- 不加 `--execute-actions` 跑通信。
- 确认 server 返回 action。
- 确认 action 数值和当前 joint 不发生明显跳变。

## Real Run

- 加 `--execute-actions` 前确认 home 流程。
- 确认 `--max-joint-step` 设置。
- 确认 `--waypoint-sleep` 设置。
- 运行时保留急停可触达。
