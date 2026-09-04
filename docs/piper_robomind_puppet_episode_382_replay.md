# RoboMIND Puppet episode 382 replay on Piper

This replay is model-free. It reads the fixed RoboMIND Parquet trajectory on the development machine, holds the left arm at the fixed observation pose, and publishes the 453 Puppet right-arm waypoints once.

## Before starting

The old policy client must be stopped first. In the terminal where it is running, press `Ctrl-C` and wait until its process disappears. Do not run the replay while the old client is alive.

On the robot, keep exactly one `roscore`, one `start_ms_piper.launch`, and one `multi_camera.launch`. Terminal 2 below only verifies; it does not launch a second copy.

## Terminal 1 — development machine replay server and tunnel

Run this on the local development machine:

```bash
ssh -p 7156 \
  -L 10.13.0.133:17082:127.0.0.1:7082 \
  root@10.40.1.215 '
cd /bh/zbh_self/projects/piper-openpi-real-robot
exec /bh/zbh_self/envs/fastwam/bin/python \
  scripts/piper_robomind_puppet_replay_server.py \
  --host 127.0.0.1 \
  --port 7082
'
```

The server must print `Puppet preflight passed` before it listens. It does not load the UVA-DiT checkpoint.

## Terminal 2 — robot status verification

Run this on the local development machine:

```bash
ssh -p 22 agilex@10.13.0.141
```

Then on the robot, after sourcing the Piper ROS workspace:

```bash
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
rosnode list
rostopic list | grep -E 'puppet/joint|master/joint|camera_.*/color/image_raw/compressed|enable_flag'
rostopic hz /puppet/joint_left
rostopic hz /puppet/joint_right
```

Expected: one Piper node per arm, camera topics available, and the Puppet joint topics publishing. Do not run another ROS launch command here.

## Terminal 3 — one-shot physical replay

Run this on the same robot shell after Terminal 1 is listening:

```bash
cd /home/agilex/piper-openpi-real-robot
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
python3 scripts/websocket_policy_client.py \
  --uri ws://10.13.0.133:17082 \
  --instruction 'RoboMIND Puppet episode 382 replay' \
  --source ros \
  --executor ros \
  --execute-actions \
  --max-steps 1 \
  --start-pose-config configs/piper_robomind_puppet_episode_382_start_pose.json
```

The client first moves to the episode 382 first frame at the configured low speed. It then requests one 453-waypoint chunk. The left-arm part of every waypoint is constant; only the right-arm part follows Puppet. After the chunk, the client exits its request loop and holds the measured position.

## Stopping

Press `Ctrl-C` in Terminal 3 to leave the hold loop. Press `Ctrl-C` in Terminal 1 to stop the replay server and tunnel. Restart the replay server before another replay because it is one-shot.
