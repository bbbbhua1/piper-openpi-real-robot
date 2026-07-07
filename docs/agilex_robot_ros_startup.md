# Agilex Robot ROS Startup And Data Collection Notes

这份文档记录松灵 Piper / Agilex 真机的底层 ROS 启动、相机启动、rosbag 录制、数据上传和数据处理命令。内容来自现场流程笔记，后续可以继续拆分为更细的启动手册、采集手册和故障排查文档。

# 1 启动硬件
## 1.1 新建一个终端，启动
roscore

# 1.2 进入agx_arm目录，使能can，并启动机械臂数据读取

cd cobot_magic/Piper_ros_private-ros-noetic/
bash can_config.sh
source devel/setup.bash
roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=false

## 1.3 启动相机
realsense相机：


cd cobot_magic/camera_ws
source devel/setup.bash
roslaunch realsense2_camera multi_camera.launch


# 1.4 终端运行rostopic list查看ros话题，确认 topic 是否存在
rostopic list  

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
