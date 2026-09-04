#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Client node for policy inference over a client-server boundary.

This node keeps the model runtime out of the robot process:
  1. collect robot observations from ROS topics published by the Piper driver,
  2. POST the observation to a server,
  3. execute the returned action chunk by publishing joint commands.
"""

import json
import socket
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


JointAction = Dict[str, List[float]]


class PiperPolicyClientNode:
    def __init__(self) -> None:
        rospy.init_node("piper_policy_client_node", anonymous=True)

        self.server_url = rospy.get_param("~server_url", "http://127.0.0.1:8000/infer")
        self.request_hz = float(rospy.get_param("~request_hz", 5.0))
        self.execute_hz = float(rospy.get_param("~execute_hz", 20.0))
        self.timeout_s = float(rospy.get_param("~timeout_s", 2.0))
        self.enable_on_start = self._get_bool_param("~enable_on_start", False)
        self.require_both_arms = self._get_bool_param("~require_both_arms", True)

        self.left_observation_topic = rospy.get_param("~left_observation_topic", "/puppet/joint_left")
        self.right_observation_topic = rospy.get_param("~right_observation_topic", "/puppet/joint_right")
        self.left_action_topic = rospy.get_param("~left_action_topic", "/master/joint_left")
        self.right_action_topic = rospy.get_param("~right_action_topic", "/master/joint_right")
        self.enable_topic = rospy.get_param("~enable_topic", "/enable_flag")

        self._lock = threading.Lock()
        self._sequence = 0
        self._latest_left_joint: Optional[JointState] = None
        self._latest_right_joint: Optional[JointState] = None

        rospy.Subscriber(
            self.left_observation_topic,
            JointState,
            self._left_joint_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.right_observation_topic,
            JointState,
            self._right_joint_callback,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.left_action_pub = rospy.Publisher(
            self.left_action_topic,
            JointState,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.right_action_pub = rospy.Publisher(
            self.right_action_topic,
            JointState,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.enable_pub = rospy.Publisher(self.enable_topic, Bool, queue_size=1, latch=True)

        if self.enable_on_start:
            rospy.sleep(0.5)
            self.enable_pub.publish(Bool(data=True))

    def _get_bool_param(self, name: str, default: bool) -> bool:
        value = rospy.get_param(name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _left_joint_callback(self, msg: JointState) -> None:
        with self._lock:
            self._latest_left_joint = msg

    def _right_joint_callback(self, msg: JointState) -> None:
        with self._lock:
            self._latest_right_joint = msg

    def _joint_state_to_dict(self, msg: JointState) -> Dict[str, Any]:
        return {
            "name": list(msg.name),
            "position": list(msg.position),
            "velocity": list(msg.velocity),
            "effort": list(msg.effort),
            "stamp": msg.header.stamp.to_sec(),
        }

    def _build_observation(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            left = self._latest_left_joint
            right = self._latest_right_joint

        if left is None or (self.require_both_arms and right is None):
            return None

        self._sequence += 1
        observation = {
            "timestamp": time.time(),
            "sequence": self._sequence,
            "observations": {
                "left": self._joint_state_to_dict(left),
            },
        }
        if right is not None:
            observation["observations"]["right"] = self._joint_state_to_dict(right)
        return observation

    def _request_action_chunk(self, observation: Dict[str, Any]) -> List[JointAction]:
        body = json.dumps(observation).encode("utf-8")
        request = Request(
            self.server_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            response_body = response.read().decode("utf-8")
        payload = json.loads(response_body)
        return self._parse_action_chunk(payload)

    def _parse_action_chunk(self, payload: Any) -> List[JointAction]:
        action_payload = payload
        if isinstance(payload, dict):
            action_payload = (
                payload.get("action_chunk")
                or payload.get("actions")
                or payload.get("action")
                or payload
            )

        if isinstance(action_payload, dict):
            return self._parse_dict_action_chunk(action_payload)
        if isinstance(action_payload, list):
            return [self._parse_action_step(step) for step in action_payload]

        raise ValueError("server response must contain an action chunk")

    def _parse_dict_action_chunk(self, action_payload: Dict[str, Any]) -> List[JointAction]:
        left_chunk = action_payload.get("left")
        right_chunk = action_payload.get("right")

        if self._is_joint_vector(left_chunk) or self._is_joint_vector(right_chunk):
            return [self._parse_action_step(action_payload)]

        chunk_len = max(len(left_chunk or []), len(right_chunk or []))
        actions = []
        for index in range(chunk_len):
            step = {}
            if left_chunk and index < len(left_chunk):
                step["left"] = left_chunk[index]
            if right_chunk and index < len(right_chunk):
                step["right"] = right_chunk[index]
            actions.append(self._parse_action_step(step))
        return actions

    def _parse_action_step(self, step: Any) -> JointAction:
        if isinstance(step, dict):
            action: JointAction = {}
            if "left" in step:
                action["left"] = self._extract_joint_position(step["left"])
            if "right" in step:
                action["right"] = self._extract_joint_position(step["right"])
            if not action and "joint_position" in step:
                action["left"] = self._extract_joint_position(step)
            if action:
                return action

        if self._is_joint_vector(step):
            values = [float(value) for value in step]
            if len(values) == 14:
                return {"left": values[:7], "right": values[7:]}
            if len(values) == 7:
                return {"left": values}

        raise ValueError("invalid action step format: {}".format(step))

    def _extract_joint_position(self, value: Any) -> List[float]:
        if isinstance(value, dict):
            value = value.get("joint_position") or value.get("position") or value.get("joints")
        if not self._is_joint_vector(value):
            raise ValueError("joint action must be a list of 7 numeric values")

        joints = [float(item) for item in value]
        if len(joints) != 7:
            raise ValueError("joint action must contain exactly 7 values")
        return joints

    def _is_joint_vector(self, value: Any) -> bool:
        return isinstance(value, list) and all(isinstance(item, (int, float)) for item in value)

    def _publish_joint_action(self, arm: str, joints: List[float]) -> None:
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        msg.position = joints
        msg.velocity = [0.0] * 7
        msg.effort = [0.0] * 7

        if arm == "left":
            self.left_action_pub.publish(msg)
        elif arm == "right":
            self.right_action_pub.publish(msg)
        else:
            raise ValueError("unknown arm: {}".format(arm))

    def _execute_action_chunk(self, actions: List[JointAction]) -> None:
        rate = rospy.Rate(self.execute_hz)
        for action in actions:
            if rospy.is_shutdown():
                return
            if "left" in action:
                self._publish_joint_action("left", action["left"])
            if "right" in action:
                self._publish_joint_action("right", action["right"])
            rate.sleep()

    def spin(self) -> None:
        rate = rospy.Rate(self.request_hz)
        rospy.loginfo("Piper policy client started, server_url=%s", self.server_url)

        while not rospy.is_shutdown():
            observation = self._build_observation()
            if observation is None:
                rospy.logwarn_throttle(2.0, "waiting for robot observations")
                rate.sleep()
                continue

            try:
                actions = self._request_action_chunk(observation)
                if actions:
                    self._execute_action_chunk(actions)
                else:
                    rospy.logwarn("server returned an empty action chunk")
            except (URLError, TimeoutError, socket.timeout) as exc:
                rospy.logerr_throttle(1.0, "policy server request failed: %s", exc)
            except (ValueError, json.JSONDecodeError) as exc:
                rospy.logerr_throttle(1.0, "invalid policy server response: %s", exc)

            rate.sleep()


if __name__ == "__main__":
    try:
        PiperPolicyClientNode().spin()
    except rospy.ROSInterruptException:
        pass
