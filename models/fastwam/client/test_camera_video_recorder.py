from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest


CLIENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CLIENT_DIR))

from camera_video_recorder import CameraVideoRecorder  # noqa: E402

if "websockets" not in sys.modules:
    sys.modules["websockets"] = types.ModuleType("websockets")
import websocket_policy_client as policy_client  # noqa: E402


class FakeProcess:
    def __init__(self, stdin=None, stdout=None, stderr=None, **_kwargs):
        self.stdin = io.BytesIO()
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = None

    def wait(self, timeout=None):
        del timeout
        self.returncode = 0
        return self.returncode


class FakePopenFactory:
    def __init__(self, process_type=FakeProcess):
        self.commands = []
        self.processes = []
        self.process_type = process_type

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        process = self.process_type(**kwargs)
        self.processes.append(process)
        return process


class BlockingStdin:
    def __init__(self):
        self.buffer = io.BytesIO()
        self.write_started = threading.Event()
        self.allow_write = threading.Event()
        self.closed = False

    def write(self, data):
        self.write_started.set()
        self.allow_write.wait(timeout=2)
        return self.buffer.write(data)

    def flush(self):
        return None

    def close(self):
        self.closed = True


class BlockingProcess(FakeProcess):
    def __init__(self, stdin=None, stdout=None, stderr=None, **kwargs):
        super().__init__(stdin=stdin, stdout=stdout, stderr=stderr, **kwargs)
        self.stdin = BlockingStdin()


class FakeRunFactory:
    def __init__(self, fail_on_call=None):
        self.commands = []
        self.fail_on_call = fail_on_call

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        if self.fail_on_call == len(self.commands):
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)


class CameraVideoRecorderTest(unittest.TestCase):
    def make_recorder(self, root, **kwargs):
        popen_factory = kwargs.pop("popen_factory", FakePopenFactory())
        run_factory = kwargs.pop("run_factory", FakeRunFactory())
        recorder = CameraVideoRecorder(
            camera_names=("head", "left_wrist", "right_wrist"),
            root_dir=root,
            fps=30.0,
            queue_size=3,
            session_name="domino_test",
            popen_factory=popen_factory,
            run_factory=run_factory,
            **kwargs,
        )
        return recorder, popen_factory, run_factory

    def test_builds_one_ffmpeg_command_per_camera(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder, popen_factory, _ = self.make_recorder(temp_dir)
            recorder.close()

            self.assertEqual(len(popen_factory.commands), 3)
            for camera_name, command in zip(
                ("head", "left_wrist", "right_wrist"),
                popen_factory.commands,
            ):
                self.assertIn("-framerate", command)
                self.assertEqual(command[command.index("-framerate") + 1], "30")
                self.assertIn("-c:v", command)
                self.assertEqual(command[command.index("-c:v") + 1], "copy")
                self.assertTrue(command[-1].endswith("/{}.mp4".format(camera_name)))

    def test_submit_writes_frames_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder, _, _ = self.make_recorder(temp_dir)
            recorder.submit("head", b"\xff\xd8head\xff\xd9")
            recorder.submit("left_wrist", b"\xff\xd8left\xff\xd9")
            recorder.submit("right_wrist", b"\xff\xd8right\xff\xd9")
            recorder.close()
            manifest_path = recorder.write_manifest()

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["session_name"], "domino_test")
            self.assertEqual(manifest["fps"], 30.0)
            self.assertEqual(manifest["cameras"]["head"]["received"], 1)
            self.assertEqual(manifest["cameras"]["head"]["written"], 1)
            self.assertEqual(manifest["cameras"]["left_wrist"]["written"], 1)
            self.assertEqual(manifest["cameras"]["right_wrist"]["written"], 1)

    def test_full_queue_drops_oldest_frame_without_blocking_submitter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            popen_factory = FakePopenFactory(process_type=BlockingProcess)
            recorder = CameraVideoRecorder(
                camera_names=("head",),
                root_dir=temp_dir,
                fps=30.0,
                queue_size=1,
                session_name="domino_queue",
                popen_factory=popen_factory,
                run_factory=FakeRunFactory(),
            )
            process = popen_factory.processes[0]
            recorder.submit("head", b"first")
            self.assertTrue(process.stdin.write_started.wait(timeout=1))
            recorder.submit("head", b"second")
            recorder.submit("head", b"third")
            process.stdin.allow_write.set()
            recorder.close()

            stats = recorder.stats()["head"]
            self.assertEqual(stats["received"], 3)
            self.assertGreaterEqual(stats["dropped"], 1)
            self.assertEqual(stats["written"] + stats["dropped"], 3)

    def test_upload_creates_remote_root_then_copies_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder, _, run_factory = self.make_recorder(temp_dir)
            recorder.close()
            recorder.write_manifest()
            remote_path = recorder.upload()

            self.assertEqual(len(run_factory.commands), 2)
            self.assertEqual(run_factory.commands[0][0], "ssh")
            self.assertIn("mkdir -p", run_factory.commands[0][-1])
            self.assertEqual(run_factory.commands[1][0], "scp")
            self.assertIn("-r", run_factory.commands[1])
            self.assertEqual(
                remote_path,
                "/bh/media/section/iclr_proj/zbh/piper_openpi_real_robot_videos/domino_test",
            )

    def test_upload_failure_preserves_local_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_factory = FakeRunFactory(fail_on_call=2)
            recorder, _, _ = self.make_recorder(temp_dir, run_factory=run_factory)
            recorder.close()
            recorder.write_manifest()

            with self.assertRaises(subprocess.CalledProcessError):
                recorder.upload()
            self.assertTrue(recorder.session_dir.is_dir())
            self.assertTrue((recorder.session_dir / "manifest.json").is_file())


class PolicyClientRecordingIntegrationTest(unittest.TestCase):
    def test_recording_cli_defaults(self):
        parser = policy_client.build_arg_parser()
        args = parser.parse_args(
            ["--uri", "ws://127.0.0.1:18001", "--instruction", "test"]
        )

        self.assertFalse(args.record_videos)
        self.assertEqual(
            args.record_dir,
            "/home/agilex/piper-openpi-real-robot/recordings",
        )
        self.assertEqual(args.record_fps, 30.0)
        self.assertEqual(args.record_queue_size, 90)
        self.assertEqual(args.record_upload_host, "root@10.40.1.215")
        self.assertEqual(args.record_upload_port, 3763)
        self.assertEqual(
            args.record_upload_dir,
            "/bh/media/section/iclr_proj/zbh/piper_openpi_real_robot_videos",
        )
        self.assertIsNone(args.record_session_name)

    def test_compressed_callback_submits_exact_jpeg_payload(self):
        class FakeRecorder:
            def __init__(self):
                self.calls = []

            def submit(self, camera_name, data):
                self.calls.append((camera_name, data))

        class Message:
            data = bytearray(b"\xff\xd8camera-jpeg\xff\xd9")

        source = policy_client.RosObservationSource.__new__(
            policy_client.RosObservationSource
        )
        source.lock = threading.Lock()
        source.images = {}
        source.video_recorder = FakeRecorder()

        source._compressed_image_callback(Message(), "head")

        self.assertEqual(
            source.video_recorder.calls,
            [("head", b"\xff\xd8camera-jpeg\xff\xd9")],
        )
        self.assertEqual(
            source.images["head"],
            {"encoding": "jpeg", "data": b"\xff\xd8camera-jpeg\xff\xd9"},
        )


if __name__ == "__main__":
    unittest.main()
