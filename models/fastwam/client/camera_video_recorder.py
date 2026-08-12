#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-blocking MP4 recording for compressed ROS camera messages."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import queue
import shlex
import subprocess
import threading
import time
from typing import Any, Callable, Iterable, Optional


DEFAULT_RECORD_DIR = "/home/agilex/piper-openpi-real-robot/recordings"
DEFAULT_UPLOAD_HOST = "root@10.40.1.215"
DEFAULT_UPLOAD_PORT = 3763
DEFAULT_UPLOAD_DIR = (
    "/bh/media/section/iclr_proj/zbh/piper_openpi_real_robot_videos"
)

_STOP = object()


def _safe_component(value: str, max_len: int = 120) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    )
    return (safe or "recording")[:max_len]


def default_session_name() -> str:
    return time.strftime("domino_%Y%m%d_%H%M%S")


@dataclass
class _CameraState:
    name: str
    output_path: Path
    frame_queue: queue.Queue
    process: Any
    log_handle: Any
    thread: Optional[threading.Thread] = None
    received: int = 0
    written: int = 0
    dropped: int = 0
    returncode: Optional[int] = None
    error: Optional[str] = None


class CameraVideoRecorder:
    """Record compressed JPEG frames without blocking ROS callbacks."""

    def __init__(
        self,
        *,
        camera_names: Iterable[str],
        root_dir: str | Path = DEFAULT_RECORD_DIR,
        fps: float = 30.0,
        queue_size: int = 90,
        session_name: Optional[str] = None,
        ffmpeg_bin: str = "ffmpeg",
        upload_host: str = DEFAULT_UPLOAD_HOST,
        upload_port: int = DEFAULT_UPLOAD_PORT,
        upload_dir: str = DEFAULT_UPLOAD_DIR,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        run_factory: Callable[..., Any] = subprocess.run,
    ) -> None:
        names = tuple(_safe_component(name) for name in camera_names)
        if not names:
            raise ValueError("at least one camera name is required")
        if len(set(names)) != len(names):
            raise ValueError("camera names must be unique")
        if fps <= 0:
            raise ValueError("recording FPS must be positive")
        if queue_size <= 0:
            raise ValueError("recording queue size must be positive")
        if upload_port <= 0:
            raise ValueError("upload port must be positive")

        self.fps = float(fps)
        self.queue_size = int(queue_size)
        self.upload_host = upload_host
        self.upload_port = int(upload_port)
        self.upload_dir = upload_dir.rstrip("/")
        self.session_name = _safe_component(
            session_name if session_name else default_session_name()
        )
        self.root_dir = Path(root_dir).expanduser()
        self.session_dir = self.root_dir / self.session_name
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=False, exist_ok=False)

        self._popen_factory = popen_factory
        self._run_factory = run_factory
        self._lock = threading.Lock()
        self._accepting = True
        self._closed = False
        self._manifest_path: Optional[Path] = None
        self.started_at_unix = time.time()
        self.ended_at_unix: Optional[float] = None
        self._states: dict[str, _CameraState] = {}

        try:
            for name in names:
                output_path = self.session_dir / "{}.mp4".format(name)
                log_handle = (self.session_dir / "{}.ffmpeg.log".format(name)).open(
                    "ab",
                    buffering=0,
                )
                command = self._build_ffmpeg_command(
                    ffmpeg_bin=ffmpeg_bin,
                    output_path=output_path,
                )
                process = self._popen_factory(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=log_handle,
                )
                if process.stdin is None:
                    raise RuntimeError(
                        "FFmpeg stdin was not created for camera {}".format(name)
                    )
                state = _CameraState(
                    name=name,
                    output_path=output_path,
                    frame_queue=queue.Queue(maxsize=self.queue_size),
                    process=process,
                    log_handle=log_handle,
                )
                self._states[name] = state
                state.thread = threading.Thread(
                    target=self._writer_loop,
                    args=(state,),
                    name="camera-recorder-{}".format(name),
                    daemon=True,
                )
                state.thread.start()
        except Exception:
            self.close()
            raise

    def _build_ffmpeg_command(
        self,
        *,
        ffmpeg_bin: str,
        output_path: Path,
    ) -> list[str]:
        fps_text = "{:g}".format(self.fps)
        return [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-y",
            "-f",
            "image2pipe",
            "-framerate",
            fps_text,
            "-vcodec",
            "mjpeg",
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]

    def submit(self, camera_name: str, jpeg_payload: bytes) -> None:
        """Queue the newest complete JPEG frame without blocking the caller."""
        if not isinstance(jpeg_payload, bytes):
            jpeg_payload = bytes(jpeg_payload)
        with self._lock:
            if not self._accepting:
                return
            if camera_name not in self._states:
                raise KeyError("unknown recording camera {!r}".format(camera_name))
            state = self._states[camera_name]
            state.received += 1

        try:
            state.frame_queue.put_nowait(jpeg_payload)
            return
        except queue.Full:
            pass

        try:
            state.frame_queue.get_nowait()
            state.frame_queue.task_done()
            with self._lock:
                state.dropped += 1
        except queue.Empty:
            pass

        try:
            state.frame_queue.put_nowait(jpeg_payload)
        except queue.Full:
            with self._lock:
                state.dropped += 1

    def _writer_loop(self, state: _CameraState) -> None:
        writer_failed = False
        try:
            while True:
                item = state.frame_queue.get()
                try:
                    if item is _STOP:
                        break
                    if writer_failed:
                        with self._lock:
                            state.dropped += 1
                        continue
                    try:
                        state.process.stdin.write(item)
                        with self._lock:
                            state.written += 1
                    except Exception as exc:
                        writer_failed = True
                        with self._lock:
                            state.error = "FFmpeg write failed: {}".format(exc)
                            state.dropped += 1
                finally:
                    state.frame_queue.task_done()
        finally:
            try:
                state.process.stdin.close()
            except Exception as exc:
                with self._lock:
                    if state.error is None:
                        state.error = "FFmpeg stdin close failed: {}".format(exc)
            try:
                state.returncode = state.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    state.process.terminate()
                    state.returncode = state.process.wait(timeout=5)
                except Exception:
                    try:
                        state.process.kill()
                        state.returncode = state.process.wait(timeout=5)
                    except Exception as exc:
                        with self._lock:
                            state.error = "FFmpeg shutdown failed: {}".format(exc)
            except Exception as exc:
                with self._lock:
                    state.error = "FFmpeg wait failed: {}".format(exc)

            if state.returncode not in (None, 0):
                with self._lock:
                    state.error = "FFmpeg exited with code {}".format(
                        state.returncode
                    )
            try:
                state.log_handle.close()
            except Exception:
                pass

    def close(self) -> None:
        """Finalize all MP4 files. Safe to call more than once."""
        with self._lock:
            if self._closed:
                return
            self._accepting = False
            self._closed = True
            states = tuple(self._states.values())

        for state in states:
            if state.thread is not None:
                state.frame_queue.put(_STOP)
        for state in states:
            if state.thread is not None:
                state.thread.join()
            else:
                try:
                    state.process.stdin.close()
                    state.returncode = state.process.wait(timeout=30)
                except Exception:
                    pass
                try:
                    state.log_handle.close()
                except Exception:
                    pass
        self.ended_at_unix = time.time()

    def stats(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                name: {
                    "file": state.output_path.name,
                    "received": state.received,
                    "written": state.written,
                    "dropped": state.dropped,
                    "ffmpeg_returncode": state.returncode,
                    "error": state.error,
                }
                for name, state in self._states.items()
            }

    @property
    def remote_session_dir(self) -> str:
        return "{}/{}".format(self.upload_dir, self.session_name)

    def write_manifest(self) -> Path:
        if not self._closed:
            self.close()
        manifest = {
            "session_name": self.session_name,
            "started_at_unix": self.started_at_unix,
            "ended_at_unix": self.ended_at_unix,
            "duration_seconds": (
                None
                if self.ended_at_unix is None
                else self.ended_at_unix - self.started_at_unix
            ),
            "fps": self.fps,
            "queue_size": self.queue_size,
            "local_session_dir": str(self.session_dir),
            "upload_host": self.upload_host,
            "remote_session_dir": self.remote_session_dir,
            "cameras": self.stats(),
        }
        manifest_path = self.session_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        self._manifest_path = manifest_path
        return manifest_path

    def upload(self) -> str:
        """Upload a finalized session, preserving the local copy."""
        if not self._closed:
            self.close()
        if self._manifest_path is None or not self._manifest_path.exists():
            self.write_manifest()

        ssh_options = [
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
        ]
        mkdir_command = "mkdir -p -- {}".format(shlex.quote(self.upload_dir))
        self._run_factory(
            [
                "ssh",
                "-p",
                str(self.upload_port),
                *ssh_options,
                self.upload_host,
                mkdir_command,
            ],
            check=True,
        )
        self._run_factory(
            [
                "scp",
                "-P",
                str(self.upload_port),
                *ssh_options,
                "-r",
                str(self.session_dir),
                "{}:{}/".format(self.upload_host, self.upload_dir),
            ],
            check=True,
        )
        return self.remote_session_dir
