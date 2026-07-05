#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Host B websocket policy server."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import json
import os
import traceback
from typing import Any, Callable, Dict, Optional

import websockets

from ws_policy_protocol import build_policy_response, decode_policy_request


PolicyFn = Callable[[Dict[str, Any]], Any]


def load_policy_function(spec: Optional[str]) -> PolicyFn:
    """Load `module:function` or `/path/to/file.py:function`."""

    if not spec:
        from example_remote_policy import predict_action

        return predict_action

    if ":" not in spec:
        raise ValueError("--policy must look like module_name:function_name")

    module_spec, function_name = spec.split(":", 1)
    if module_spec.endswith(".py") or os.path.sep in module_spec:
        module_path = os.path.abspath(module_spec)
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        spec_obj = importlib.util.spec_from_file_location(module_name, module_path)
        if spec_obj is None or spec_obj.loader is None:
            raise RuntimeError("failed to load policy module from {}".format(module_path))
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)

    return getattr(module, function_name)


class PolicyServer:
    def __init__(self, policy_fn: PolicyFn) -> None:
        self.policy_fn = policy_fn

    async def handler(self, websocket: Any) -> None:
        peer = getattr(websocket, "remote_address", None)
        print("[+] client connected: {}".format(peer))
        try:
            async for message in websocket:
                response = await self.handle_message(message)
                await websocket.send(json.dumps(response, ensure_ascii=False))
        except websockets.exceptions.ConnectionClosed:
            print("[-] client disconnected: {}".format(peer))
        except Exception as exc:
            print("[!] connection error with {}: {}".format(peer, exc))
            traceback.print_exc()

    async def handle_message(self, message: str) -> Dict[str, Any]:
        request_id = "unknown"
        step = -1
        try:
            raw_request = json.loads(message)
            request_id = raw_request.get("request_id", request_id)
            step = int(raw_request.get("step", step))
            request = decode_policy_request(raw_request)

            maybe_action = self.policy_fn(request)
            if asyncio.iscoroutine(maybe_action):
                action = await maybe_action
            else:
                action = maybe_action

            print(
                "[step {}] instruction={!r} request_id={} images={} state_keys={}".format(
                    request["step"],
                    request["instruction"],
                    request["request_id"],
                    list(request.get("images", {}).keys()),
                    list(request.get("state", {}).keys()),
                )
            )
            return build_policy_response(
                request_id=request["request_id"],
                step=request["step"],
                action=action,
            )
        except Exception as exc:
            print("[!] failed to process request: {}".format(exc))
            traceback.print_exc()
            return build_policy_response(
                request_id=request_id,
                step=step,
                action={},
                error=str(exc),
            )


async def serve(args: argparse.Namespace) -> None:
    policy_fn = load_policy_function(args.policy)
    server = PolicyServer(policy_fn)

    async with websockets.serve(
        server.handler,
        args.host,
        args.port,
        max_size=args.max_size_mb * 1024 * 1024,
        ping_interval=20,
        ping_timeout=60,
    ):
        print("policy server listening on ws://{}:{}".format(args.host, args.port))
        await asyncio.Future()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Piper websocket policy server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7081)
    parser.add_argument(
        "--policy",
        default="example_remote_policy:predict_action",
        help="module:function or /abs/path/to/file.py:function",
    )
    parser.add_argument("--max-size-mb", type=int, default=32)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(serve(args))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\npolicy server stopped")
