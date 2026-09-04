from __future__ import annotations

import base64
import importlib.util
import io
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image


CLIENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CLIENT_DIR))

from ws_policy_protocol import FASTWAM_TRANSPORT_IMAGE_SIZES, build_policy_request, encode_image  # noqa: E402


def jpeg_bytes(width: int, height: int) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


@unittest.skipUnless(importlib.util.find_spec("cv2"), "transport JPEG resize requires cv2")
class WebsocketPolicyProtocolTest(unittest.TestCase):
    def test_compressed_jpeg_is_preserved_without_transport_profile(self) -> None:
        source = jpeg_bytes(640, 480)

        encoded = encode_image({"encoding": "jpeg", "data": source})

        self.assertEqual(base64.b64decode(encoded["data"]), source)
        self.assertIsNone(encoded["width"])
        self.assertIsNone(encoded["height"])

    def test_fastwam_transport_profile_resizes_each_compressed_camera_before_send(self) -> None:
        source = jpeg_bytes(640, 480)
        images = {name: {"encoding": "jpeg", "data": source} for name in FASTWAM_TRANSPORT_IMAGE_SIZES}

        request = build_policy_request(
            step=0,
            instruction="crimp",
            state={"left_arm": [0.0] * 7, "right_arm": [0.0] * 7},
            images=images,
            transport_image_sizes=FASTWAM_TRANSPORT_IMAGE_SIZES,
        )

        for name, (width, height) in FASTWAM_TRANSPORT_IMAGE_SIZES.items():
            payload = request["images"][name]
            self.assertEqual((payload["width"], payload["height"]), (width, height))
            with Image.open(io.BytesIO(base64.b64decode(payload["data"]))) as image:
                self.assertEqual(image.size, (width, height))


if __name__ == "__main__":
    unittest.main()
