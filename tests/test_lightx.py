import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import requests
from dotenv import load_dotenv

from api.utils import remove_background_with_lightx_api


class MockResponse:
    def __init__(self, status_code=200, json_data=None, content=b""):
        self.status_code = status_code
        self._json = json_data or {}
        self.content = content

    def json(self):
        return self._json


SAMPLE_IMAGE_URL = "https://d2s4ngnid78ki4.cloudfront.net/1039096-494314513_9705683892825571_1793767886091926810_n.jpg"


class TestLightXPayload(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="lightx-test-")
        response = requests.get(SAMPLE_IMAGE_URL, timeout=30)
        response.raise_for_status()
        self.temp_image_path = os.path.join(self.temp_dir, "sample_input.jpg")
        with open(self.temp_image_path, "wb") as temp_file:
            temp_file.write(response.content)
        self.created_files = []

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        for path in self.created_files:
            if path and os.path.exists(path):
                os.remove(path)

    @patch("requests.get")
    @patch("requests.put")
    @patch("requests.post")
    def test_lightx_payload_logged_and_sent(self, mock_post, mock_put, mock_get):
        load_dotenv()

        captured_payloads = []

        upload_response_body = {
            "status": "SUCCESS",
            "statusCode": 2000,
            "body": {
                "uploadImage": "https://upload.lightxeditor.com/file",
                "imageUrl": "https://cdn.lightxeditor.com/test-image"
            }
        }

        remove_bg_response_body = {
            "status": "SUCCESS",
            "statusCode": 2000,
            "body": {
                "orderId": "order123",
                "maxRetriesAllowed": 5,
                "avgResponseTimeInSec": 1
            }
        }

        order_status_body = {
            "status": "SUCCESS",
            "statusCode": 2000,
            "body": {
                "status": "active",
                "output": "https://cdn.lightxeditor.com/result.png"
            }
        }

        def post_side_effect(url, headers=None, json=None, timeout=30):
            if url.endswith("/uploadImageUrl"):
                raise AssertionError("Upload flow should not be used when public URL is available")
            if url.endswith("/remove-background"):
                captured_payloads.append(json)
                return MockResponse(200, remove_bg_response_body)
            if url.endswith("/order-status"):
                order_status_body["body"]["status"] = "active" if len(self.created_files) == 0 else "complete"
                return MockResponse(200, order_status_body)
            raise AssertionError(f"Unexpected POST url {url}")

        mock_put.return_value = MockResponse(200)
        mock_get.return_value = MockResponse(200, content=b"fake-image-bytes")

        mock_post.side_effect = post_side_effect

        env = {
            "LIGHTX_API_KEY": "test_key",
            "LIGHTX_IMAGE_BASE_URL": "https://dev-google-ai.mosida.com/outputs",
            "LIGHTX_IMAGE_BASE_PATH": self.temp_dir
        }

        with patch.dict(os.environ, env, clear=False):
            result_path = remove_background_with_lightx_api(self.temp_image_path)
        self.created_files.append(result_path)

        self.assertIsNotNone(result_path)
        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(len(captured_payloads), 1)

        payload = captured_payloads[0]
        expected_url = "https://dev-google-ai.mosida.com/outputs/sample_input.jpg"
        self.assertEqual(payload["imageUrl"], expected_url)
        self.assertEqual(payload["background"], "")
        self.assertEqual(payload["prompt"], "")


if __name__ == "__main__":
    unittest.main()

