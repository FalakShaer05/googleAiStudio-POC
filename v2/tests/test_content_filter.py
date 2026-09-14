import unittest
from unittest.mock import Mock, patch

from creative_system.stations.content_filter.classifier import (
    classify_content,
    normalize_result,
    validate_input,
)


class ContentFilterTests(unittest.TestCase):
    def test_allowed_result_always_has_null_reason(self):
        self.assertEqual(
            normalize_result({"blocked": False, "reason": "unused"}),
            {"blocked": False, "reason": None},
        )

    def test_blocked_result_requires_a_useful_reason(self):
        self.assertEqual(
            normalize_result({"blocked": True, "reason": "Promotional content is not allowed."}),
            {"blocked": True, "reason": "Promotional content is not allowed."},
        )
        with self.assertRaises(ValueError):
            normalize_result({"blocked": True, "reason": None})

    def test_input_validation(self):
        self.assertEqual(validate_input(" WISH ", " Hope for joy. "), ("wish", "Hope for joy."))
        with self.assertRaises(ValueError):
            validate_input("story", "Once upon a time")
        with self.assertRaises(ValueError):
            validate_input("wisdom", " ")

    @patch("creative_system.stations.content_filter.classifier.get_gemini_client")
    def test_classification_uses_structured_gemini_result(self, get_client):
        response = Mock()
        response.text = '{"blocked": true, "reason": "This is promotional content."}'
        get_client.return_value.models.generate_content.return_value = response

        result = classify_content("wish", "Buy my product today!")

        self.assertEqual(
            result,
            {"blocked": True, "reason": "This is promotional content."},
        )
        call = get_client.return_value.models.generate_content.call_args
        self.assertIn("SELECTED ENTRY TYPE: wish", call.kwargs["contents"])
        self.assertIn("Buy my product today!", call.kwargs["contents"])


if __name__ == "__main__":
    unittest.main()
