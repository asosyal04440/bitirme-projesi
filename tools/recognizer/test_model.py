from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import (  # noqa: E402
    ALPHABET,
    FIELD_ALPHABETS,
    INPUT_HEIGHT,
    MAX_WIDTH,
    DigitCRNN,
    decode,
    encode,
    label_text,
    preprocess,
    resize_bilinear,
    target_width,
)
from train import choose_threshold, export_onnx  # noqa: E402


def one_hot(path: list[int], confidence: float = 0.9) -> np.ndarray:
    classes = len(ALPHABET) + 1
    matrix = np.full((len(path), classes), (1 - confidence) / (classes - 1), np.float32)
    matrix[np.arange(len(path)), path] = confidence
    return matrix


class ContractTests(unittest.TestCase):
    def test_labels_normalize_decimal_separator_and_mark_corrections(self) -> None:
        self.assertEqual(label_text("19,5", False), "19.5")
        self.assertEqual(label_text("14", True), "~14")
        self.assertEqual(label_text("", False), "")
        self.assertEqual(encode("1.~"), [2, 11, 12])

    def test_greedy_decode_collapses_repeats_and_drops_blanks(self) -> None:
        one, two, dot = encode("1")[0], encode("2")[0], encode(".")[0]
        text, confidence = decode(one_hot([0, one, one, 0, one, two, two, dot, 0]))
        self.assertEqual(text, "112.")
        self.assertAlmostEqual(confidence, 0.9, places=5)
        matrix = one_hot([0, one, 0])
        matrix[2] = 1 / matrix.shape[1]  # one uncertain blank frame lowers the cell
        self.assertAlmostEqual(decode(matrix)[1], 1 / matrix.shape[1], places=5)
        self.assertEqual(decode(one_hot([0, 0, 0]))[0], "")

    def test_field_alphabet_ignores_impossible_characters(self) -> None:
        one, dot, tilde = encode("1")[0], encode(".")[0], encode("~")[0]
        matrix = one_hot([one, 0, dot, 0, one], confidence=0.6)
        matrix[2, encode("7")[0]] = 0.3  # second choice under the '.'
        self.assertEqual(decode(matrix)[0], "1.1")
        text, confidence = decode(matrix, FIELD_ALPHABETS["number"])
        self.assertEqual(text, "171")
        self.assertAlmostEqual(confidence, 0.3, places=5)  # not renormalised
        self.assertEqual(decode(one_hot([tilde, 0, one]), FIELD_ALPHABETS["number"])[0], "1")

    def test_bilinear_resize_matches_opencv_linear(self) -> None:
        image = np.random.default_rng(0).uniform(0, 255, (64, 150, 3)).astype(np.float32)
        for size in [(32, 75), (32, 300), (17, 41)]:
            expected = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
            np.testing.assert_allclose(resize_bilinear(image, *size), expected, atol=1e-3)

    def test_preprocess_scales_pads_white_and_bounds_width(self) -> None:
        crop = np.zeros((64, 150, 3), np.uint8)
        tensor = preprocess(crop)
        self.assertEqual(tensor.shape, (3, INPUT_HEIGHT, 76))  # 75 rounded up to a multiple of 4
        self.assertTrue(np.all(tensor[:, :, :75] == -1.0))
        self.assertTrue(np.all(tensor[:, :, 75:] == 1.0))
        self.assertEqual(preprocess(np.zeros((10, 4000, 3), np.uint8)).shape[2], MAX_WIDTH)
        # 217 * 32 / 64 = 108.5 rounds half up (Python's round() would give 108).
        self.assertEqual(target_width(64, 217), 109)

    def test_onnx_export_matches_pytorch_for_any_width(self) -> None:
        import onnxruntime

        torch.manual_seed(0)
        model = DigitCRNN().eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.onnx"
            export_onnx(model, path)
            session = onnxruntime.InferenceSession(str(path))
            for width in (16, 112, 340):
                images = np.random.default_rng(width).uniform(-1, 1, (2, 3, 32, width)).astype(np.float32)
                actual = session.run(None, {"image": images})[0]
                expected = model(torch.from_numpy(images)).softmax(dim=2).detach().numpy()
                self.assertEqual(actual.shape, (2, width // 4, len(ALPHABET) + 1))
                np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_threshold_is_the_lowest_meeting_the_target_with_enough_support(self) -> None:
        curve = [
            {"threshold": 0.5, "coverage": 1.0, "accepted_exact_match": 0.95},
            {"threshold": 0.8, "coverage": 0.9, "accepted_exact_match": 0.99},
            {"threshold": 0.9, "coverage": 0.01, "accepted_exact_match": 1.0},
        ]
        self.assertEqual(choose_threshold(curve, 0.99, 100, 1000), 0.8)
        self.assertIsNone(choose_threshold(curve, 0.999, 100, 1000))  # 0.9 has too few fields


if __name__ == "__main__":
    unittest.main()
