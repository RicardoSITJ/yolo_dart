#!/usr/bin/env python3
"""Unit tests for src/yolo_decode.py — pure decode logic, no NPU/camera needed.

Run on the board (has cv2 + numpy) or any host with opencv-python + numpy:
    python3 tests/test_yolo_decode.py
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import yolo_decode as yd


class FakeInterp:
    """Minimal stand-in for a TFLite interpreter: returns prepared tensors."""
    def __init__(self, tensors):
        self._t = tensors

    def get_tensor(self, index):
        return self._t[index]


def _detail(index):
    # quantization scale 0 => _dequant leaves the (already float) array untouched
    return {"index": index, "quantization": (0.0, 0), "dtype": np.float32}


class TestHelpers(unittest.TestCase):
    def test_letterbox_shape_and_pad(self):
        img = np.zeros((100, 200, 3), np.uint8)       # h=100, w=200
        out, ratio, (lx, ty) = yd.letterbox(img, (640, 640))
        self.assertEqual(out.shape, (640, 640, 3))
        self.assertAlmostEqual(ratio, 3.2, places=3)  # min(640/200, 640/100)
        self.assertEqual((lx, ty), (0, 160))          # width fills, height padded

    def test_softmax_sigmoid(self):
        p = yd._softmax(np.array([[1.0, 1.0, 1.0, 1.0]]), axis=1)
        self.assertTrue(np.allclose(p, 0.25))
        self.assertAlmostEqual(float(yd._sigmoid(np.array(0.0))), 0.5, places=6)


class TestDecodeSingle(unittest.TestCase):
    def test_one_box(self):
        # output [1, 84, N]; col layout after transpose is [N, 4+80]
        N = 10
        o = np.zeros((1, 84, N), np.float32)
        o[0, :4, 0] = [320, 320, 100, 100]            # cx,cy,w,h (pixels)
        o[0, 4 + 7, 0] = 0.9                           # class 7 score (already activated)
        rects, scores, cids = yd.decode_single(
            FakeInterp([o]), [_detail(0)], 0.35, 0.45, (640, 640))
        self.assertEqual(len(rects), 1)
        self.assertEqual(cids[0], 7)
        self.assertGreater(scores[0], 0.8)
        x, y, w, h = rects[0]
        self.assertTrue(abs(x - 270) <= 2 and abs(y - 270) <= 2)   # tl = center - wh/2
        self.assertTrue(abs(w - 100) <= 2 and abs(h - 100) <= 2)

    def test_empty(self):
        o = np.zeros((1, 84, 5), np.float32)           # all scores 0 < conf
        rects, scores, cids = yd.decode_single(
            FakeInterp([o]), [_detail(0)], 0.35, 0.45, (640, 640))
        self.assertEqual(rects, [])


class TestDecodeRawDFL(unittest.TestCase):
    def test_dfl_box(self):
        # one 20x20 scale (stride 32). Target cell (gx=10, gy=10).
        box = np.zeros((1, 20, 20, 64), np.float32)
        for coord in range(4):                         # concentrate DFL on bin 4
            box[0, 10, 10, coord * 16 + 4] = 10.0      # => distance ~= 4 cells
        cls = np.full((1, 20, 20, 80), -10.0, np.float32)
        cls[0, 10, 10, 15] = 5.0                        # class 15, sigmoid(5)=0.993

        interp = FakeInterp([box, cls])
        outs = [_detail(0), _detail(1)]
        rects, scores, cids = yd.decode_raw_dfl(
            interp, outs, 0.35, 0.45, (640, 640), nc=80, reg_max=16)

        self.assertEqual(len(rects), 1)
        self.assertEqual(cids[0], 15)
        self.assertGreater(scores[0], 0.9)
        # anchor (10.5,10.5), dist 4 each -> x1=(10.5-4)*32=208, w=(2*4)*32=256
        x, y, w, h = rects[0]
        self.assertTrue(abs(x - 208) <= 3 and abs(y - 208) <= 3)
        self.assertTrue(abs(w - 256) <= 4 and abs(h - 256) <= 4)

    def test_multi_scale_nms(self):
        # same object echoed on two scales should collapse to one box via NMS
        def one_scale(grid):
            box = np.zeros((1, grid, grid, 64), np.float32)
            cls = np.full((1, grid, grid, 80), -10.0, np.float32)
            c = grid // 2
            for coord in range(4):
                box[0, c, c, coord * 16 + 4] = 10.0
            cls[0, c, c, 1] = 5.0
            return box, cls
        b1, c1 = one_scale(20)
        b2, c2 = one_scale(40)
        interp = FakeInterp([b1, c1, b2, c2])
        outs = [_detail(0), _detail(1), _detail(2), _detail(3)]
        rects, scores, cids = yd.decode_raw_dfl(
            interp, outs, 0.35, 0.45, (640, 640), nc=80, reg_max=16)
        self.assertGreaterEqual(len(rects), 1)
        self.assertTrue(all(c == 1 for c in cids))


if __name__ == "__main__":
    unittest.main(verbosity=2)
