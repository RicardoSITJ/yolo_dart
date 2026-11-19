#!/usr/bin/env python3
"""Run a quantized (INT8) YOLO TFLite model on the i.MX 8M Plus NPU.

The NPU (Verisilicon VIP8000) is used through TensorFlow Lite's external
VX delegate (``libvx_delegate.so``). Without the delegate the model runs on
the CPU, so passing ``--delegate`` is what actually engages the NPU.

Example (on the DART):
    python3 src/detect.py \
        --model models/yolov8n_int8.tflite \
        --source test.jpg \
        --delegate /usr/lib/libvx_delegate.so
"""
import argparse
import time

import cv2
import numpy as np

try:
    # On the BSP this is the lightweight runtime.
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:  # fall back to full TF on a host PC
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate


def build_interpreter(model_path: str, delegate_path: str | None):
    delegates = []
    if delegate_path:
        print(f"[npu] loading VX delegate: {delegate_path}")
        delegates = [load_delegate(delegate_path)]
    interp = Interpreter(model_path=model_path, experimental_delegates=delegates)
    interp.allocate_tensors()
    return interp


def preprocess(image_bgr, in_shape, in_dtype, scale, zero_point):
    _, h, w, _ = in_shape
    img = cv2.resize(image_bgr, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if in_dtype == np.int8 or in_dtype == np.uint8:
        # Quantize float [0,1] -> int using the model's input quant params.
        img = (img.astype(np.float32) / 255.0) / scale + zero_point
        img = np.clip(img, np.iinfo(in_dtype).min, np.iinfo(in_dtype).max)
        img = img.astype(in_dtype)
    else:
        img = (img.astype(np.float32) / 255.0)
    return np.expand_dims(img, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="path to INT8 .tflite model")
    ap.add_argument("--source", required=True, help="image file to run detection on")
    ap.add_argument("--delegate", default=None,
                    help="path to libvx_delegate.so to run on the NPU (omit for CPU)")
    ap.add_argument("--warmup", type=int, default=1, help="warmup iterations")
    args = ap.parse_args()

    interp = build_interpreter(args.model, args.delegate)
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()

    scale, zero_point = in_det["quantization"]
    image = cv2.imread(args.source)
    if image is None:
        raise SystemExit(f"could not read image: {args.source}")
    x = preprocess(image, in_det["shape"], in_det["dtype"], scale or 1.0, zero_point)

    # Warmup (first NPU run compiles the graph — can take tens of seconds).
    for _ in range(max(0, args.warmup)):
        interp.set_tensor(in_det["index"], x)
        interp.invoke()

    t0 = time.perf_counter()
    interp.set_tensor(in_det["index"], x)
    interp.invoke()
    dt = (time.perf_counter() - t0) * 1000

    where = "NPU" if args.delegate else "CPU"
    print(f"[{where}] inference: {dt:.1f} ms")
    for od in out_det:
        print(f"  output {od['name']}: shape={od['shape']} dtype={od['dtype'].__name__}")

    # NOTE: decoding YOLO outputs (dequantize -> boxes -> NMS) depends on the
    # exact model head and is left as the next implementation step.


if __name__ == "__main__":
    main()
