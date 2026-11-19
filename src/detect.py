#!/usr/bin/env python3
"""Single-image YOLO detection on the i.MX 8M Plus NPU (sanity check).

Runs one image through the TFLite model on the NPU (via the VX delegate),
prints detections, and writes an annotated image. Use this to validate the
model / decoder before running the live camera pipeline.

Example (on the board):
    python3 src/detect.py --model models/yolov5n-det-int8.tflite \
        --labels models/coco_labels.txt --source cat.jpg \
        --delegate /usr/lib/libvx_delegate.so --out outputs/cat_det.jpg
"""
import argparse
import os
import time

import cv2
import numpy as np

import yolo_decode as yd
import npu

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/yolov5n-det-int8.tflite")
    ap.add_argument("--labels", default="models/coco_labels.txt")
    ap.add_argument("--source", required=True, help="image file")
    ap.add_argument("--delegate", default="/usr/lib/libvx_delegate.so",
                    help="pass '' to force CPU")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--out", default="outputs/detection.jpg")
    ap.add_argument("--cache-dir", default="~/.cache/yolo_dart/npu",
                    help="persistent NPU graph cache (faster re-launch)")
    ap.add_argument("--no-cache", action="store_true", help="disable the NPU graph cache")
    args = ap.parse_args()

    labels = [l.strip() for l in open(args.labels)] if os.path.exists(args.labels) else []

    delegates = []
    if args.delegate:
        if not args.no_cache:
            npu.enable_npu_cache(args.cache_dir)   # must run before load_delegate
        print(f"[npu] loading VX delegate: {args.delegate}")
        delegates = [load_delegate(args.delegate)]
    interp = Interpreter(model_path=args.model, experimental_delegates=delegates)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()
    _, ih, iw, _ = inp["shape"]

    frame = cv2.imread(args.source)
    if frame is None:
        raise SystemExit(f"cannot read image: {args.source}")
    lb, ratio, pad = yd.letterbox(frame, (iw, ih))
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)

    scale, zp = inp["quantization"]
    if inp["dtype"] in (np.int8, np.uint8):
        x = (rgb.astype(np.float32) / 255.0) / (scale or 1.0) + zp
        lo, hi = np.iinfo(inp["dtype"]).min, np.iinfo(inp["dtype"]).max
        x = np.clip(x, lo, hi).astype(inp["dtype"])
    else:
        x = rgb.astype(np.float32) / 255.0
    x = np.expand_dims(x, 0)

    # warmup (first NPU run compiles the graph)
    interp.set_tensor(inp["index"], x); interp.invoke()
    t0 = time.time()
    interp.set_tensor(inp["index"], x); interp.invoke()
    infer_ms = (time.time() - t0) * 1000

    rects, scores, cids = yd.decode(interp, outs, args.conf, args.iou, (iw, ih))
    where = "NPU" if args.delegate else "CPU"
    print(f"[{where}] inference {infer_ms:.1f} ms | {len(rects)} detections:")
    for (x0, y0, w, h), s, cid in zip(rects, scores, cids):
        name = labels[cid] if cid < len(labels) else str(cid)
        print(f"  {name:15s} {s:.2f}  box=({x0},{y0},{w},{h})")

    yd.draw(frame, rects, scores, cids, labels, ratio, pad)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, frame)
    print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
