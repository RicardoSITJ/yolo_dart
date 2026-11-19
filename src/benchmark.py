#!/usr/bin/env python3
"""Benchmark YOLO inference on the i.MX 8M Plus NPU (and optionally the CPU).

Reproducible by default: it runs a fixed synthetic input through the model, so the
numbers do not depend on a camera or a particular image. Reports the cold-start
(graph compile) time, steady-state latency statistics and the implied FPS.

Examples (on the board):
    # NPU latency (default)
    python3 src/benchmark.py

    # NPU vs CPU A/B
    python3 src/benchmark.py --cpu

    # End-to-end live pipeline FPS off the camera
    python3 src/benchmark.py --camera 3
"""
import argparse
import os
import statistics
import time

import cv2
import numpy as np

import yolo_decode as yd
import npu

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate


def make_interp(model, delegate, cache_dir):
    delegates = []
    if delegate:
        npu.enable_npu_cache(cache_dir)
        delegates = [load_delegate(delegate)]
    it = Interpreter(model_path=model, experimental_delegates=delegates)
    it.allocate_tensors()
    return it


def fixed_input(inp):
    """Deterministic input tensor matching the model's shape/dtype."""
    _, h, w, c = inp["shape"]
    rng = np.random.RandomState(0)
    rgb = rng.randint(0, 256, size=(h, w, c)).astype(np.float32)
    scale, zp = inp["quantization"]
    if inp["dtype"] in (np.int8, np.uint8):
        x = (rgb / 255.0) / (scale or 1.0) + zp
        lo, hi = np.iinfo(inp["dtype"]).min, np.iinfo(inp["dtype"]).max
        x = np.clip(x, lo, hi).astype(inp["dtype"])
    else:
        x = rgb / 255.0
    return np.expand_dims(x, 0)


def stats(times_ms, label):
    times_ms = sorted(times_ms)
    avg = statistics.mean(times_ms)
    p95 = times_ms[min(len(times_ms) - 1, int(0.95 * len(times_ms)))]
    print(f"\n[{label}]  {len(times_ms)} iters")
    print(f"  avg {avg:6.1f} ms   median {statistics.median(times_ms):6.1f} ms"
          f"   min {times_ms[0]:6.1f} ms   p95 {p95:6.1f} ms")
    print(f"  => {1000.0/avg:5.1f} inferences/s")
    return avg


def bench_model(model, delegate, cache_dir, iters, warmup, label):
    it = make_interp(model, delegate, cache_dir)
    inp = it.get_input_details()[0]
    x = fixed_input(inp)

    t0 = time.time()
    it.set_tensor(inp["index"], x)
    it.invoke()                                   # first call compiles the graph
    cold = (time.time() - t0) * 1000
    print(f"[{label}] cold-start (graph compile + first invoke): {cold:.0f} ms")

    for _ in range(warmup):
        it.set_tensor(inp["index"], x); it.invoke()
    times = []
    for _ in range(iters):
        t = time.time()
        it.set_tensor(inp["index"], x); it.invoke()
        times.append((time.time() - t) * 1000)
    return stats(times, label)


def bench_camera(model, delegate, cache_dir, labels, cam, conf, iou, frames):
    it = make_interp(model, delegate, cache_dir)
    inp = it.get_input_details()[0]
    outs = it.get_output_details()
    _, ih, iw, _ = inp["shape"]
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {cam}")
    # warmup (compile + first frames)
    for _ in range(5):
        ok, frame = cap.read()
        if not ok:
            continue
        lb, _, _ = yd.letterbox(frame, (iw, ih))
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        x = fixed_input(inp) if frame is None else _quant(rgb, inp)
        it.set_tensor(inp["index"], x); it.invoke()

    times = []
    for _ in range(frames):
        t = time.time()
        ok, frame = cap.read()
        if not ok:
            continue
        lb, ratio, pad = yd.letterbox(frame, (iw, ih))
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        x = _quant(rgb, inp)
        it.set_tensor(inp["index"], x); it.invoke()
        yd.decode(it, outs, conf, iou, (iw, ih))
        times.append((time.time() - t) * 1000)
    cap.release()
    stats(times, "end-to-end (camera capture+NPU+decode)")


def _quant(rgb, inp):
    scale, zp = inp["quantization"]
    if inp["dtype"] in (np.int8, np.uint8):
        x = (rgb.astype(np.float32) / 255.0) / (scale or 1.0) + zp
        lo, hi = np.iinfo(inp["dtype"]).min, np.iinfo(inp["dtype"]).max
        x = np.clip(x, lo, hi).astype(inp["dtype"])
    else:
        x = rgb.astype(np.float32) / 255.0
    return np.expand_dims(x, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/yolov5n-det-int8.tflite")
    ap.add_argument("--labels", default="models/coco_labels.txt")
    ap.add_argument("--delegate", default="/usr/lib/libvx_delegate.so")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--cpu", action="store_true", help="also benchmark the CPU for A/B")
    ap.add_argument("--camera", type=int, default=None,
                    help="measure end-to-end FPS off this camera index instead")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--cache-dir", default="~/.cache/yolo_dart/npu")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    cache = None if args.no_cache else args.cache_dir

    if args.camera is not None:
        labels = [l.strip() for l in open(args.labels)] if os.path.exists(args.labels) else []
        bench_camera(args.model, args.delegate, cache, labels,
                     args.camera, args.conf, args.iou, args.frames)
        return

    npu_avg = bench_model(args.model, args.delegate, cache, args.iters, args.warmup, "NPU")
    if args.cpu:
        cpu_avg = bench_model(args.model, "", None, max(5, args.iters // 5),
                              args.warmup, "CPU")
        print(f"\n[A/B] NPU is {cpu_avg/npu_avg:.1f}x faster than CPU "
              f"({npu_avg:.0f} ms vs {cpu_avg:.0f} ms)")


if __name__ == "__main__":
    main()
