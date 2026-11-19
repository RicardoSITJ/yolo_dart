#!/usr/bin/env python3
"""Real-time YOLO object detection on the i.MX 8M Plus NPU, shown on screen.

Pipeline (all on the DART-MX8M-PLUS):
    C920 webcam (V4L2) --> OpenCV capture --> letterbox + quantize
    --> TFLite + VX delegate (VIP8000 NPU) --> DFL decode + NMS
    --> draw boxes/labels --> cv2.imshow() on the Weston/LVDS display

The NPU is engaged only when --delegate points at libvx_delegate.so; otherwise
the model runs on the A53 CPU (useful for A/B comparison).

Example (on the board):
    export XDG_RUNTIME_DIR=/run/user/0
    python3 src/realtime_detect.py \
        --model models/yolov5n-det-int8.tflite \
        --labels models/coco_labels.txt \
        --camera 3 --delegate /usr/lib/libvx_delegate.so
"""
import argparse
import os
import time

import cv2
import numpy as np

import yolo_decode as yd

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate


def build_interpreter(model_path, delegate_path):
    delegates = []
    if delegate_path:
        if not os.path.exists(delegate_path):
            raise FileNotFoundError(f"delegate not found: {delegate_path}")
        print(f"[npu] loading VX delegate: {delegate_path}")
        delegates = [load_delegate(delegate_path)]
    interp = Interpreter(model_path=model_path, experimental_delegates=delegates)
    interp.allocate_tensors()
    return interp


def quantize_input(rgb, inp):
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
    ap.add_argument("--camera", type=int, default=3, help="V4L2 index (C920 = 3)")
    ap.add_argument("--delegate", default="/usr/lib/libvx_delegate.so",
                    help="path to libvx_delegate.so; pass '' to force CPU")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--cap-w", type=int, default=1280)
    ap.add_argument("--cap-h", type=int, default=720)
    ap.add_argument("--headless", action="store_true",
                    help="no window; write annotated frames to outputs/latest.jpg")
    args = ap.parse_args()

    os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/0")

    labels = [l.strip() for l in open(args.labels)] if os.path.exists(args.labels) else []
    interp = build_interpreter(args.model, args.delegate or None)
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()
    _, ih, iw, _ = inp["shape"]
    mode = "raw-DFL head" if len(outs) >= 6 else "decoded head"
    print(f"[model] input {iw}x{ih} {inp['dtype'].__name__} | {len(outs)} outputs ({mode}) "
          f"| {'NPU' if args.delegate else 'CPU'}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_h)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera index {args.camera}")
    if args.headless:
        os.makedirs("outputs", exist_ok=True)

    win = "yolo_dart (NPU)"
    fps, n, t_fps = 0.0, 0, time.time()
    print("[run] press 'q' in the window (or Ctrl-C) to quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[warn] dropped frame"); continue

            lb, ratio, pad = yd.letterbox(frame, (iw, ih))
            rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
            x = quantize_input(rgb, inp)

            t0 = time.time()
            interp.set_tensor(inp["index"], x)
            interp.invoke()
            infer_ms = (time.time() - t0) * 1000

            rects, scores, cids = yd.decode(interp, outs, args.conf, args.iou, (iw, ih))
            yd.draw(frame, rects, scores, cids, labels, ratio, pad)

            n += 1
            if time.time() - t_fps >= 0.5:
                fps, n, t_fps = n / (time.time() - t_fps), 0, time.time()
            cv2.putText(frame, f"{fps:4.1f} FPS | {infer_ms:4.0f} ms | {len(rects)} obj",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            if args.headless:
                cv2.imwrite("outputs/latest.jpg", frame)
            else:
                cv2.imshow(win, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[done]")


if __name__ == "__main__":
    main()
