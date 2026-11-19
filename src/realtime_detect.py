#!/usr/bin/env python3
"""Real-time YOLOv8 object detection on the i.MX 8M Plus NPU, shown on screen.

Pipeline (all on the DART-MX8M-PLUS):
    C920 webcam (V4L2) --> OpenCV capture --> letterbox + INT8 quantize
    --> TFLite + VX delegate (VIP8000 NPU) --> decode + NMS
    --> draw boxes/labels --> cv2.imshow() on the Weston/LVDS display

The NPU is engaged only when --delegate points at libvx_delegate.so; otherwise
the model runs on the A53 CPU (useful for A/B comparison).

Example (on the board):
    export XDG_RUNTIME_DIR=/run/user/0
    python3 src/realtime_detect.py \
        --model models/yolov8n_int8.tflite \
        --labels models/coco_labels.txt \
        --camera 3 \
        --delegate /usr/lib/libvx_delegate.so
"""
import argparse
import os
import time

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:  # full TensorFlow fallback (host dev)
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
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


def letterbox(img, new_shape, color=(114, 114, 114)):
    """Resize keeping aspect ratio, pad to new_shape (w, h). Returns img, ratio, (dw, dh)."""
    h0, w0 = img.shape[:2]
    nw, nh = new_shape
    r = min(nw / w0, nh / h0)
    uw, uh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (uw, uh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (nw - uw) / 2, (nh - uh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)


def decode_yolov8(output, conf_thres, iou_thres, input_wh):
    """output: dequantized array shaped [1, 4+nc, N] or [1, N, 4+nc].
    Returns boxes (xywh in input-pixel space), scores, class_ids after NMS."""
    o = np.squeeze(output)                      # -> [4+nc, N] or [N, 4+nc]
    if o.shape[0] < o.shape[1]:                 # [4+nc, N] -> [N, 4+nc]
        o = o.transpose(1, 0)
    boxes_xywh = o[:, :4]
    class_scores = o[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_scores)), class_ids]

    keep = confidences > conf_thres
    boxes_xywh, confidences, class_ids = boxes_xywh[keep], confidences[keep], class_ids[keep]
    if len(boxes_xywh) == 0:
        return [], [], []

    iw, ih = input_wh
    # Ultralytics TFLite exports coords normalized to [0,1]; scale to pixels.
    if boxes_xywh.max() <= 1.5:
        boxes_xywh = boxes_xywh * np.array([iw, ih, iw, ih], dtype=np.float32)

    # xywh (center) -> xyxy then to cv2 rects (x, y, w, h)
    xy = boxes_xywh[:, :2]
    wh = boxes_xywh[:, 2:4]
    tl = xy - wh / 2
    rects = np.concatenate([tl, wh], axis=1).astype(int).tolist()

    idxs = cv2.dnn.NMSBoxes(rects, confidences.tolist(), conf_thres, iou_thres)
    if len(idxs) == 0:
        return [], [], []
    idxs = np.array(idxs).flatten()
    return [rects[i] for i in idxs], [float(confidences[i]) for i in idxs], \
           [int(class_ids[i]) for i in idxs]


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def color_for(cid):
    np.random.seed(cid * 7 + 1)
    return tuple(int(c) for c in np.random.randint(60, 255, 3))


def draw(frame, rects, scores, cids, labels, ratio, pad):
    (lx, ty) = pad
    for (x, y, w, h), s, cid in zip(rects, scores, cids):
        # map from letterboxed input space back to original frame
        x0 = int((x - lx) / ratio); y0 = int((y - ty) / ratio)
        x1 = int((x + w - lx) / ratio); y1 = int((y + h - ty) / ratio)
        col = color_for(cid)
        name = labels[cid] if cid < len(labels) else str(cid)
        cv2.rectangle(frame, (x0, y0), (x1, y1), col, 2)
        tag = f"{name} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x0, y0 - th - 6), (x0 + tw + 2, y0), col, -1)
        cv2.putText(frame, tag, (x0 + 1, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/yolov8n_int8.tflite")
    ap.add_argument("--labels", default="models/coco_labels.txt")
    ap.add_argument("--camera", type=int, default=3, help="V4L2 index (C920 = 3)")
    ap.add_argument("--delegate", default="/usr/lib/libvx_delegate.so",
                    help="path to libvx_delegate.so; pass '' to force CPU")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--cap-w", type=int, default=1280)
    ap.add_argument("--cap-h", type=int, default=720)
    ap.add_argument("--headless", action="store_true",
                    help="no window; write annotated frames to outputs/")
    args = ap.parse_args()

    # Make GUI reachable when launched from the serial console.
    os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/0")

    labels = [l.strip() for l in open(args.labels)] if os.path.exists(args.labels) else []
    interp = build_interpreter(args.model, args.delegate or None)
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()
    _, ih, iw, _ = inp["shape"]
    in_scale, in_zp = inp["quantization"]
    print(f"[model] input {iw}x{ih} dtype={inp['dtype'].__name__} "
          f"quant=({in_scale:.5f},{in_zp})  delegate={'NPU' if args.delegate else 'CPU'}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_h)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera index {args.camera}")

    if args.headless:
        os.makedirs("outputs", exist_ok=True)

    win = "yolo_dart (NPU)"
    fps, n = 0.0, 0
    t_fps = time.time()
    print("[run] press 'q' in the window (or Ctrl-C) to quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[warn] dropped frame")
                continue

            lb, ratio, pad = letterbox(frame, (iw, ih))
            rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
            if inp["dtype"] in (np.int8, np.uint8):
                x = (rgb.astype(np.float32) / 255.0) / in_scale + in_zp
                x = np.clip(x, np.iinfo(inp["dtype"]).min, np.iinfo(inp["dtype"]).max)
                x = x.astype(inp["dtype"])
            else:
                x = rgb.astype(np.float32) / 255.0
            x = np.expand_dims(x, 0)

            t0 = time.time()
            interp.set_tensor(inp["index"], x)
            interp.invoke()
            infer_ms = (time.time() - t0) * 1000

            od = outs[0]
            out = interp.get_tensor(od["index"]).astype(np.float32)
            o_scale, o_zp = od["quantization"]
            if o_scale and od["dtype"] in (np.int8, np.uint8):
                out = (out - o_zp) * o_scale

            rects, scores, cids = decode_yolov8(out, args.conf, args.iou, (iw, ih))
            draw(frame, rects, scores, cids, labels, ratio, pad)

            n += 1
            if time.time() - t_fps >= 0.5:
                fps = n / (time.time() - t_fps)
                n, t_fps = 0, time.time()
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
