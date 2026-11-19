"""YOLO post-processing for the i.MX 8M Plus NPU.

Supports two kinds of TFLite exports:

1. **Raw decoupled head** (6 output tensors) — e.g. EdgeFirst
   `yolov5n-det-int8-smart.tflite`. The in-graph DFL decode / concat is stripped
   so the NPU runs the conv backbone + head and we do the DFL math here. Outputs
   are, per scale (strides 8/16/32): a box tensor `[1,H,W,4*reg_max]` and a class
   tensor `[1,H,W,nc]`.

2. **Decoded head** (1 output tensor `[1, 4+nc, 8400]` or `[1, 8400, 4+nc]`) —
   e.g. a standard Ultralytics `*_full_integer_quant.tflite`.

Both return cv2-style rects (x, y, w, h) in the letterboxed input space, plus
scores and class ids, already NMS-filtered.
"""
import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Pre-processing
# --------------------------------------------------------------------------- #
def letterbox(img, new_shape, color=(114, 114, 114)):
    """Resize keeping aspect ratio and pad to (w, h). Returns img, ratio, (dw, dh)."""
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


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _dequant(arr, detail):
    scale, zp = detail["quantization"]
    arr = arr.astype(np.float32)
    if scale:
        arr = (arr - zp) * scale
    return arr


# --------------------------------------------------------------------------- #
# Decoder 1: raw decoupled DFL head (6 tensors)
# --------------------------------------------------------------------------- #
def decode_raw_dfl(interp, outs, conf_thres, iou_thres, input_wh,
                   nc=80, reg_max=16):
    iw, ih = input_wh
    # group tensors by grid size, classifying each as the box (4*reg_max ch) or
    # the class tensor by its channel count (not as a dict key, so it is robust
    # even if nc happened to equal 4*reg_max).
    by_grid = {}
    for od in outs:
        a = _dequant(interp.get_tensor(od["index"]), od)[0]   # [H,W,C]
        H, W, C = a.shape
        slot = by_grid.setdefault((H, W), {"box": None, "cls": None})
        slot["box" if C == 4 * reg_max else "cls"] = a

    all_boxes, all_scores, all_cids = [], [], []
    for (H, W), d in by_grid.items():
        if d["box"] is None or d["cls"] is None:
            continue
        box = d["box"].reshape(H * W, 4, reg_max)             # [N,4,16]
        cls = d["cls"].reshape(H * W, -1)                     # [N,nc]
        stride = iw // W

        dist = (_softmax(box, axis=2) * np.arange(reg_max)).sum(axis=2)  # [N,4] ltrb
        gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        ax = gx.reshape(-1) + 0.5
        ay = gy.reshape(-1) + 0.5
        l, t, r, b = dist[:, 0], dist[:, 1], dist[:, 2], dist[:, 3]
        x1 = (ax - l) * stride; y1 = (ay - t) * stride
        x2 = (ax + r) * stride; y2 = (ay + b) * stride

        prob = _sigmoid(cls)
        cid = prob.argmax(1)
        sc = prob[np.arange(len(prob)), cid]

        keep = sc > conf_thres
        if not keep.any():
            continue
        x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
        all_boxes.append(np.stack([x1, y1, x2 - x1, y2 - y1], axis=1))
        all_scores.append(sc[keep])
        all_cids.append(cid[keep])

    if not all_boxes:
        return [], [], []
    rects = np.concatenate(all_boxes).astype(int).tolist()
    scores = np.concatenate(all_scores).tolist()
    cids = np.concatenate(all_cids).tolist()
    return _nms(rects, scores, cids, conf_thres, iou_thres)


# --------------------------------------------------------------------------- #
# Decoder 2: single decoded output [1, 4+nc, N] / [1, N, 4+nc]
# --------------------------------------------------------------------------- #
def decode_single(interp, outs, conf_thres, iou_thres, input_wh):
    od = outs[0]
    o = _dequant(interp.get_tensor(od["index"]), od)
    o = np.squeeze(o)
    if o.shape[0] < o.shape[1]:
        o = o.transpose(1, 0)                 # -> [N, 4+nc]
    xywh = o[:, :4]
    cls = o[:, 4:]
    cid = np.argmax(cls, axis=1)
    sc = cls[np.arange(len(cls)), cid]
    keep = sc > conf_thres
    xywh, sc, cid = xywh[keep], sc[keep], cid[keep]
    if len(xywh) == 0:
        return [], [], []
    iw, ih = input_wh
    if xywh.max() <= 1.5:
        xywh = xywh * np.array([iw, ih, iw, ih], dtype=np.float32)
    tl = xywh[:, :2] - xywh[:, 2:4] / 2
    rects = np.concatenate([tl, xywh[:, 2:4]], axis=1).astype(int).tolist()
    return _nms(rects, sc.tolist(), cid.tolist(), conf_thres, iou_thres)


def _nms(rects, scores, cids, conf_thres, iou_thres):
    idxs = cv2.dnn.NMSBoxes(rects, scores, conf_thres, iou_thres)
    if len(idxs) == 0:
        return [], [], []
    idxs = np.array(idxs).flatten()
    return [rects[i] for i in idxs], [scores[i] for i in idxs], [cids[i] for i in idxs]


# --------------------------------------------------------------------------- #
# Dispatcher + drawing
# --------------------------------------------------------------------------- #
def decode(interp, outs, conf_thres, iou_thres, input_wh, nc=80):
    """Pick the right decoder based on the number of output tensors."""
    if len(outs) >= 6:
        return decode_raw_dfl(interp, outs, conf_thres, iou_thres, input_wh, nc=nc)
    return decode_single(interp, outs, conf_thres, iou_thres, input_wh)


def color_for(cid):
    np.random.seed(cid * 7 + 1)
    return tuple(int(c) for c in np.random.randint(60, 255, 3))


def draw(frame, rects, scores, cids, labels, ratio, pad):
    lx, ty = pad
    for (x, y, w, h), s, cid in zip(rects, scores, cids):
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
