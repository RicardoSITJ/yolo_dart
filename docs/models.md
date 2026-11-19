# Models

## Chosen model: EdgeFirst YOLOv5n INT8 (built for the i.MX 8M Plus NPU)

| Field        | Value                                                                 |
|--------------|-----------------------------------------------------------------------|
| Name         | `yolov5n-det-int8-smart.tflite` (stored locally as `yolov5n-det-int8.tflite`) |
| Source       | https://huggingface.co/EdgeFirst/yolov5-det (`tflite/` dir)           |
| Download     | https://huggingface.co/EdgeFirst/yolov5-det/resolve/main/tflite/yolov5n-det-int8-smart.tflite |
| Size         | ~2.8 MB                                                               |
| Classes      | COCO 2017, 80 classes (`models/coco_labels.txt`)                     |
| Input        | `[1, 640, 640, 3]` **uint8** (quant scale `1/255`, zero-point 0 → raw RGB) |
| License      | **AGPL-3.0** (Ultralytics YOLOv5 lineage)                            |

> ⚠️ **Why it is not committed to this repo.** This repo is Apache-2.0; the model
> is AGPL-3.0. To keep the licenses clean the model is **downloaded at runtime**
> by `scripts/setup_board.sh` (the board has internet), never checked in.

Picked because EdgeFirst publishes these INT8 TFLite exports **explicitly for the
i.MX 8M Plus NPU via the VeriSilicon VX delegate** — no host-side export/quantization
step is needed. NXP's own [eiq-model-zoo](https://github.com/NXP/eiq-model-zoo) is a
good alternative (e.g. yolov4-tiny, efficientdet-lite0), but yolov4-tiny is much
slower on this NPU (~300 ms) due to unsupported ops.

## Output format — raw decoupled DFL head (6 tensors)

The "smart" export **strips the in-graph decode** (DFL + concat) so the NPU only runs
the conv backbone + head; the decode is done on the CPU in `src/yolo_decode.py`. The
6 output tensors (verified on-device) are, per detection scale:

| Grid   | Stride | Box tensor `[1,H,W,64]` | Class tensor `[1,H,W,80]` |
|--------|--------|-------------------------|---------------------------|
| 80×80  | 8      | `add_68` int8           | `add_71` int8             |
| 40×40  | 16     | `add_82` int8           | `add_83` int8             |
| 20×20  | 32     | `add_91` int8           | `add_92` int8             |

- **Box (64 ch)** = 4 sides × `reg_max`(16) DFL bins. Decode: dequantize → softmax
  over the 16 bins → expected value `[0..15]` → `ltrb` distances → `(anchor ± dist) ×
  stride` → xyxy in the 640×640 input space.
- **Class (80 ch)** = per-class logits → dequantize → sigmoid.
- All scales are concatenated, thresholded (`--conf`), and run through
  `cv2.dnn.NMSBoxes` (`--iou`).

`src/yolo_decode.py` auto-selects this decoder when the model has ≥6 outputs, and
falls back to a single-tensor `[1, 4+nc, 8400]` decoder for standard Ultralytics
exports.

## Verified on-device (DART-MX8M-PLUS, 2026-11-19)

Static image sanity check on the NPU:

```
$ python3 src/detect.py --source cat.jpg --delegate /usr/lib/libvx_delegate.so
[npu] loading VX delegate: /usr/lib/libvx_delegate.so
[NPU] inference 58.7 ms | 1 detections:
  cat   0.70   box=(183,66,301,511)
[out] wrote outputs/cat_det.jpg
```

- **~58.7 ms/frame on the NPU** after the first-run graph compile (~30–60 s warmup).
- Live camera loop (`realtime_detect.py`) validated end-to-end against the C920.

## Swapping the model

Any of these work with the same code:

- **Other EdgeFirst sizes** (same raw-head decoder): change the URL to
  `yolov5s-det-int8-smart.tflite`, etc. in `scripts/setup_board.sh`.
- **Standard Ultralytics INT8** (`*_full_integer_quant.tflite`, single decoded
  output): drop it in `models/` and point `--model` at it — `yolo_decode.decode()`
  detects the single-output head automatically. Export with
  `scripts/export_tflite.sh` on a host PC.

The input tensor's dtype and quant params are read from the model at load time, so
uint8 and int8 inputs are both handled.
