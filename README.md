# yolo_dart

YOLO object detection running on the **NPU** of the [Variscite **DART-MX8M-PLUS**](https://www.variscite.com/product/system-on-module-som/cortex-a53-krait/dart-mx8m-plus-nxp-i-mx-8m-plus/), based on the **NXP i.MX 8M Plus**.

The i.MX 8M Plus integrates a **Verisilicon VIP8000 NPU (~2.3 TOPS)**. This repo runs a quantized (INT8) YOLO model on that NPU using **TensorFlow Lite** with NXP's **VX external delegate** (`libvx_delegate.so`), captures from a USB webcam and draws live detections on the board's LVDS display via OpenCV.

> ✅ **Working on hardware (2026-11-19):** EdgeFirst YOLOv5n INT8 runs on the NPU at
> **~59 ms/frame**; validated on `cat.jpg` (`cat 0.70`) and live against a Logitech
> C920. See [`docs/models.md`](docs/models.md) and [`docs/hardware.md`](docs/hardware.md).

---

## Why INT8 + VX delegate?

The VIP8000 NPU only accelerates **8-bit integer** operations. A float model will silently fall back to CPU. The pipeline is therefore:

```
YOLO (.pt / .onnx)  ──►  TFLite float32  ──►  TFLite INT8 (full integer quant)  ──►  NPU via vx_delegate
```

On **first run**, the VX delegate compiles the graph into an NPU warmup cache (`*.nb` under `/tmp`), which can take 30–60 s. Subsequent runs load the cache and are fast.

## Hardware / BSP target

| Item            | Value                                                    |
|-----------------|---------------------------------------------------------|
| SoM             | Variscite DART-MX8M-PLUS                                 |
| SoC             | NXP i.MX 8M Plus (4× Cortex-A53)                         |
| NPU             | Verisilicon VIP8000, ~2.3 TOPS, INT8                     |
| OS              | Debian 12 (bookworm), kernel 6.6.119-fslc-imx, aarch64   |
| Camera          | Logitech C920 (UVC) on `/dev/video3`                     |
| Display         | LVDS panel + Weston/Wayland                              |
| Runtime         | TFLite `tflite_runtime` 2.16.2 + `libvx_delegate.so`, OpenCV 4.6 (Qt5) — all from BSP |

## Repo layout

```
yolo_dart/
├── models/
│   └── coco_labels.txt        # 80 COCO classes (the .tflite is downloaded by setup)
├── src/
│   ├── realtime_detect.py     # live camera -> NPU -> boxes on the LVDS display
│   ├── detect.py              # single-image NPU sanity check
│   └── yolo_decode.py         # DFL decode (raw head) + NMS, shared by both
├── scripts/
│   ├── setup_board.sh         # on-board env check + model download
│   └── export_tflite.sh       # (host) export a custom YOLO to INT8 TFLite
└── docs/
    ├── hardware.md            # verified board + host inventory
    ├── models.md              # model choice, license, output decode
    └── npu_setup.md           # BSP / delegate setup notes
```

## Quick start (on the DART)

Everything the runtime needs — OpenCV 4.6 (Qt5 GUI), `tflite_runtime` 2.16.2 and
`libvx_delegate.so` — already ships in the Variscite BSP, so there is nothing to
`pip install`. See [`docs/hardware.md`](docs/hardware.md) for the verified board state.

```bash
# 1. Clone on the board (it has internet over eth1)
git clone https://github.com/RicardoSITJ/yolo_dart.git
cd yolo_dart

# 2. Verify the environment (checks NPU delegate, camera, model)
./scripts/setup_board.sh

# 3. Real-time detection on the NPU, shown on the LVDS/Weston display.
#    The window is routed to Weston automatically (the script finds the Wayland
#    socket), so this works straight from a serial/ssh session.
python3 src/realtime_detect.py --delegate /usr/lib/libvx_delegate.so

# CPU vs NPU A/B: drop --delegate to run the same model on the A53 cores.
# Headless (no display, writes outputs/latest.jpg): add --headless
```

Defaults: `--model models/yolov5n-det-int8.tflite`, `--labels models/coco_labels.txt`,
`--camera 3`. If the window does not appear, set `QT_QPA_PLATFORM=xcb` to fall back to
XWayland.

Single-image sanity check instead of live video:

```bash
python3 src/detect.py --source cat.jpg \
    --model models/yolov5n-det-int8.tflite --delegate /usr/lib/libvx_delegate.so
```

**First run compiles the NPU graph (~30–60 s warmup)** before the first result; it is
fast afterwards. The model is fetched automatically by `setup_board.sh` — see
[`docs/models.md`](docs/models.md) for the model, its license, and the output decode.

## Exporting a model (on a host PC)

```bash
./scripts/export_tflite.sh yolov8n   # produces models/yolov8n_int8.tflite
```

## Status

Working on hardware:
- [x] Pretrained INT8 YOLO running on the VIP8000 NPU via `vx_delegate` (~59 ms/frame)
- [x] DFL decode for the EdgeFirst raw-head model + NMS
- [x] Single-image validation (`cat.jpg` → `cat 0.70`)
- [x] Live C920 capture → NPU → annotated frames
- [x] On-display output via OpenCV/Qt5 on the LVDS/Weston panel

Next:
- [ ] Benchmark NPU vs CPU and measure sustained live FPS
- [ ] Tune confidence/NMS and camera resolution for best latency

## License

[Apache-2.0](LICENSE)
