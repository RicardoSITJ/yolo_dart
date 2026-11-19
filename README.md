# yolo_dart

YOLO object detection running on the **NPU** of the [Variscite **DART-MX8M-PLUS**](https://www.variscite.com/product/system-on-module-som/cortex-a53-krait/dart-mx8m-plus-nxp-i-mx-8m-plus/), based on the **NXP i.MX 8M Plus**.

The i.MX 8M Plus integrates a **Verisilicon VIP8000 NPU (~2.3 TOPS)**. This repo runs a quantized (INT8) YOLO model on that NPU using **TensorFlow Lite** with NXP's **VX external delegate** (`libvx_delegate.so`), part of the eIQ / NXP BSP.

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
| OS              | Yocto BSP (Variscite) / Linux                            |
| Runtime         | TensorFlow Lite + `libvx_delegate.so` (eIQ)             |

## Repo layout

```
yolo_dart/
├── models/           # .tflite models (INT8). Large files tracked out of git (see .gitignore)
├── src/
│   ├── detect.py     # run inference on an image/stream using the NPU
│   └── benchmark.py  # measure NPU vs CPU latency
├── scripts/
│   └── export_tflite.sh   # export + quantize a YOLO model to INT8 TFLite
├── requirements.txt  # host (export) and target (runtime) deps
└── docs/
    └── npu_setup.md  # BSP / delegate setup notes on the DART
```

## Quick start (on the DART)

```bash
# 1. Clone on the board
git clone https://github.com/RicardoSITJ/yolo_dart.git
cd yolo_dart

# 2. Install runtime deps (tflite-runtime is usually already in the BSP)
pip3 install -r requirements.txt

# 3. Run detection on the NPU
python3 src/detect.py \
    --model models/yolov8n_int8.tflite \
    --source test.jpg \
    --delegate /usr/lib/libvx_delegate.so
```

If `libvx_delegate.so` is missing, it is provided by the `tensorflow-lite-vx-delegate` package in the NXP/Variscite BSP.

## Exporting a model (on a host PC)

```bash
./scripts/export_tflite.sh yolov8n   # produces models/yolov8n_int8.tflite
```

## Status

🚧 Early scaffold. Next steps:
- [ ] Add a working INT8 TFLite export for YOLOv8n
- [ ] Validate NPU delegation (confirm ops run on VIP8000, not CPU)
- [ ] Camera / GStreamer pipeline for live detection
- [ ] Benchmark NPU vs CPU

## License

[Apache-2.0](LICENSE)
