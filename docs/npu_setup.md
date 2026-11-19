# NPU setup on the DART-MX8M-PLUS

Notes for getting TensorFlow Lite to offload to the Verisilicon VIP8000 NPU on the
NXP i.MX 8M Plus (Variscite BSP).

## 1. Confirm the delegate exists

```bash
find / -name "libvx_delegate.so" 2>/dev/null
# typically: /usr/lib/libvx_delegate.so
```

Provided by `tensorflow-lite-vx-delegate` in the NXP/Variscite Yocto BSP. If missing,
rebuild the image with that package or install it from the BSP feeds.

## 2. Confirm the NPU / galcore driver is up

```bash
dmesg | grep -i galcore        # VeriSilicon GPU/NPU kernel driver
ls /dev/galcore                # device node should exist
cat /sys/kernel/debug/gc/info  # NPU info (if debugfs mounted)
```

## 3. First run = graph compilation

The first inference with the VX delegate compiles the network into an NPU binary
(`.nb`). This is slow (30–60 s). To cache it between runs:

```bash
export VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1
export VIV_VX_CACHE_BINARY_GRAPH_DIR=/home/root/.npu_cache
mkdir -p "$VIV_VX_CACHE_BINARY_GRAPH_DIR"
```

## 4. Verify ops actually run on the NPU (not CPU fallback)

```bash
export VIV_VX_DEBUG_LEVEL=1     # delegate logs which ops it takes
python3 src/detect.py --model models/yolov8n_int8.tflite --source test.jpg \
    --delegate /usr/lib/libvx_delegate.so
```

Look for the delegate reporting the number of nodes it accelerated. Float ops and
unsupported ops fall back to CPU — if most of the graph falls back, re-check that the
model is **full-integer INT8** quantized.

## References

- NXP i.MX Machine Learning User's Guide (eIQ)
- Variscite DART-MX8M-PLUS Yocto BSP documentation
