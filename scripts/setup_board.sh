#!/usr/bin/env bash
# Reproducible setup for yolo_dart on the DART-MX8M-PLUS (Debian 12, i.MX 8M Plus).
#
# The NPU stack (tflite_runtime, libvx_delegate.so), OpenCV 4.6 (Qt5 GUI) and the
# C920 driver are all provided by the Variscite BSP, so this script does NOT
# reinstall them (doing so would break NPU offload / the GUI build). It only
# verifies the environment is ready and the model is present.
#
# Usage (on the board):
#   git clone https://github.com/RicardoSITJ/yolo_dart.git
#   cd yolo_dart
#   ./scripts/setup_board.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ok()   { printf "  \033[32m[ok]\033[0m   %s\n" "$1"; }
warn() { printf "  \033[33m[warn]\033[0m %s\n" "$1"; }
die()  { printf "  \033[31m[FAIL]\033[0m %s\n" "$1"; exit 1; }

echo "== 1/4  Python deps (all expected from BSP) =="
python3 - <<'PY' || die "missing Python modules — see above"
import importlib, sys
need = {"cv2": "OpenCV", "numpy": "NumPy", "tflite_runtime": "TFLite runtime"}
missing = []
for mod, name in need.items():
    try:
        m = importlib.import_module(mod)
        print(f"  ok   {name:14s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  MISS {name:14s} {e}")
        missing.append(mod)
sys.exit(1 if missing else 0)
PY
ok "python modules present"

echo "== 2/4  NPU delegate =="
DELEGATE=/usr/lib/libvx_delegate.so
[ -f "$DELEGATE" ] && ok "$DELEGATE" || die "VX delegate missing: $DELEGATE"
[ -e /dev/galcore ] && ok "/dev/galcore (NPU driver)" || warn "/dev/galcore not found"
python3 - <<PY || die "delegate failed to load"
from tflite_runtime.interpreter import load_delegate
load_delegate("$DELEGATE"); print("  ok   delegate loads")
PY

echo "== 3/4  Camera (C920) =="
CAM=""
for d in /dev/video3 /dev/video0 /dev/video1 /dev/video2; do
  if v4l2-ctl -d "$d" --all 2>/dev/null | grep -qi "C920\|UVC\|Webcam"; then CAM="$d"; break; fi
done
[ -n "$CAM" ] && ok "webcam at $CAM" || warn "C920 not detected on /dev/video* (plug it in?)"

echo "== 4/4  Model (EdgeFirst YOLOv5n INT8, built for the i.MX 8M Plus NPU) =="
MODEL=models/yolov5n-det-int8.tflite
MODEL_URL="https://huggingface.co/EdgeFirst/yolov5-det/resolve/main/tflite/yolov5n-det-int8-smart.tflite"
if [ -f "$MODEL" ]; then
  ok "$MODEL ($(du -h "$MODEL" | cut -f1))"
else
  echo "  downloading pretrained INT8 model from Hugging Face (EdgeFirst)..."
  if wget -q --show-progress -O "$MODEL" "$MODEL_URL"; then
    ok "downloaded $MODEL ($(du -h "$MODEL" | cut -f1))"
  else
    rm -f "$MODEL"
    die "model download failed — check the board's internet connection"
  fi
fi

echo "  sample image..."
SAMPLE=assets/sample.jpg
mkdir -p assets
if [ -f "$SAMPLE" ]; then
  ok "$SAMPLE"
else
  if wget -q -O "$SAMPLE" "https://ultralytics.com/images/bus.jpg"; then
    ok "downloaded $SAMPLE"
  else
    rm -f "$SAMPLE"; warn "sample image download failed (optional)"
  fi
fi

echo
echo "Setup check complete. To run (shows window on the LVDS/Weston display):"
echo "  python3 src/realtime_detect.py --delegate /usr/lib/libvx_delegate.so"
echo "Static sanity check instead:  python3 src/detect.py --source assets/sample.jpg"
