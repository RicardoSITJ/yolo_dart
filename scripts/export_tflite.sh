#!/usr/bin/env bash
# Export + full-integer (INT8) quantize a YOLO model to TFLite for the i.MX 8M Plus NPU.
# Run this on a HOST PC (needs ultralytics + tensorflow), NOT on the DART.
#
# Usage: ./scripts/export_tflite.sh yolov8n
set -euo pipefail

MODEL="${1:-yolov8n}"
OUT_DIR="models"
mkdir -p "$OUT_DIR"

echo ">> Exporting ${MODEL} to INT8 TFLite (full integer quantization)"
# Ultralytics handles export + INT8 quantization. int8=True requires a
# representative dataset; it defaults to a COCO sample if none is given.
yolo export model="${MODEL}.pt" format=tflite int8=True imgsz=640

# Ultralytics writes to ${MODEL}_saved_model/${MODEL}_full_integer_quant.tflite
SRC=$(find . -name "*_full_integer_quant.tflite" | head -1)
if [[ -z "${SRC}" ]]; then
  echo "!! Could not find the INT8 tflite output" >&2
  exit 1
fi

cp "${SRC}" "${OUT_DIR}/${MODEL}_int8.tflite"
echo ">> Wrote ${OUT_DIR}/${MODEL}_int8.tflite"
echo ">> Copy it to the DART and run: python3 src/detect.py --model ${OUT_DIR}/${MODEL}_int8.tflite --source test.jpg --delegate /usr/lib/libvx_delegate.so"
