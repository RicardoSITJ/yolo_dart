# yolo_dart — one-command workflows for the DART-MX8M-PLUS.
#
#   git clone https://github.com/RicardoSITJ/yolo_dart.git && cd yolo_dart
#   make streaming        # download model (if missing) + live NPU detection on the display
#
# Override any knob on the command line, e.g.  make streaming CAMERA=0 CONF=0.5

DELEGATE ?= /usr/lib/libvx_delegate.so
MODEL    ?= models/yolov5n-det-int8.tflite
LABELS   ?= models/coco_labels.txt
CAMERA   ?= 3
CONF     ?= 0.35
IOU      ?= 0.45
MODEL_URL = https://huggingface.co/EdgeFirst/yolov5-det/resolve/main/tflite/yolov5n-det-int8-smart.tflite
SAMPLE     = assets/sample.jpg
SAMPLE_URL = https://ultralytics.com/images/bus.jpg
PY        = python3

.PHONY: help streaming run setup detect benchmark cpu test clean model

help:
	@echo "yolo_dart targets:"
	@echo "  make streaming   live camera -> NPU -> boxes on the LVDS/Weston display"
	@echo "  make detect      single-image NPU sanity check (assets/sample.jpg)"
	@echo "  make benchmark   NPU latency + FPS"
	@echo "  make cpu         NPU vs CPU A/B benchmark"
	@echo "  make setup       verify board env (NPU, camera) + download model"
	@echo "  make test        decoder unit tests"
	@echo "  make clean       remove outputs/ and __pycache__"
	@echo "Knobs: CAMERA=$(CAMERA) CONF=$(CONF) IOU=$(IOU) DELEGATE=$(DELEGATE)"

# Download the pretrained INT8 model on demand (so 'make streaming' just works).
$(MODEL):
	@mkdir -p models
	@echo ">> downloading model..."
	@wget -q --show-progress -O $(MODEL) "$(MODEL_URL)"
model: $(MODEL)

$(SAMPLE):
	@mkdir -p assets
	@echo ">> downloading sample image..."
	@wget -q -O $(SAMPLE) "$(SAMPLE_URL)"

# Flagship: start streaming with one command.
streaming run: $(MODEL)
	$(PY) src/realtime_detect.py --model $(MODEL) --labels $(LABELS) \
		--camera $(CAMERA) --conf $(CONF) --iou $(IOU) --delegate $(DELEGATE)

setup:
	./scripts/setup_board.sh

detect: $(MODEL) $(SAMPLE)
	$(PY) src/detect.py --model $(MODEL) --labels $(LABELS) --source $(SAMPLE) \
		--delegate $(DELEGATE) --conf $(CONF) --iou $(IOU)

benchmark: $(MODEL)
	$(PY) src/benchmark.py --model $(MODEL) --delegate $(DELEGATE)

cpu: $(MODEL)
	$(PY) src/benchmark.py --model $(MODEL) --delegate '' --cpu

test:
	$(PY) tests/test_yolo_decode.py

clean:
	rm -rf outputs __pycache__ src/__pycache__ tests/__pycache__
