# Runbook — run YOLO on the DART-MX8M-PLUS NPU

Operational guide to go from nothing to live object detection on the board's display.

## Prerequisites

- DART-MX8M-PLUS powered on, with the **LVDS panel** connected (Weston running).
- **Logitech C920** (or any UVC webcam) plugged into the board's USB.
- Board on the network (**ethernet / eth1**) — it needs internet to clone + download the model.
- A shell on the board: either a USB keyboard, SSH, or the serial console
  (see [`docs/serial_console.md`](docs/serial_console.md)).

Everything the runtime needs (OpenCV 4.6 Qt5, `tflite_runtime` 2.16.2,
`libvx_delegate.so`) already ships in the Variscite BSP — nothing to `pip install`.

## The flow (3 steps)

```bash
# 1. Clone the repo on the board
git clone https://github.com/RicardoSITJ/yolo_dart.git
cd yolo_dart

# 2. Verify the environment and fetch the model
make setup

# 3. Start live detection on the display
make streaming
```

### What each step does

**`make setup`** runs `scripts/setup_board.sh` and prints a 4-point check. Expected:

```
== 1/4  Python deps ==        ok OpenCV 4.6.0 / NumPy / TFLite runtime 2.16.2
== 2/4  NPU delegate ==       ok /usr/lib/libvx_delegate.so, /dev/galcore, delegate loads
== 3/4  Camera (C920) ==      ok webcam at /dev/video3
== 4/4  Model ==              downloaded models/yolov5n-det-int8.tflite (2.8M) + sample
```

If any line is not `ok`, fix it before step 3 (see Troubleshooting).

**`make streaming`** launches `src/realtime_detect.py` on the NPU. The **first run
compiles the NPU graph (~15–60 s)** with no output; after that a window with the live
camera, bounding boxes, labels and an FPS counter appears **on the LVDS panel**
(~17 FPS, ~58 ms/frame). The compiled graph is cached (`~/.cache/yolo_dart/npu`), so
the next launch starts in ~1 s.

> The window is routed to Weston automatically — the script finds the Wayland socket,
> so this works even when launched from a serial/SSH session with no graphical login.

## Stop

- In the window: press **`q`**.
- From another shell / serial: `pkill -f realtime_detect`

## Variants

```bash
make detect                 # single-image sanity check (downloads a sample)
make benchmark              # NPU latency + FPS (reproducible, no camera)
make cpu                    # NPU vs CPU A/B
make test                   # decoder unit tests
make streaming CAMERA=0     # different webcam index
make streaming CONF=0.5     # higher confidence threshold
```

Headless (no display, writes `outputs/latest.jpg` each frame — useful over SSH):

```bash
python3 src/realtime_detect.py --delegate /usr/lib/libvx_delegate.so --headless
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `make setup` cam check warns | Webcam not enumerated. Replug it; `v4l2-ctl --list-devices` should list the C920. Then set `CAMERA=<n>` to its `/dev/video<n>`. |
| Window never appears | Weston not running or wrong socket. Check `ls /run/user/0/wayland-*`; try `QT_QPA_PLATFORM=xcb make streaming` (XWayland fallback). |
| First run hangs ~30–60 s | Normal: NPU graph compile. It is cached afterwards. |
| `cannot read image: assets/sample.jpg` | Only affects `make detect`; it downloads the sample — rerun with internet up. |
| Model download fails | No internet on the board. Check `ip -4 addr show eth1` and `ping -c1 8.8.8.8`. |
| Boxes look wrong / none | Tune `CONF`/`IOU`; confirm the camera actually shows COCO-class objects (person, bottle, chair…). |
| Slow (~300 ms) not ~58 ms | Model fell back to CPU. Ensure `--delegate /usr/lib/libvx_delegate.so` (the default) and that it loaded in the log. |
| Serial console "Resource busy" | Another `screen`/program holds the port: `pkill -f "screen.*usbserial"`. |

## Notes

- The YOLO model is **not** in the repo (AGPL-3.0); `make setup`/`make streaming`
  download it. See [`docs/models.md`](docs/models.md).
- Board facts and the full inventory: [`docs/hardware.md`](docs/hardware.md).
- Pending enhancements (e.g. systemd auto-start): [`docs/improvements.md`](docs/improvements.md).
