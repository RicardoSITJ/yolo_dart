# Possible improvements / pending

Things that are **not** required for the working, reproducible pipeline but would
make it nicer. Tracked here instead of being implemented now.

## 1. systemd service for auto-start on boot (pending)

Run the live detector automatically when the DART powers on, so no shell is needed.

Sketch — `/etc/systemd/system/yolo-dart.service`:

```ini
[Unit]
Description=yolo_dart live detection
After=weston.service
Wants=weston.service

[Service]
User=root
Environment=XDG_RUNTIME_DIR=/run/user/0
WorkingDirectory=/root/yolo_dart
ExecStart=/usr/bin/python3 src/realtime_detect.py --delegate /usr/lib/libvx_delegate.so
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Open questions before adding it: should it start only after the LVDS/Weston session
is up (ordering/race), and do we want it enabled by default or opt-in via a flag in
`setup_board.sh`?

## 2. Sample image — RESOLVED ✅

Earlier the examples referenced `cat.jpg`, which only existed on the board's `/root`
and was absent from a fresh clone. Fixed: `setup_board.sh` now downloads a sample to
`assets/sample.jpg` (gitignored, same pattern as the model) and `src/detect.py`
defaults `--source` to it. The sanity check works on a clean clone.

## 3. Nice-to-haves (not started)

- Benchmark sweep across input sizes / camera resolutions to find the best
  latency/quality trade-off.
- Confidence/NMS auto-tuning per class.
- Optional GStreamer `waylandsink` zero-copy path for higher sustained FPS than
  `cv2.imshow`.
- Clear the NPU graph cache automatically when the model file changes (today the
  cache dir just accumulates entries keyed by graph).
