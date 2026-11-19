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

## 2. Doc nit: `cat.jpg` sample image (pending)

`docs/models.md` / the README sanity check reference `cat.jpg`, which lives in the
board's `/root` (a leftover from earlier experiments) and is **not** in the repo. A
fresh clone has no `cat.jpg`. Options:
- commit a small sample image under `assets/` and point the examples at it, or
- change the examples to `--source <your_image.jpg>` and make `detect.py` print a
  clear message when the source is missing.

## 3. Nice-to-haves (not started)

- Benchmark sweep across input sizes / camera resolutions to find the best
  latency/quality trade-off.
- Confidence/NMS auto-tuning per class.
- Optional GStreamer `waylandsink` zero-copy path for higher sustained FPS than
  `cv2.imshow`.
- Clear the NPU graph cache automatically when the model file changes (today the
  cache dir just accumulates entries keyed by graph).
