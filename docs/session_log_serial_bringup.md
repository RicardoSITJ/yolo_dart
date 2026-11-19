# Session Log — Serial Bring-up & Live Streaming

> Transcript of a hands-on session driving the Variscite DART-MX8M-PLUS board
> entirely over the USB-serial console from a host Mac: connect, clone the repo,
> run `make setup`, start live NPU detection on the LVDS panel, and stop it.

**Date:** 2026-11-19
**Board:** Variscite DART-MX8M-PLUS v1.1 (i.MX 8M Plus), Debian 12 bookworm, kernel `6.6.119-fslc-imx` aarch64
**Host:** macOS, connected via FTDI FT230X at `/dev/cu.usbserial-DK0JTWFN` @ 115200 8N1
**Console:** `ttymxc0`, login `root` (no password)
**Helper:** `tools/dart_console.py` (pyserial auto-login; feeds shell commands on stdin — see [serial_console.md](serial_console.md))

---

## 1. Connect over serial

The board was mid-boot when the session started, so the first connection captured
the full U-Boot + kernel log (reset cause: POR). After it settled at the login
prompt, a clean auto-login landed at the shell:

```
root@imx8mp-var-dart:~#
```

Sanity check:

```console
$ hostname
imx8mp-var-dart
$ uname -r
6.6.119-fslc-imx-gac11a677c871
$ uptime
 up 3 min,  2 users,  load average: 0.25, 0.39, 0.18
```

> **Note:** the serial port is exclusive — do not keep `screen`/`minicom` open
> while scripting against the console.

---

## 2. Clone the repository

```console
$ cd ~ && rm -rf yolo_dart && git clone https://github.com/RicardoSITJ/yolo_dart.git
Cloning into 'yolo_dart'...
Receiving objects: 100% (105/105), 53.79 KiB | 153.00 KiB/s, done.
Resolving deltas: 100% (44/44), done.
$ cd yolo_dart && ls
LICENSE  Makefile  README.md  RUNBOOK.md  docs  models  requirements.txt  scripts  src  tests  tools
```

Repo lands at `/root/yolo_dart`.

---

## 3. `make setup` — check NPU + camera, download the model

```console
$ make setup
./scripts/setup_board.sh
== 1/4  Python deps (all expected from BSP) ==
  ok   OpenCV         4.6.0
  ok   NumPy          1.24.2
  ok   TFLite runtime 2.16.2
  [ok]   python modules present
== 2/4  NPU delegate ==
  [ok]   /usr/lib/libvx_delegate.so
  [ok]   /dev/galcore (NPU driver)
  ok   delegate loads
== 3/4  Camera (C920) ==
  [ok]   webcam at /dev/video3
== 4/4  Model (EdgeFirst YOLOv5n INT8, built for the i.MX 8M Plus NPU) ==
  downloading pretrained INT8 model from Hugging Face (EdgeFirst)...
  [ok]   downloaded models/yolov5n-det-int8.tflite (2.8M)
  [ok]   downloaded assets/sample.jpg

Setup check complete.
```

All four checks pass: Python deps, NPU delegate, camera, and model download.

---

## 4. `make streaming` — live detection on the LVDS display

`make streaming` runs `src/realtime_detect.py` (camera → NPU → boxes on the
LVDS/Weston display). Because it runs indefinitely and draws to the physical
panel, it was launched **detached** so the GUI window shows on the display
without holding the serial port:

```console
$ nohup make streaming > /tmp/streaming.log 2>&1 &
[1] 2072
```

Log after NPU cold-start:

```
python3 src/realtime_detect.py --model models/yolov5n-det-int8.tflite \
    --labels models/coco_labels.txt --camera 3 --conf 0.35 --iou 0.45 \
    --delegate /usr/lib/libvx_delegate.so
INFO: Vx delegate: allowed_cache_mode set to 0.
INFO: Vx delegate: device num set to 0.
...
[ WARN:0] OpenCV | GStreamer warning: Cannot query video position: ... (benign for a live webcam)
```

Process confirmed alive (`pgrep -af realtime_detect` → PID 2074), delegate
initialized, detection window drawing on the LVDS panel. Expected NPU latency
~58 ms (~17 FPS).

> The GStreamer `Cannot query video position` warning is benign — it is normal
> for a live webcam source that has no seekable duration.

---

## 5. Stop streaming

```console
$ pkill -f realtime_detect
[1]+  Exit 2   nohup make streaming > /tmp/streaming.log 2>&1
$ pgrep -af realtime_detect || echo "no process running"
no process running
```

Streaming stopped; the LVDS window closed and the serial port is free again.

---

## One-liner recap

```bash
git clone https://github.com/RicardoSITJ/yolo_dart.git
cd yolo_dart
make setup        # check NPU + camera, download the model
make streaming    # live detection on the LVDS display  (stop with: pkill -f realtime_detect)
```
