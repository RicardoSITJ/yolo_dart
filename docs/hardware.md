# Hardware setup

Bench setup for running YOLO object detection on the NPU of the **DART-MX8M-PLUS**
(NXP i.MX 8M Plus). The DART is the compute target; the Mac is used only as a serial
console / development host.

> **Verification.** Host facts were read from the Mac (`system_profiler`, `ifconfig`).
> Target facts were read **directly from the board** over the serial console on
> 2026-11-19 (`uname`, `lsusb`, `v4l2-ctl`, `ip`, `dpkg`). Everything below marked ✅
> was observed first-hand.

## Topology

```
                 USB (serial console, FTDI FT230X)
   ┌──────────┐  /dev/cu.usbserial-DK0JTWFN @115200   ┌────────────────────────┐
   │  MacBook │ ◄───────────────────────────────────► │  DART-MX8M-PLUS (SoM)  │
   │  Pro M3  │        ttymxc0, root (no password)     │  NXP i.MX 8M Plus      │
   └──────────┘                                        │  + VIP8000 NPU         │
                                                       └──────────┬─────────────┘
                                                                  │ on the DART:
                                                     ┌────────────┼───────────────┐
                                                  USB C920      eth1            (eth0 down)
                                                /dev/video3   192.168.0.169/24
```

The only physical link between Mac and DART is the USB-serial console. The keyboard,
camera, and ethernet hang off the DART, not the Mac (confirmed: the Mac's only USB
device is the FTDI adapter).

## Host (Mac) — verified ✅

| Spec           | Value                                       |
|----------------|---------------------------------------------|
| Model          | MacBook Pro (`Mac15,3`)                      |
| Chip           | Apple **M3**, 8 cores (4P + 4E)             |
| Memory         | 16 GB unified                               |
| macOS          | 14.2 (build 23C64)                         |
| Active network | `en0` (Wi-Fi), `192.168.10.87/23`           |
| USB device     | FTDI **FT230X** (VID `0403`/PID `6015`, SN `DK0JTWFN`) → `/dev/cu.usbserial-DK0JTWFN` |
| Role           | Serial console + model-export host          |

## Target (DART-MX8M-PLUS) — verified ✅

| Component       | Value                                                            |
|-----------------|------------------------------------------------------------------|
| SoM / SoC       | Variscite DART-MX8M-PLUS, NXP i.MX 8M Plus                        |
| Hostname        | `imx8mp-var-dart`                                                 |
| OS              | **Debian 12 (bookworm)** — *not Yocto*                            |
| Kernel          | `6.6.119-fslc-imx` (NXP FSLC), aarch64                           |
| CPU             | 4× Cortex-A53                                                     |
| RAM             | 3.5 GB (≈3.2 GB free at idle)                                    |
| NPU             | Verisilicon VIP8000 (~2.3 TOPS, INT8), driver `/dev/galcore` ✅  |
| Console         | `ttymxc0`, login **root**, **no password**                       |

### Camera — verified ✅
| Field      | Value                                              |
|------------|----------------------------------------------------|
| Model      | **Logitech HD Pro Webcam C920** (UVC 1.00)         |
| USB ID     | `046d:0892`                                        |
| V4L2 node  | `/dev/video3` (capture) — driver `uvcvideo`        |
| Formats    | YUYV (raw) + MJPG (for 1080p); exposes up to 1080p |

> The three platform nodes `/dev/video0..2` are **hardware codecs** (VSI
> encoder/decoder, ISI m2m), not the webcam. The C920 enumerated as `/dev/video3`.

### Networking — verified ✅
| Interface | State | Address                        |
|-----------|-------|--------------------------------|
| `eth1`    | UP    | `192.168.0.169/24` (DHCP)      |
| `eth0`    | down  | —                              |

> ⚠️ The DART is on `192.168.0.0/24` but the Mac is on `192.168.10.x` — **different
> subnets**, so direct SSH Mac→DART won't route without putting them on the same LAN.
> That's why we drive the board over the **serial console** instead.

## Software / ML stack — verified ✅

| Item                     | Value                                   | NPU-ready? |
|--------------------------|-----------------------------------------|------------|
| `/dev/galcore`           | present (NPU kernel driver)             | ✅ |
| `/usr/lib/libvx_delegate.so` | present (856 KB) — TFLite NPU delegate | ✅ |
| `libtensorflow-lite.so`  | 2.16.2                                  | ✅ |
| Python `tflite_runtime`  | 2.16.2                                  | ✅ |
| Python                   | 3.11.2 (`/usr/bin/python3`)             | ✅ |
| TensorFlow (full)        | not installed (not needed on target)    | n/a |
| Python `cv2` (OpenCV)    | to confirm / may need `pip install`      | ⚠️ |

**Bottom line: the board is NPU-ready.** The delegate, TFLite runtime, Python binding,
and NPU driver are all present — we only need to push an INT8 YOLO `.tflite` model and
run `detect.py --delegate /usr/lib/libvx_delegate.so`.

## Does it make sense? — sanity check

- ✅ **Camera on the DART** (C920 at `/dev/video3`) — correct, frames reach the SoC for NPU inference.
- ✅ **NPU stack already installed** — `galcore` + `libvx_delegate.so` + `tflite_runtime 2.16.2`.
- ✅ **USB-serial to the Mac** — our working control channel (root shell, no password).
- ⚠️ **Different subnets** — SSH Mac→DART needs same LAN; for now use serial.
- ⚠️ **OpenCV Python** not yet confirmed on the board; `detect.py` imports `cv2`.
- ⚠️ **Power supply** not inventoried (USB-serial does not power the SoM).

## Reproduce these checks (over serial, from the Mac)

A helper `~/dart.py` drives the console (auto-login as root) — pipe commands via stdin:

```bash
python3 ~/dart.py <<'EOF'
uname -a
lsusb
v4l2-ctl --list-devices
ip -4 addr
ls -l /usr/lib/libvx_delegate.so
python3 -c "import tflite_runtime; print(tflite_runtime.__version__)"
EOF
```
