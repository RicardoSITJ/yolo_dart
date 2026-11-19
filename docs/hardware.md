# Hardware setup

Bench setup for running YOLO object detection on the NPU of the **DART-MX8M-PLUS**
(NXP i.MX 8M Plus). The DART is the compute target; the Mac is used only as a serial
console / development host.

> **Verification note.** Everything under "Host (Mac)" was read directly from the
> machine (`system_profiler`, `sw_vers`, `ifconfig`) on 2026-11-19. The DART-side
> peripherals are listed as reported by the user — they are **not** on the Mac's USB
> bus (confirmed: the only USB device on the Mac is the FTDI serial adapter), so they
> can only be verified from the board's own console once serial access works.

## Topology

```
                 USB (serial console, FTDI FT230X)
   ┌──────────┐  /dev/cu.usbserial-DK0JTWFN @115200   ┌────────────────────────┐
   │  MacBook │ ◄───────────────────────────────────► │  DART-MX8M-PLUS (SoM)  │
   │  Pro M3  │                                        │  NXP i.MX 8M Plus      │
   └──────────┘                                        │  + VIP8000 NPU         │
                                                       └──────────┬─────────────┘
                                                                  │ on the DART / carrier:
                                                     ┌────────────┼───────────────┐
                                                     │            │               │
                                                USB keyboard   USB camera      Ethernet
                                                             (Logitech 1080p)   (RJ45)
```

The Mac does **not** see the keyboard, camera, or ethernet — those hang off the DART's
carrier board. The only physical link between Mac and DART is the USB-serial console.

## Host (Mac) — verified ✅

| Spec              | Value                                            |
|-------------------|--------------------------------------------------|
| Model             | MacBook Pro (`Mac15,3`)                           |
| Chip              | Apple **M3**                                      |
| CPU cores         | 8 (4 performance + 4 efficiency)                  |
| Memory            | 16 GB unified                                     |
| macOS             | 14.2 (build 23C64)                               |
| Active network    | `en0` (Wi-Fi), IP `192.168.10.87/23`             |
| Role              | Serial console + dev host (model export)          |

USB devices actually enumerated on the Mac:

| Device             | Detail                                                  |
|--------------------|---------------------------------------------------------|
| FTDI **FT230X Basic UART** | VID `0x0403` / PID `0x6015`, serial `DK0JTWFN`, up to 12 Mb/s → `/dev/cu.usbserial-DK0JTWFN` |

> Built-in **FaceTime HD Camera** exists on the Mac but is unrelated to the DART
> inference pipeline. Two USB-Ethernet adapters (`en3`, `en4`) are present but
> currently **inactive**.

## Target (DART-MX8M-PLUS) — as reported ⚠️ (verify from board)

| Component         | Details                                                   | Status |
|-------------------|-----------------------------------------------------------|--------|
| SoM               | Variscite DART-MX8M-PLUS (NXP i.MX 8M Plus, 4× Cortex-A53) | reported |
| NPU               | Verisilicon VIP8000, ~2.3 TOPS, INT8                       | by SoC spec |
| USB keyboard      | Generic USB keyboard (local input on the board)           | reported |
| USB camera        | Logitech HD 1080p webcam (UVC). Likely C920/C922 class     | reported |
| Ethernet          | RJ45 cable to LAN (SSH / `git clone` from the board)       | reported |
| Serial console    | FTDI FT230X → Mac, 115200 8N1                              | partial* |

\* The serial link enumerates on the Mac, but a clean console read is still pending
(see `docs/npu_setup.md` / serial troubleshooting). Readable boot output is needed to
confirm the OS, NPU driver (`galcore`), and the attached peripherals from the board side.

## Does it make sense? — sanity check

- ✅ **Camera on the DART, not the Mac.** Correct: YOLO must run on the i.MX 8M Plus
  NPU, so the camera frames have to reach the board. A UVC Logitech 1080p webcam is a
  standard, well-supported V4L2 source on the i.MX BSP.
- ✅ **Keyboard on the DART.** Lets you log in locally without depending on the serial
  console — reasonable for a dev bench.
- ✅ **Ethernet on the DART.** Needed so the board can `git clone` this public repo and
  SSH in — exactly what the repo README assumes.
- ✅ **USB-serial to the Mac.** Standard way to get the boot/U-Boot console and debug
  early boot before the network is up.
- ⚠️ **Power not listed.** The DART carrier needs its own power supply (USB-serial does
  not power the SoM). Worth adding to the inventory.
- ⚠️ **USB bandwidth.** Keyboard + 1080p webcam share the i.MX 8M Plus USB. Fine for
  MJPEG/compressed 1080p; raw YUY2 at high FPS can saturate a single USB 2.0 port.

## To verify next (from the board)

Once the serial console (or SSH over ethernet) is readable, confirm on the DART:

```bash
uname -a                      # kernel / BSP
cat /etc/os-release           # Yocto image
v4l2-ctl --list-devices       # the Logitech webcam as /dev/videoN
v4l2-ctl -d /dev/video0 --list-formats-ext   # supported resolutions/FPS
ip addr                       # ethernet link + IP
ls /dev/galcore               # NPU kernel driver present
find / -name libvx_delegate.so 2>/dev/null   # NPU TFLite delegate
```
