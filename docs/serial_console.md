# Connecting to the DART over USB serial (headless)

How to operate the DART-MX8M-PLUS from a host (Mac/Linux) with **no monitor or
keyboard on the board** — the method this whole project was built with. Useful for
scripts and for AI agents like **Claude Code** that need a shell on the board.

## The link

The board's debug UART reaches the host through an **FTDI FT230X USB-serial** bridge:

| | |
|---|---|
| Device (macOS) | `/dev/cu.usbserial-XXXXXXXX` (e.g. `/dev/cu.usbserial-DK0JTWFN`) |
| Device (Linux) | `/dev/ttyUSB0` |
| Baud           | **115200** 8N1 |
| Login          | user **`root`**, **no password** (Variscite BSP default) |
| Console tty    | `ttymxc0` |

Find the device:

```bash
ls /dev/cu.usbserial-*        # macOS
ls /dev/ttyUSB*               # Linux
# identify it:
python3 -c "import serial.tools.list_ports as p; [print(x.device, x.description, x.hwid) for x in p.comports()]"
```

> ⚠️ **A serial port is exclusive.** Only one program can hold it. Close any
> `screen`/`minicom` before scripting against the port, or you get *Resource busy*:
> `pkill -f "screen.*usbserial"`.

## Option A — interactive (human)

```bash
screen /dev/cu.usbserial-DK0JTWFN 115200
# exit screen: Ctrl-A then K, confirm y
```

## Option B — programmatic (scripts / Claude Code)

Use the helper in this repo, [`tools/dart_console.py`](../tools/dart_console.py). It
auto-detects the port, logs in as root, runs commands piped on stdin, and prints the
output. Requires `pip install pyserial` on the host.

```bash
python3 tools/dart_console.py <<'EOF'
uname -a
ip -4 addr show eth1
ls /dev/video*
EOF
```

### Long-running commands (clone, pip, NPU graph compile)

The first NPU inference compiles the graph (~15-60 s) and emits no output meanwhile,
which can trip the helper's idle timeout. For anything slow, run it on the board under
`nohup` to a log and poll the log instead of blocking on the serial read:

```bash
python3 tools/dart_console.py <<'EOF'
cd ~/yolo_dart
nohup python3 src/detect.py > /tmp/d.log 2>&1 &
echo "started $!"
EOF

# ...later...
python3 tools/dart_console.py <<'EOF'
cat /tmp/d.log
EOF
```

## Networking note

This board's ethernet is on **`eth1`** (e.g. `192.168.0.169/24`), which may be a
different subnet than the host — so SSH host→board does not necessarily route. The
serial console always works and needs no network, which is why it is the control
channel here. If both ends are on the same LAN, `ssh root@<board-ip>` is an
alternative that is not exclusive (serial and SSH can be used at once).

## Gotchas seen in practice

- Baud **must** be 115200; other rates return garbage bytes.
- Killing a `screen` that held the port can drop the login session → the next
  connection lands at `imx8mp-var-dart login:`; the helper logs back in automatically.
- `cu.*` (call-out) is the right macOS node to open, not `tty.*`.
