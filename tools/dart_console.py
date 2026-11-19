#!/usr/bin/env python3
"""Drive the DART-MX8M-PLUS serial console from a host (Mac/Linux).

This is the exact helper used to bring this project up headlessly: it opens the
FTDI USB-serial console, logs in as root (the Variscite BSP root has no password),
runs the shell commands piped on stdin, and prints the cleaned output. It is how
an agent like Claude Code — or any script — can operate the board without a
keyboard/monitor, and without an interactive `screen` session.

IMPORTANT: a serial port is exclusive. Close any `screen`/`minicom` on the same
device first, or you will get "Resource busy".

Usage:
    python3 tools/dart_console.py <<'EOF'
    uname -a
    ip -4 addr show eth1
    EOF

    # explicit port / baud:
    python3 tools/dart_console.py --port /dev/cu.usbserial-XXXX --baud 115200 <<'EOF'
    ls /dev/video*
    EOF

For long-running commands (git clone, wget, pip, NPU graph compile), launch them
on the board with nohup to a log and poll the log, e.g.:
    nohup python3 src/detect.py > /tmp/d.log 2>&1 &   # then: cat /tmp/d.log
"""
import argparse
import glob
import re
import sys
import time

import serial   # pip install pyserial

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def find_port():
    for pat in ("/dev/cu.usbserial-*", "/dev/ttyUSB*", "/dev/tty.usbserial-*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    raise SystemExit("no USB-serial device found (is the DART FTDI plugged in?)")


def clean(text):
    return ANSI.sub("", text).replace("\r", "")


def read_until_prompt(ser, prompt, hard_timeout, idle_timeout):
    buf = bytearray()
    start = last = time.time()
    while True:
        d = ser.read(4096)
        if d:
            buf += d
            last = time.time()
            if clean(buf.decode("utf-8", "replace"))[-40:].rstrip().endswith(prompt) \
                    and time.time() - start > 0.4:
                break
        if time.time() - last > idle_timeout or time.time() - start > hard_timeout:
            break
    return clean(buf.decode("utf-8", "replace"))


def ensure_login(ser, user, prompt):
    ser.write(b"\n"); ser.flush()
    out = read_until_prompt(ser, prompt, 4, 1.2)
    if out.strip().endswith(("#", "$")):
        return
    if "Password:" in out:
        ser.write(b"\n"); ser.flush(); out = read_until_prompt(ser, prompt, 4, 1.2)
    if "login:" in out:
        ser.write((user + "\n").encode()); ser.flush()
        out = read_until_prompt(ser, prompt, 4, 1.2)
        if "Password" in out:                      # BSP root has no password
            ser.write(b"\n"); ser.flush(); read_until_prompt(ser, prompt, 4, 1.2)
    ser.write(b"printf '\\033[?2004l'\n"); ser.flush()   # disable bracketed paste
    read_until_prompt(ser, prompt, 3, 0.8)


def run(ser, cmd, prompt, hard_timeout, idle_timeout):
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode()); ser.flush()
    out = read_until_prompt(ser, prompt, hard_timeout, idle_timeout)
    lines = [l for l in out.splitlines()
             if l.strip() != cmd.strip() and not l.strip().endswith(prompt)]
    return "\n".join(lines).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None, help="serial device (auto-detected if omitted)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--user", default="root")
    ap.add_argument("--prompt", default="var-dart:~#",
                    help="shell prompt tail used to detect command completion")
    ap.add_argument("--hard-timeout", type=float, default=240.0)
    ap.add_argument("--idle-timeout", type=float, default=8.0)
    args = ap.parse_args()

    cmds = [l.rstrip("\n") for l in sys.stdin if l.strip()]
    if not cmds:
        raise SystemExit("no commands on stdin")

    port = args.port or find_port()
    ser = serial.Serial(port, args.baud, timeout=0.3)
    print(f"[dart] {port} @ {args.baud}")
    try:
        ensure_login(ser, args.user, args.prompt)
        for cmd in cmds:
            print(f"\n===== $ {cmd} =====")
            print(run(ser, cmd, args.prompt, args.hard_timeout, args.idle_timeout))
    finally:
        ser.close()


if __name__ == "__main__":
    main()
