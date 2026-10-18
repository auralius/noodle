"""PlatformIO USBasp helper for Arduino Nano/Uno ATmega328P.

Built-in PlatformIO handles uploads (`upload_protocol = usbasp`).
This script adds a separate, one-time, opt-in task to disable BOOTRST,
retaining all other fuse settings.

Usage:
  pio run -e uno-int8-pgm -t usbasp-fuse-info
  pio run -e uno-int8-pgm -t usbasp-bootloader-off
  pio run -e uno-int8-pgm -t upload

Keep ISP wired while running the first two tasks. Disabling BOOTRST
means reset starts at application address 0; the AVR bootloader will
no longer run. A subsequent ISP flash erase/write replaces its code.
"""

Import("env")

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

BITCLOCK = "20"  # Conservative ISP bitclock (microseconds); equivalent to upload_flags = -B20


def _avrdude_paths():
    package = env.PioPlatform().get_package_dir("tool-avrdude")
    candidates = []
    config_candidates = []
    if package:
        root = Path(package)
        candidates += [root / "bin" / "avrdude", root / "avrdude"]
        if os.name == "nt":
            candidates += [root / "bin" / "avrdude.exe", root / "avrdude.exe"]
        config_candidates += [root / "avrdude.conf", root / "etc" / "avrdude.conf"]

    executable = next((str(p) for p in candidates if p.is_file()), None)
    if executable is None:
        executable = shutil.which("avrdude")
    if executable is None:
        raise RuntimeError("AVRDUDE executable not found. Install PlatformIO tool-avrdude or avrdude on PATH.")

    conf = next((str(p) for p in config_candidates if p.is_file()), None)
    return executable, conf


def _mcu():
    mcu = str(env.BoardConfig().get("build.mcu", "")).lower()
    if mcu in ("atmega328p", "m328p"):
        return "m328p"
    # Reclaiming flash via BOOTRST differs across MCU families; refuse changes.
    raise RuntimeError(
        "Fuse task supports only ATmega328P; board build.mcu is %r. "
        "Standard PlatformIO USBasp uploads remain supported for other MCUs." % mcu
    )


def _run_avrdude(args):
    exe, conf = _avrdude_paths()
    cmd = [exe]
    if conf:
        cmd += ["-C", conf]
    cmd += ["-c", "usbasp", "-P", "usb", "-p", _mcu(), "-B", BITCLOCK]
    cmd += args
    print("USBasp command: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _read_hfuse():
    # A raw temporary file is more robust than parsing different versions'
    # stdout/stderr text for -U hfuse:r:-:h.
    with tempfile.TemporaryDirectory(prefix="usbasp_fuse_") as td:
        location = Path(td) / "high_fuse.bin"
        _run_avrdude(["-U", "hfuse:r:%s:r" % location])
        data = location.read_bytes()
        if len(data) != 1:
            raise RuntimeError("Expected exactly one hfuse byte; read %d bytes" % len(data))
        return data[0]


def show_fuses(target, source, env):
    old = _read_hfuse()
    print("ATmega328P high fuse: 0x%02X" % old)
    print("BOOTRST: %s" % ("disabled (application starts at 0x0000)" if (old & 1) else "enabled (reset enters boot section)"))
    print("Proposed bootloader-off high fuse (only BOOTRST bit changed): 0x%02X" % (old | 1))


def disable_bootrst(target, source, env):
    old = _read_hfuse()
    new = old | 0x01   # BOOTRST=1 (unprogrammed): Reset goes to 0x0000.
    print("ATmega328P hfuse: 0x%02X -> 0x%02X" % (old, new))
    if new == old:
        print("BOOTRST is already disabled. No fuse write needed.")
        return
    # Do NOT modify RSTDISBL, SPIEN, oscillator, BOD, or BOOTSZ bits.
    _run_avrdude(["-U", "hfuse:w:0x%02X:m" % new])
    actual = _read_hfuse()
    if actual != new:
        raise RuntimeError("Fuse verification failed: wrote 0x%02X, read 0x%02X" % (new, actual))
    print("BOOTRST disabled and verified. Use `pio run -e <environment> -t upload` to flash via ISP.")
    print("NOTE: USB-serial bootloader uploads will no longer work, but Serial remains usable.")


def prepare_fullflash(target, source, env):
    print("Preparing ATmega328P for 32 KB application flash. This will erase the firmware and bootloader.")
    print("WARNING: chip erase may also erase EEPROM, depending on the EESAVE fuse.")
    disable_bootrst(target, source, env)
    _run_avrdude(["-e"])
    print("Chip erased. Now upload the freshly built firmware with `pio run -e <environment> -t upload`.")


env.AddCustomTarget(
    name="usbasp-fuse-info",
    dependencies=None,
    actions=[show_fuses],
    title="Read AVR boot-reset fuse",
    description="Read ATmega328P BOOTRST without modifying the MCU",
)

env.AddCustomTarget(
    name="usbasp-bootloader-off",
    dependencies=None,
    actions=[disable_bootrst],
    title="Disable ATmega328P bootloader reset",
    description="One-time USBasp fuse write: only set BOOTRST bit to 1",
)


env.AddCustomTarget(
    name="usbasp-fullflash-init",
    dependencies=None,
    actions=[prepare_fullflash],
    title="Enable 32 KB application flash via ISP",
    description="ONE-TIME: preserve fuses except BOOTRST, then erase flash/bootloader",
)
