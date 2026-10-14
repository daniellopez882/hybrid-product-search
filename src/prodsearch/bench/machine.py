"""Provenance recorded with every results file: machine, packages, git state, local date."""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

PACKAGES = ("numpy", "numba", "onnxruntime", "tokenizers", "falcon", "bm25s", "faiss-cpu")


def _cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _physical_cores() -> int | None:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor).NumberOfCores",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout
            return sum(int(x) for x in out.split())
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cores = {
            line.split(":", 1)[1].strip()
            for line in cpuinfo.read_text().splitlines()
            if line.startswith(("core id", "physical id"))
        }
        return len(cores) or None
    return None


def _ram_bytes() -> int | None:
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.ullTotalPhys)
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return None


def machine() -> dict[str, object]:
    ram = _ram_bytes()
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "cpu": _cpu_name(),
        "physical_cores": _physical_cores(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": None if ram is None else round(ram / 2**30, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": versions,
        "accelerator": "none (CPU only)",
    }


def _machine_cpu_seconds() -> tuple[float, float] | None:
    """(busy, total) CPU seconds since boot, summed over all logical CPUs."""
    if sys.platform == "win32":
        idle, kernel, user = ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong()
        ok = ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        )
        if not ok:
            return None
        # FILETIME units of 100 ns; kernel time includes idle time.
        total = (kernel.value + user.value) / 1e7
        return total - idle.value / 1e7, total
    stat = Path("/proc/stat")
    if stat.is_file():
        ticks = [int(x) for x in stat.read_text().splitlines()[0].split()[1:9]]
        hz = os.sysconf("SC_CLK_TCK")
        idle = ticks[3] + ticks[4]  # idle + iowait
        return (sum(ticks) - idle) / hz, sum(ticks) / hz
    return None


class CpuMeter:
    """CPU time used by this process and by every other process over an interval.

    Latency figures are only comparable when the rest of the machine is quiet, so the benchmark
    records how many cores other processes kept busy while it was timing.
    """

    def __init__(self) -> None:
        self._wall = time.perf_counter()
        self._process = time.process_time()
        self._machine = _machine_cpu_seconds()

    def read(self) -> dict[str, float | None]:
        wall = time.perf_counter() - self._wall
        process = time.process_time() - self._process
        out: dict[str, float | None] = {
            "wall_seconds": wall,
            "this_process_cpu_seconds": process,
            "this_process_mean_cores": process / wall if wall > 0 else None,
            "other_processes_cpu_seconds": None,
            "other_processes_mean_cores": None,
        }
        now = _machine_cpu_seconds()
        if self._machine is not None and now is not None and wall > 0:
            other = max(0.0, now[0] - self._machine[0] - process)
            out["other_processes_cpu_seconds"] = other
            out["other_processes_mean_cores"] = other / wall
        return out


def git_state(repo: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "git_sha": run("rev-parse", "HEAD"),
            "git_dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "git_dirty": None}


def local_now() -> datetime:
    """Local wall-clock time with its UTC offset; results are dated in the machine's zone."""
    return datetime.now().astimezone()
