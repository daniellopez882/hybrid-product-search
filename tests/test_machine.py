import time

from prodsearch.bench.machine import CpuMeter, machine


def test_cpu_meter_reports_this_process_and_the_rest() -> None:
    meter = CpuMeter()
    end = time.perf_counter() + 0.2
    while time.perf_counter() < end:
        pass
    load = meter.read()
    assert load["wall_seconds"] >= 0.2
    assert load["this_process_cpu_seconds"] > 0
    # Other processes' share is measured where the OS exposes machine-wide CPU times.
    if load["other_processes_cpu_seconds"] is not None:
        assert load["other_processes_cpu_seconds"] >= 0
        assert load["other_processes_mean_cores"] >= 0


def test_machine_block_names_cpu_and_packages() -> None:
    block = machine()
    assert block["cpu"]
    assert block["python"].startswith("3.12")
    assert "numba" in block["packages"]
