import py_compile
import time
from pathlib import Path

from validate_safety import main as run_safety_checks


CRITICAL_MODULES = [
    "config.py",
    "core_simulation.py",
    "run_sampling.py",
    "run_sweep.py",
    "run_phaseD.py",
    "time_analysis.py",
    "utils.py",
    "validate_safety.py",
]


def compile_critical_modules():
    src_dir = Path(__file__).resolve().parent
    for module in CRITICAL_MODULES:
        py_compile.compile(src_dir / module, doraise=True)


def main():
    start = time.perf_counter()
    print("Running safety gate...")
    compile_critical_modules()
    print("OK py_compile critical modules")
    run_safety_checks()
    print(f"Safety gate passed in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
