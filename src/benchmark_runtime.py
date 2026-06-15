import time
from dataclasses import replace

import numpy as np

from config import SimulationConfig
from core_simulation import inicia_vizinhos, monte_carlo_single
from run_sampling import create_sample_pool, run_batches


def timed(label, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.3f}s")
    return result, elapsed


def direct_warm_run(cfg):
    params = cfg.simulation_params()
    total_jog = cfg.total_jog
    viz = np.zeros((total_jog, 4), dtype=np.int32)
    inicia_vizinhos(viz, total_jog, cfg.L)

    monte_carlo_single(
        viz,
        params,
        total_jog,
        1,
        cfg.L,
        cfg.seed,
        absorbing_window=cfg.absorbing_window,
    )

    return monte_carlo_single(
        viz,
        params,
        total_jog,
        cfg.total_passos,
        cfg.L,
        cfg.seed + 1,
        absorbing_window=cfg.absorbing_window,
    )


def run_batches_single_point(cfg):
    return run_batches(
        cfg.L,
        cfg.amostras,
        cfg.total_passos,
        cfg.simulation_params(),
        base_seed=cfg.seed,
        absorbing_window=cfg.absorbing_window,
    )


def sweep_reused_pool(cfg, values, param_name):
    with create_sample_pool(cfg.amostras) as pool:
        for value in values:
            cfg_v = replace(cfg, **{param_name: float(value)})
            run_batches(
                cfg_v.L,
                cfg_v.amostras,
                cfg_v.total_passos,
                cfg_v.simulation_params(),
                base_seed=cfg_v.seed,
                absorbing_window=cfg_v.absorbing_window,
                pool=pool,
            )


def run_case(name, cfg, values):
    print(f"\n=== {name} ===")
    print(
        f"L={cfg.L}, amostras={cfg.amostras}, "
        f"passos={cfg.total_passos}, pontos={len(values)}"
    )

    _, direct_time = timed("direct_warm_run", lambda: direct_warm_run(cfg))
    _, batch_time = timed("run_batches_single_point", lambda: run_batches_single_point(cfg))
    _, sweep_time = timed(
        "sweep_reused_pool",
        lambda: sweep_reused_pool(cfg, values, "r"),
    )

    n_points = len(values)
    total_mcs_samples = n_points * cfg.amostras * cfg.total_passos
    print(f"time_per_sweep_point: {sweep_time / n_points:.3f}s")
    print(f"time_per_mcs_sample: {sweep_time / total_mcs_samples:.6f}s")
    print(f"direct_time_per_mcs: {direct_time / cfg.total_passos:.6f}s")
    print(f"single_batch_per_sample: {batch_time / cfg.amostras:.3f}s")


def benchmark_cases():
    return [
        (
            "small",
            SimulationConfig(
                L=40,
                amostras=2,
                total_passos=40,
                absorbing_window=0,
                make_plots=False,
                compute_time_analysis=False,
                compress_output=False,
            ),
            np.linspace(3.0, 3.2, 3),
        ),
        (
            "medium",
            SimulationConfig(
                L=80,
                amostras=2,
                total_passos=100,
                absorbing_window=0,
                make_plots=False,
                compute_time_analysis=False,
                compress_output=False,
            ),
            np.linspace(3.0, 3.2, 3),
        ),
    ]


def main():
    print("Runtime benchmark")
    print("This measures runtime only; use safety_gate.py for physical validation.")
    start = time.perf_counter()
    for name, cfg, values in benchmark_cases():
        run_case(name, cfg, values)
    print(f"\nTotal benchmark time: {time.perf_counter() - start:.3f}s")


if __name__ == "__main__":
    main()
