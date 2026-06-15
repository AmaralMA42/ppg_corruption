import time

import numpy as np

from config import SimulationConfig
from core_simulation import (
    atualiza_total_estrat,
    calcula_payoff,
    inicia_estrategias,
    inicia_vizinhos,
    monte_carlo_single,
    seed_numba,
)
from run_sampling import (
    collect_batch_results,
    create_sample_pool,
    run_batches,
    sample_seeds,
)


ATOL = 1e-10


def assert_close(name, actual, expected):
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=ATOL,
        err_msg=name,
    )


def assert_batch_close(actual, expected):
    labels = [
        "estrat_t",
        "estrat_medio",
        "payavg_t",
        "payavg_medio",
        "activity_t",
        "activity_medio",
        "absorbed_at",
    ]
    for label, actual_array, expected_array in zip(labels, actual, expected):
        assert_close(label, actual_array, expected_array)


def direct_batch(cfg):
    params = cfg.simulation_params()
    total_jog = cfg.total_jog
    viz = np.zeros((total_jog, 4), dtype=np.int32)
    inicia_vizinhos(viz, total_jog, cfg.L)

    results = [
        monte_carlo_single(
            viz,
            params,
            total_jog,
            cfg.total_passos,
            cfg.L,
            seed,
            absorbing_window=cfg.absorbing_window,
        )
        for seed in sample_seeds(cfg.seed, cfg.amostras)
    ]
    return collect_batch_results(results)


def check_seed_repeatability():
    cfg = SimulationConfig(
        L=12,
        amostras=1,
        total_passos=16,
        seed=101,
        absorbing_window=0,
        make_plots=False,
        compute_time_analysis=False,
    )
    params = cfg.simulation_params()
    total_jog = cfg.total_jog
    viz = np.zeros((total_jog, 4), dtype=np.int32)
    inicia_vizinhos(viz, total_jog, cfg.L)

    first = monte_carlo_single(
        viz,
        params,
        total_jog,
        cfg.total_passos,
        cfg.L,
        cfg.seed,
        absorbing_window=cfg.absorbing_window,
    )
    second = monte_carlo_single(
        viz,
        params,
        total_jog,
        cfg.total_passos,
        cfg.L,
        cfg.seed,
        absorbing_window=cfg.absorbing_window,
    )

    for label, actual, expected in zip(
        ["estrat_t", "payavg_t", "activity_t", "absorbed_at"],
        first,
        second,
    ):
        assert_close(label, actual, expected)


def check_pool_equivalence():
    cfg = SimulationConfig(
        L=12,
        amostras=2,
        total_passos=12,
        seed=2026,
        absorbing_window=0,
        make_plots=False,
        compute_time_analysis=False,
    )
    params = cfg.simulation_params()
    expected = direct_batch(cfg)

    new_pool = run_batches(
        cfg.L,
        cfg.amostras,
        cfg.total_passos,
        params,
        base_seed=cfg.seed,
        absorbing_window=cfg.absorbing_window,
    )
    assert_batch_close(new_pool, expected)

    with create_sample_pool(cfg.amostras) as pool:
        reused_pool = run_batches(
            cfg.L,
            cfg.amostras,
            cfg.total_passos,
            params,
            base_seed=cfg.seed,
            absorbing_window=cfg.absorbing_window,
            pool=pool,
        )
        reused_pool_again = run_batches(
            cfg.L,
            cfg.amostras,
            cfg.total_passos,
            params,
            base_seed=cfg.seed,
            absorbing_window=cfg.absorbing_window,
            pool=pool,
        )

    assert_batch_close(reused_pool, expected)
    assert_batch_close(reused_pool_again, expected)


def check_fraction_invariants():
    cfg = SimulationConfig(
        L=14,
        amostras=1,
        total_passos=18,
        seed=303,
        absorbing_window=4,
        make_plots=False,
        compute_time_analysis=False,
    )
    estrat_t, _, payavg_t, _, activity_t, _, absorbed_at = direct_batch(cfg)

    sums = np.sum(estrat_t, axis=0)
    assert_close("fraction sums", sums, np.ones_like(sums))

    if np.any(estrat_t < -ATOL) or np.any(estrat_t > 1.0 + ATOL):
        raise AssertionError("strategy fractions outside [0, 1]")
    if not np.all(np.isfinite(payavg_t)):
        raise AssertionError("non-finite payoff average")
    if np.any(activity_t < -ATOL) or np.any(activity_t > 1.0 + ATOL):
        raise AssertionError("activity outside [0, 1]")
    if np.any(absorbed_at < 0) or np.any(absorbed_at > cfg.total_passos):
        raise AssertionError("absorbed_at outside valid range")

    stop = int(absorbed_at[0])
    if stop < cfg.total_passos:
        tail_steps = cfg.total_passos - stop
        expected_strat = np.repeat(estrat_t[:, 0, stop - 1 : stop], tail_steps, axis=1)
        expected_payoff = np.repeat(payavg_t[:, 0, stop - 1 : stop], tail_steps, axis=1)
        assert_close("absorbed strategy tail", estrat_t[:, 0, stop:], expected_strat)
        assert_close("absorbed payoff tail", payavg_t[:, 0, stop:], expected_payoff)
        assert_close(
            "absorbed activity tail",
            activity_t[0, stop:],
            np.zeros_like(activity_t[0, stop:]),
        )


def check_cd_initial_condition_has_no_p():
    cfg = SimulationConfig(
        L=16,
        amostras=1,
        total_passos=20,
        cond_ini=4,
        seed=404,
        absorbing_window=0,
        make_plots=False,
        compute_time_analysis=False,
    )
    estrat_t, _, _, _, _, _, _ = direct_batch(cfg)
    assert_close("P fraction in C/D limit", estrat_t[2], np.zeros_like(estrat_t[2]))


def check_local_payoff_consistency():
    cfg = SimulationConfig(
        L=10,
        amostras=1,
        total_passos=1,
        seed=505,
        absorbing_window=0,
        make_plots=False,
        compute_time_analysis=False,
    )
    params = cfg.simulation_params()
    total_jog = cfg.total_jog
    viz = np.zeros((total_jog, 4), dtype=np.int32)
    estrategia = np.zeros(total_jog, dtype=np.int32)
    payoff = np.zeros(total_jog)

    inicia_vizinhos(viz, total_jog, cfg.L)
    seed_numba(cfg.seed)
    inicia_estrategias(estrategia, total_jog, cfg.L, params[6])
    for sitio in range(total_jog):
        payoff[sitio] = calcula_payoff(sitio, estrategia, viz, params)

    for _ in range(6):
        atualiza_total_estrat(estrategia, payoff, viz, params, total_jog)
        fresh = np.zeros(total_jog)
        for sitio in range(total_jog):
            fresh[sitio] = calcula_payoff(sitio, estrategia, viz, params)
        assert_close("local payoff update consistency", payoff, fresh)


CHECKS = [
    ("seed_repeatability", check_seed_repeatability),
    ("pool_equivalence", check_pool_equivalence),
    ("fraction_invariants", check_fraction_invariants),
    ("cd_initial_condition_has_no_p", check_cd_initial_condition_has_no_p),
    ("local_payoff_consistency", check_local_payoff_consistency),
]


def main():
    print("Running simulation safety checks...")
    start = time.perf_counter()
    for name, check in CHECKS:
        t0 = time.perf_counter()
        check()
        print(f"OK {name}: {time.perf_counter() - t0:.2f}s")
    print(f"All safety checks passed in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
