"""
Diagnostic script to check payoff calculations for classical PGG (cond_ini=4).

In the classical 2D square lattice PGG:
- Each player participates in 5 games (self + 4 neighbors)
- Cost c per game if cooperating
- All players pay c (cooperators AND defectors)
- Both get the public good benefit

This script checks if the payoff function correctly handles this.
"""

import numpy as np
from config import SimulationConfig
from core_simulation import calcula_payoff, inicia_vizinhos, public_good_benefit, conta_estrat

def check_payoff_consistency():
    """Verify payoff consistency for classical game (only C and D)."""

    cfg = SimulationConfig(
        L=11,  # Small lattice for debugging
        cond_ini=4,
        r=3.7,  # Literature critical value
        G=5,
        c=1.0,
        sigma=1.02,  # Irrelevant for classical C vs D
        alpha=0.3,   # Irrelevant for classical C vs D
        k=0.1
    )

    total_jog = cfg.L * cfg.L
    params = cfg.simulation_params()

    # Initialize lattice
    viz = np.zeros((total_jog, 4), dtype=np.int32)
    inicia_vizinhos(viz, total_jog, cfg.L)

    # Test case 1: All cooperators
    print("=" * 70)
    print("TEST 1: Uniform Cooperator lattice (all C)")
    print("=" * 70)
    estrategia = np.zeros(total_jog, dtype=np.int32)  # All cooperators (0)

    # Check central player
    center = cfg.L * (cfg.L // 2) + (cfg.L // 2)
    payoff = calcula_payoff(center, estrategia, viz, params)

    print(f"Player at center ({center}): Strategy = C (Cooperator)")
    print(f"Neighbors: 4 Cooperators")
    print(f"Groups this player is in: 5 (self + 4 neighbors)")
    print(f"\nExpected payoff calculation:")
    print(f"  - Each group has 5 cooperators (C+D+corrupt = 5+0+0)")
    print(f"  - Public good = r*c * (5/G) = {cfg.r} * 1.0 * (5/{cfg.G}) = {cfg.r * 5 / cfg.G:.4f}")
    print(f"  - Per game: +{cfg.r * 5 / cfg.G:.4f} (benefit)")
    print(f"  - Per game: -c = -1.0 (cost to cooperate)")
    print(f"  - Net per game: {cfg.r * 5 / cfg.G - 1.0:.4f}")
    print(f"  - Total (5 games): 5 * {cfg.r * 5 / cfg.G - 1.0:.4f} = {5 * (cfg.r * 5 / cfg.G - 1.0):.4f}")
    print(f"\nActual payoff: {payoff:.4f}")
    expected_c_all_c = 5 * (cfg.r * cfg.c * 5 / cfg.G - cfg.c)
    print(f"Expected: {expected_c_all_c:.4f}")
    print(f"Match: {np.isclose(payoff, expected_c_all_c)}")

    # Test case 2: Single defector surrounded by cooperators
    print("\n" + "=" * 70)
    print("TEST 2: Single Defector surrounded by Cooperators")
    print("=" * 70)
    estrategia = np.zeros(total_jog, dtype=np.int32)  # All C
    estrategia[center] = 1  # Make center a defector

    payoff_d = calcula_payoff(center, estrategia, viz, params)
    payoff_c = calcula_payoff(center + 1, estrategia, viz, params)

    print(f"\nDefector at center (surrounded by 5 C):")
    print(f"  - Each group has 4 cooperators + 1 defector (self)")
    print(f"  - Public good benefit = r*c * (4/G) = {cfg.r * cfg.c * 4 / cfg.G:.4f}")
    print(f"  - Defector pays: 0 (BUG if true!)")
    print(f"  - Net per game: {cfg.r * cfg.c * 4 / cfg.G:.4f}")
    print(f"  - Total (5 games): 5 * {cfg.r * cfg.c * 4 / cfg.G:.4f} = {5 * cfg.r * cfg.c * 4 / cfg.G:.4f}")
    print(f"\nActual defector payoff: {payoff_d:.4f}")

    print(f"\nCooperator neighbor:")
    print(f"  - Groups: 4 with center D + 1 with only C neighbors")
    print(f"  - Group with D: 4 C + 1 D => benefit = r*c*(4/5) = {cfg.r * cfg.c * 4 / cfg.G:.4f}")
    print(f"  - Cost per game: -c = -1.0")
    print(f"\nActual cooperator payoff: {payoff_c:.4f}")

    print(f"\nPayoff DIFFERENCE (D - C neighbor): {payoff_d - payoff_c:.4f}")
    print(f"(Positive means defectors benefit - this is the advantage bug!)")

    # Test case 3: All defectors
    print("\n" + "=" * 70)
    print("TEST 3: Uniform Defector lattice (all D)")
    print("=" * 70)
    estrategia = np.ones(total_jog, dtype=np.int32)  # All defectors (1)

    payoff = calcula_payoff(center, estrategia, viz, params)
    print(f"Player at center: Strategy = D (Defector)")
    print(f"Neighbors: 4 Defectors")
    print(f"\nExpected payoff calculation:")
    print(f"  - Each group has 5 defectors (0 cooperators)")
    print(f"  - Public good = r*c * (0/G) = 0.0 (no cooperators!)")
    print(f"  - Defector pays: 0")
    print(f"  - Net per game: 0.0")
    print(f"  - Total (5 games): 0.0")
    print(f"\nActual payoff: {payoff:.4f}")

    # Summary and diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS:")
    print("=" * 70)

    estrategia = np.zeros(total_jog, dtype=np.int32)
    estrategia[center] = 1
    payoff_d = calcula_payoff(center, estrategia, viz, params)

    estrategia[center] = 0
    payoff_c_pure = calcula_payoff(center, estrategia, viz, params)

    # What payoff should a defector get?
    # In classical PGG all pay c and all get benefit
    theoretical_d_payoff = 5 * cfg.r * cfg.c * 4 / cfg.G - 5 * cfg.c  # Should pay c!
    actual_d_payoff = payoff_d

    print(f"\nFor classical PGG (C and D only):")
    print(f"  Defector should pay c per game: YES (like everyone)")
    print(f"  Actual defector cost in code: {'NO COST' if payoff_d > 5 * cfg.r * cfg.c * 4 / cfg.G else 'COST c'}")
    print(f"\nActual defector payoff: {actual_d_payoff:.4f}")
    print(f"Theoretical (if paying c): {theoretical_d_payoff:.4f}")
    print(f"Difference: {actual_d_payoff - theoretical_d_payoff:.4f}")

    if np.isclose(actual_d_payoff, 5 * cfg.r * cfg.c * 4 / cfg.G, atol=0.01):
        print(f"\n🐛 BUG CONFIRMED: Defectors do NOT pay cost c!")
        print(f"    They gain unfair advantage of {5 * cfg.c:.1f} per MC step")
        print(f"    This pushes critical r-value from ~3.74 to higher values")
    else:
        print(f"\n✓ Defectors correctly pay c")

if __name__ == "__main__":
    check_payoff_consistency()

