#!/usr/bin/env python3
"""Step A: CPU validation (N=3..10) with multicore execution."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

try:
    import cudaq
except Exception:
    cudaq = None

from labs_solver.s_tier_solver import STierLABSSolver


def _run_one(n: int) -> tuple[int, int]:
    if cudaq is not None:
        try:
            cudaq.set_target("qpp-cpu")
        except Exception:
            pass
    solver = STierLABSSolver(
        N=n,
        shots_per_schedule=100,
        population_size=20,
        boltzmann_beta=1.0,
        n_trotter_steps=1,
        use_gpu=False,
    )
    _, energy, _ = solver.solve(max_generations=50, verbose=False)
    return n, int(energy)


def main() -> int:
    cpu_threads = os.environ.get("CPU_WORKERS", "4")
    workers = max(1, int(cpu_threads))
    ns = list(range(3, 11))

    print("Step A: CPU validation (N=3..10)")
    print(f"Workers: {workers}")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_run_one, ns))

    for n, e in results:
        print(f"N={n}: E={e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
