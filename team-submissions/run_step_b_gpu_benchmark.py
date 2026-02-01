#!/usr/bin/env python3
"""Step B: GPU benchmark for N=14,16,18,20,24."""

from __future__ import annotations

from labs_solver.benchmarks import BenchmarkRunner


def main() -> int:
    runner = BenchmarkRunner(
        N_values=[14, 16, 18, 20, 24],
        runs=2,
        max_generations=200,
        population_size=30,
        verbose=True,
    )
    runner.run_all()
    runner.print_summary()
    runner.save_results("benchmark_results.json")
    try:
        runner.plot_comparison("benchmark_results.png")
    except Exception as exc:
        print(f"Plotting skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
