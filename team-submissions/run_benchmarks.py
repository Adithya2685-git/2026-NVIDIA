#!/usr/bin/env python3
"""
Run Benchmarks and Generate Plots for S-Tier LABS Solver

Usage:
    cd team-submissions
    python3 run_benchmarks.py

This will:
1. Run benchmarks comparing Classical MTS vs S-Tier
2. Test different N values
3. Generate plots showing energy and timing results
4. Save results to benchmark_results.png and benchmark_data.json
"""

import sys
import time
import json
import numpy as np
from datetime import datetime

# Setup matplotlib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
import matplotlib.pyplot as plt

# Import our modules
from labs_solver.labs_energy import calculate_energy, KNOWN_OPTIMA, random_sequence
from labs_solver.mts import memetic_tabu_search
from labs_solver.boltzmann_seeding import boltzmann_seeding, diversity_metrics


def run_benchmark(N, runs=3, max_generations=100, population_size=30, verbose=True):
    """
    Run benchmark for a single N value.
    
    Returns dict with results for classical and s_tier methods.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmarking N={N} ({runs} runs, {max_generations} generations)")
        print(f"{'='*60}")
    
    results = {
        'classical': [],
        's_tier': []
    }
    
    for run in range(runs):
        # Classical MTS
        start = time.time()
        _, energy, stats = memetic_tabu_search(
            N, 
            population_size=population_size,
            max_generations=max_generations,
            verbose=False
        )
        elapsed = time.time() - start
        
        results['classical'].append({
            'energy': int(energy),
            'time': elapsed,
            'generations': stats['generations']
        })
        
        # S-Tier (with Boltzmann seeding)
        start = time.time()
        
        # Generate random samples (replace with quantum samples when CUDA-Q available)
        samples = [random_sequence(N) for _ in range(500)]
        sample_energies = [calculate_energy(s) for s in samples]
        
        # Boltzmann seeding
        population = boltzmann_seeding(
            list(zip(samples, sample_energies)),
            K=population_size,
            beta=1.0
        )
        
        # Run MTS
        _, energy, stats = memetic_tabu_search(
            N,
            population=population,
            population_size=population_size,
            max_generations=max_generations,
            verbose=False
        )
        elapsed = time.time() - start
        
        results['s_tier'].append({
            'energy': int(energy),
            'time': elapsed,
            'generations': stats['generations']
        })
        
        if verbose:
            c = results['classical'][-1]
            s = results['s_tier'][-1]
            print(f"Run {run+1}: Classical E={c['energy']} t={c['time']:.2f}s | "
                  f"S-Tier E={s['energy']} t={s['time']:.2f}s")
    
    return results


def calculate_summary(results, N_values):
    """Calculate summary statistics from results."""
    summary = {}
    
    for method in ['classical', 's_tier']:
        summary[method] = {}
        for N in N_values:
            if N in results and method in results[N]:
                energies = [r['energy'] for r in results[N][method]]
                times = [r['time'] for r in results[N][method]]
                
                summary[method][N] = {
                    'mean_energy': float(np.mean(energies)),
                    'std_energy': float(np.std(energies)),
                    'min_energy': int(min(energies)),
                    'max_energy': int(max(energies)),
                    'mean_time': float(np.mean(times)),
                    'std_time': float(np.std(times))
                }
    
    return summary


def plot_results(summary, N_values, save_path='benchmark_results.png'):
    """Generate comparison plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'classical': '#1f77b4', 's_tier': '#2ca02c'}
    labels = {'classical': 'Classical MTS', 's_tier': 'S-Tier (Multi-Schedule + Boltzmann)'}
    
    # Plot 1: Energy vs N
    for method in ['classical', 's_tier']:
        if method not in summary:
            continue
        
        means = [summary[method][N]['mean_energy'] for N in N_values if N in summary[method]]
        stds = [summary[method][N]['std_energy'] for N in N_values if N in summary[method]]
        valid_Ns = [N for N in N_values if N in summary[method]]
        
        if means:
            ax1.errorbar(valid_Ns, means, yerr=stds, marker='o', markersize=8,
                        label=labels[method], color=colors[method],
                        capsize=5, capthick=2, linewidth=2)
    
    # Optimal line
    opt_Ns = [N for N in N_values if N in KNOWN_OPTIMA]
    opt_Es = [KNOWN_OPTIMA[N][0] for N in opt_Ns]
    if opt_Ns:
        ax1.plot(opt_Ns, opt_Es, 'k--', label='Known Optimal', 
                linewidth=2, markersize=8, marker='s')
    
    ax1.set_xlabel('Sequence Length N', fontsize=12)
    ax1.set_ylabel('Energy (lower is better)', fontsize=12)
    ax1.set_title('Solution Quality vs Sequence Length', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(N_values)
    
    # Plot 2: Time vs N
    for method in ['classical', 's_tier']:
        if method not in summary:
            continue
        
        times = [summary[method][N]['mean_time'] for N in N_values if N in summary[method]]
        valid_Ns = [N for N in N_values if N in summary[method]]
        
        if times:
            ax2.plot(valid_Ns, times, marker='s', markersize=8,
                    label=labels[method], color=colors[method], linewidth=2)
    
    ax2.set_xlabel('Sequence Length N', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Computation Time vs Sequence Length', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(N_values)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {save_path}")
    
    return fig


def print_summary_table(summary, N_values):
    """Print formatted summary table."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f"{'N':<5} {'Method':<20} {'Mean E':<10} {'Std':<8} {'Best':<8} {'Time(s)':<12}")
    print("-"*80)
    
    for N in N_values:
        for method in ['classical', 's_tier']:
            if method in summary and N in summary[method]:
                s = summary[method][N]
                print(f"{N:<5} {method:<20} {s['mean_energy']:<10.1f} "
                      f"{s['std_energy']:<8.1f} {s['min_energy']:<8} "
                      f"{s['mean_time']:<12.2f}")
        
        if N in KNOWN_OPTIMA:
            opt_E = KNOWN_OPTIMA[N][0]
            # Calculate approximation ratio for best method
            if 's_tier' in summary and N in summary['s_tier']:
                best_E = summary['s_tier'][N]['min_energy']
                ratio = opt_E / best_E if best_E > 0 else 0
                print(f"{N:<5} {'OPTIMAL':<20} {opt_E:<10} {'-':<8} {'-':<8} "
                      f"Ratio: {ratio:.3f}")
        print()


def main():
    """Main benchmark runner."""
    print("S-Tier LABS Solver - Benchmark Runner")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    N_values = [8, 10, 12, 15, 18, 20]  # Adjust as needed
    runs = 3
    max_generations = 100  # Use 500-1000 for better results
    population_size = 30
    
    print(f"\nConfiguration:")
    print(f"  N values: {N_values}")
    print(f"  Runs per N: {runs}")
    print(f"  Max generations: {max_generations}")
    print(f"  Population size: {population_size}")
    
    # Run benchmarks
    all_results = {}
    
    for N in N_values:
        try:
            results = run_benchmark(
                N, 
                runs=runs,
                max_generations=max_generations,
                population_size=population_size,
                verbose=True
            )
            all_results[N] = results
        except Exception as e:
            print(f"\n✗ Error benchmarking N={N}: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate summary
    summary = calculate_summary(all_results, N_values)
    
    # Print results
    print_summary_table(summary, N_values)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_results(summary, N_values, 'benchmark_results.png')
    
    # Save data to JSON
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'N_values': N_values,
            'runs': runs,
            'max_generations': max_generations,
            'population_size': population_size
        },
        'results': all_results,
        'summary': summary,
        'known_optima': {str(k): v[0] for k, v in KNOWN_OPTIMA.items()}
    }
    
    with open('benchmark_data.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print("✓ Data saved to: benchmark_data.json")
    
    print("\n" + "="*60)
    print("Benchmark complete!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated files:")
    print("  - benchmark_results.png (plots)")
    print("  - benchmark_data.json (raw data)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
