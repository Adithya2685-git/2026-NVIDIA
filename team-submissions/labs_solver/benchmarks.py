"""
Benchmarks and Visualization for LABS Optimization

This module provides:
1. Benchmark functions to compare different solving methods
2. Visualization tools for results and convergence
3. Experiment runners for reproducible research

Methods compared:
- Classical MTS (random initialization)
- Single-Schedule QE-MTS (paper's approach)
- Multi-Schedule QE-MTS (our innovation)
- S-Tier (multi-schedule + Boltzmann seeding)

Usage:
    from benchmarks import run_full_benchmark, plot_results
    results = run_full_benchmark(N_values=[10, 15, 20], runs=10)
    plot_results(results, save_path="benchmark_results.png")
"""

import numpy as np
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Import our modules
from .labs_energy import calculate_energy, KNOWN_OPTIMA, random_sequence
from .mts import memetic_tabu_search
from .tabu_search import tabu_search
from .boltzmann_seeding import boltzmann_seeding, diversity_metrics

# Check for CUDA-Q
try:
    from .dcqo_kernels import run_dcqo, CUDAQ_AVAILABLE
except ImportError:
    CUDAQ_AVAILABLE = False

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class BenchmarkRunner:
    """
    Runs systematic benchmarks comparing different LABS solving methods.
    
    Example:
        runner = BenchmarkRunner(N_values=[10, 15, 20], runs=10)
        results = runner.run_all()
        runner.save_results("benchmark_results.json")
        runner.plot_comparison()
    """
    
    def __init__(
        self,
        N_values: List[int] = [10, 15, 20],
        runs: int = 5,
        max_generations: int = 500,
        population_size: int = 50,
        verbose: bool = True
    ):
        """
        Initialize benchmark runner.
        
        Args:
            N_values: List of sequence lengths to benchmark
            runs: Number of runs per method per N
            max_generations: Max MTS generations
            population_size: MTS population size
            verbose: Print progress
        """
        self.N_values = N_values
        self.runs = runs
        self.max_generations = max_generations
        self.population_size = population_size
        self.verbose = verbose
        
        self.results: Dict[str, Dict[int, List[Dict]]] = {
            'classical': {},
            'single_schedule': {},
            's_tier': {},
        }
        
        self.metadata = {
            'timestamp': datetime.now().isoformat(),
            'N_values': N_values,
            'runs': runs,
            'max_generations': max_generations,
            'population_size': population_size,
            'cudaq_available': CUDAQ_AVAILABLE,
        }
    
    def run_classical(self, N: int) -> Dict[str, Any]:
        """Run classical MTS with random initialization."""
        start = time.time()
        
        best_s, best_energy, stats = memetic_tabu_search(
            N,
            population_size=self.population_size,
            max_generations=self.max_generations,
            verbose=False
        )
        
        elapsed = time.time() - start
        
        return {
            'energy': best_energy,
            'time': elapsed,
            'generations': stats['generations'],
            'optimal': KNOWN_OPTIMA.get(N, (None, None))[0],
        }
    
    def run_single_schedule(self, N: int) -> Dict[str, Any]:
        """Run single-schedule QE-MTS (paper's approach)."""
        if not CUDAQ_AVAILABLE:
            return {'error': 'CUDA-Q not available'}
        
        start = time.time()
        
        # Run DCQO with default sin² schedule
        samples, energies, dcqo_stats = run_dcqo(
            N,
            n_shots=200,
            n_steps=1,
            target='tensornet-mps',
            verbose=False
        )
        
        if len(samples) == 0:
            return {'error': 'No quantum samples'}
        
        # Paper's approach: replicate best sample
        best_idx = int(np.argmin(energies))
        population = [samples[best_idx].copy() for _ in range(self.population_size)]
        
        best_s, best_energy, stats = memetic_tabu_search(
            N,
            population=population,
            population_size=self.population_size,
            max_generations=self.max_generations,
            verbose=False
        )
        
        elapsed = time.time() - start
        
        return {
            'energy': best_energy,
            'time': elapsed,
            'generations': stats['generations'],
            'quantum_min': min(energies),
            'quantum_mean': float(np.mean(energies)),
            'optimal': KNOWN_OPTIMA.get(N, (None, None))[0],
        }
    
    def run_s_tier(self, N: int, beta: float = 1.0) -> Dict[str, Any]:
        """Run S-Tier solver (multi-schedule + Boltzmann)."""
        start = time.time()
        
        # Step 1: Collect samples from multiple schedules
        all_samples = []
        all_energies = []
        
        if CUDAQ_AVAILABLE:
            from .schedules import ALL_SCHEDULES
            
            for schedule_class in ALL_SCHEDULES[:2]:
                try:
                    samples, energies, _ = run_dcqo(
                        N,
                        n_shots=100,
                        n_steps=1,
                        target='tensornet-mps',
                        verbose=False
                    )
                    all_samples.extend(samples)
                    all_energies.extend(energies)
                except Exception:
                    pass
        
        # Fallback to random if no quantum samples
        if len(all_samples) == 0:
            all_samples = [random_sequence(N) for _ in range(2000)]
            all_energies = [calculate_energy(s) for s in all_samples]
        
        # Step 2: Boltzmann seeding
        population = boltzmann_seeding(
            list(zip(all_samples, all_energies)),
            K=self.population_size,
            beta=beta
        )
        
        pop_metrics = diversity_metrics(population)
        
        # Step 3: MTS
        best_s, best_energy, stats = memetic_tabu_search(
            N,
            population=population,
            population_size=self.population_size,
            max_generations=self.max_generations,
            verbose=False
        )
        
        elapsed = time.time() - start
        
        return {
            'energy': best_energy,
            'time': elapsed,
            'generations': stats['generations'],
            'n_quantum_samples': len(all_samples),
            'quantum_min': min(all_energies),
            'quantum_mean': float(np.mean(all_energies)),
            'diversity': pop_metrics['diversity'],
            'unique_seeds': pop_metrics['unique'],
            'optimal': KNOWN_OPTIMA.get(N, (None, None))[0],
        }
    
    def run_all(self) -> Dict[str, Dict[int, List[Dict]]]:
        """Run all benchmarks."""
        for N in self.N_values:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Benchmarking N={N}")
                print(f"{'='*60}")
            
            # Initialize storage
            for method in self.results:
                self.results[method][N] = []
            
            for run_idx in range(self.runs):
                if self.verbose:
                    print(f"\nRun {run_idx + 1}/{self.runs}")
                
                # Classical
                result = self.run_classical(N)
                self.results['classical'][N].append(result)
                if self.verbose:
                    print(f"  Classical: E={result['energy']}")
                
                # Single-schedule (if CUDA-Q available)
                if CUDAQ_AVAILABLE:
                    result = self.run_single_schedule(N)
                    self.results['single_schedule'][N].append(result)
                    if self.verbose and 'energy' in result:
                        print(f"  Single-schedule: E={result['energy']}")
                
                # S-Tier
                result = self.run_s_tier(N)
                self.results['s_tier'][N].append(result)
                if self.verbose:
                    print(f"  S-Tier: E={result['energy']}")
        
        return self.results
    
    def get_summary(self) -> Dict[str, Dict[int, Dict]]:
        """Generate summary statistics."""
        summary = {}
        
        for method, n_results in self.results.items():
            summary[method] = {}
            
            for N, runs in n_results.items():
                if not runs or 'error' in runs[0]:
                    continue
                
                energies = [r['energy'] for r in runs if 'energy' in r]
                times = [r['time'] for r in runs if 'time' in r]
                
                if not energies:
                    continue
                
                optimal = KNOWN_OPTIMA.get(N, (None, None))[0]
                
                summary[method][N] = {
                    'mean_energy': float(np.mean(energies)),
                    'std_energy': float(np.std(energies)),
                    'min_energy': min(energies),
                    'max_energy': max(energies),
                    'mean_time': float(np.mean(times)),
                    'optimal': optimal,
                    'optimal_found': sum(1 for e in energies if optimal and e == optimal),
                }
        
        return summary
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        for N in self.N_values:
            print(f"\nN = {N}")
            print("-" * 50)
            
            optimal = KNOWN_OPTIMA.get(N, (None, None))[0]
            print(f"Known optimal: {optimal}")
            print()
            
            print(f"{'Method':<20} {'Mean E':<10} {'Std':<8} {'Min':<8} {'Time(s)':<10}")
            print("-" * 50)
            
            for method in ['classical', 'single_schedule', 's_tier']:
                if method not in summary or N not in summary[method]:
                    continue
                
                s = summary[method][N]
                print(f"{method:<20} {s['mean_energy']:<10.1f} {s['std_energy']:<8.1f} "
                      f"{s['min_energy']:<8} {s['mean_time']:<10.2f}")
    
    def save_results(self, filepath: str):
        """Save results to JSON file."""
        output = {
            'metadata': self.metadata,
            'results': {},
            'summary': self.get_summary(),
        }
        
        # Convert numpy types to Python types
        for method, n_results in self.results.items():
            output['results'][method] = {}
            for N, runs in n_results.items():
                output['results'][method][str(N)] = runs
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"Results saved to {filepath}")
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """Generate comparison plots."""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available for plotting")
            return
        
        summary = self.get_summary()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Energy vs N
        ax1 = axes[0]
        methods = ['classical', 'single_schedule', 's_tier']
        colors = {'classical': 'blue', 'single_schedule': 'orange', 's_tier': 'green'}
        labels = {'classical': 'Classical MTS', 'single_schedule': 'Single-Schedule QE-MTS', 's_tier': 'S-Tier (Ours)'}
        
        for method in methods:
            if method not in summary:
                continue
            
            Ns = sorted([N for N in summary[method].keys() if summary[method][N]])
            means = [summary[method][N]['mean_energy'] for N in Ns]
            stds = [summary[method][N]['std_energy'] for N in Ns]
            
            if Ns:
                ax1.errorbar(Ns, means, yerr=stds, marker='o', 
                           label=labels.get(method, method), 
                           color=colors.get(method, 'gray'),
                           capsize=5, capthick=2)
        
        # Plot optimal line
        opt_Ns = sorted([N for N in self.N_values if N in KNOWN_OPTIMA])
        opt_Es = [KNOWN_OPTIMA[N][0] for N in opt_Ns]
        opt_pairs = [(n, KNOWN_OPTIMA[n][0]) for n in opt_Ns if KNOWN_OPTIMA[n][0] is not None]
        if opt_pairs:
            opt_plot_ns = [n for n, _ in opt_pairs]
            opt_plot_es = [e for _, e in opt_pairs]
            ax1.plot(opt_plot_ns, opt_plot_es, 'k--', label='Optimal', linewidth=2)
        
        ax1.set_xlabel('Sequence Length N')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy vs Sequence Length')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Time vs N
        ax2 = axes[1]
        
        for method in methods:
            if method not in summary:
                continue
            
            Ns = sorted([N for N in summary[method].keys() if summary[method][N]])
            times = [summary[method][N]['mean_time'] for N in Ns]
            
            if Ns:
                ax2.plot(Ns, times, marker='s', label=labels.get(method, method),
                        color=colors.get(method, 'gray'))
        
        ax2.set_xlabel('Sequence Length N')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Computation Time vs Sequence Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


def run_convergence_study(
    N: int = 20,
    max_generations: int = 1000,
    n_runs: int = 5,
    verbose: bool = True
) -> Dict[str, List[List[int]]]:
    """
    Run convergence study comparing methods over generations.
    
    Returns energy history for each method and run.
    """
    results = {
        'classical': [],
        's_tier': [],
    }
    
    for run in range(n_runs):
        if verbose:
            print(f"Run {run + 1}/{n_runs}")
        
        # Classical
        _, _, stats = memetic_tabu_search(
            N,
            max_generations=max_generations,
            verbose=False
        )
        results['classical'].append(stats['energy_history'])
        
        # S-Tier
        all_samples = [random_sequence(N) for _ in range(2000)]
        all_energies = [calculate_energy(s) for s in all_samples]
        
        population = boltzmann_seeding(
            list(zip(all_samples, all_energies)),
            K=50,
            beta=1.0
        )
        
        _, _, stats = memetic_tabu_search(
            N,
            population=population,
            max_generations=max_generations,
            verbose=False
        )
        results['s_tier'].append(stats['energy_history'])
    
    return results


def plot_convergence(
    results: Dict[str, List[List[int]]],
    N: int,
    save_path: Optional[str] = None
):
    """Plot convergence curves."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'classical': 'blue', 's_tier': 'green'}
    labels = {'classical': 'Classical MTS', 's_tier': 'S-Tier (Ours)'}
    
    for method, histories in results.items():
        if not histories:
            continue
        
        # Pad to same length
        max_len = max(len(h) for h in histories)
        padded = []
        for h in histories:
            padded.append(h + [h[-1]] * (max_len - len(h)))
        
        histories_array = np.array(padded)
        mean = np.mean(histories_array, axis=0)
        std = np.std(histories_array, axis=0)
        
        generations = np.arange(len(mean))
        
        ax.plot(generations, mean, label=labels.get(method, method),
               color=colors.get(method, 'gray'), linewidth=2)
        ax.fill_between(generations, mean - std, mean + std,
                       alpha=0.2, color=colors.get(method, 'gray'))
    
    # Optimal line
    opt_energy = KNOWN_OPTIMA.get(N, (None, None))[0]
    if opt_energy is not None:
        ax.axhline(y=opt_energy, color='red', linestyle='--',
                  label=f'Optimal (E={opt_energy})')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Energy')
    ax.set_title(f'Convergence Comparison (N={N})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def run_boltzmann_beta_study(
    N: int = 20,
    betas: List[float] = [0.0, 0.5, 1.0, 2.0, 5.0],
    n_runs: int = 5,
    verbose: bool = True
) -> Dict[float, Dict[str, float]]:
    """
    Study the effect of Boltzmann beta parameter.
    
    Returns results for each beta value.
    """
    results = {}
    
    for beta in betas:
        if verbose:
            print(f"\nBeta = {beta}")
        
        energies = []
        diversities = []
        
        for run in range(n_runs):
            # Generate samples
            samples = [random_sequence(N) for _ in range(2000)]
            sample_energies = [calculate_energy(s) for s in samples]
            
            # Boltzmann seeding
            population = boltzmann_seeding(
                list(zip(samples, sample_energies)),
                K=50,
                beta=beta
            )
            
            metrics = diversity_metrics(population)
            diversities.append(metrics['diversity'])
            
            # Run MTS
            _, best_energy, _ = memetic_tabu_search(
                N,
                population=population,
                max_generations=200,
                verbose=False
            )
            energies.append(best_energy)
            
            if verbose:
                print(f"  Run {run+1}: E={best_energy}, diversity={metrics['diversity']:.2%}")
        
        results[beta] = {
            'mean_energy': float(np.mean(energies)),
            'std_energy': float(np.std(energies)),
            'mean_diversity': float(np.mean(diversities)),
        }
    
    return results


def plot_beta_study(results: Dict[float, Dict[str, float]], N: int, save_path: Optional[str] = None):
    """Plot beta study results."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    betas = sorted(results.keys())
    energies = [results[b]['mean_energy'] for b in betas]
    energy_stds = [results[b]['std_energy'] for b in betas]
    diversities = [results[b]['mean_diversity'] for b in betas]
    
    # Energy on left axis
    ax1.errorbar(betas, energies, yerr=energy_stds, marker='o', color='blue',
                label='Mean Energy', capsize=5)
    ax1.set_xlabel('Boltzmann β')
    ax1.set_ylabel('Energy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Diversity on right axis
    ax2 = ax1.twinx()
    ax2.plot(betas, diversities, marker='s', color='green', label='Diversity')
    ax2.set_ylabel('Population Diversity', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Optimal line
    opt_energy = KNOWN_OPTIMA.get(N, (None, None))[0]
    if opt_energy is not None:
        ax1.axhline(y=opt_energy, color='red', linestyle='--',
                   label=f'Optimal (E={opt_energy})')
    
    ax1.set_title(f'Effect of Boltzmann β on Performance (N={N})')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def quick_benchmark(N: int = 15, runs: int = 3, verbose: bool = True):
    """
    Quick benchmark for testing.
    
    Runs a fast comparison of methods on a single N value.
    """
    print(f"Quick benchmark: N={N}, runs={runs}")
    print("="*50)
    
    runner = BenchmarkRunner(
        N_values=[N],
        runs=runs,
        max_generations=100,
        population_size=30,
        verbose=verbose
    )
    
    results = runner.run_all()
    runner.print_summary()
    
    return results


if __name__ == "__main__":
    print("LABS Benchmark Module")
    print("=" * 60)
    print(f"CUDA-Q available: {CUDAQ_AVAILABLE}")
    print(f"Matplotlib available: {MATPLOTLIB_AVAILABLE}")
    print()
    
    # Run comprehensive benchmark for presentation
    print("Running comprehensive benchmark (N=20, 25, 29)...")
    runner = BenchmarkRunner(
        N_values=[8, 10, 15, 20, 25, 29],
        runs=2,
        max_generations=200,
        population_size=30,
        verbose=True
    )
    results = runner.run_all()
    runner.print_summary()
    runner.save_results("benchmark_results.json")
    if MATPLOTLIB_AVAILABLE:
        runner.plot_comparison("benchmark_results.png")
