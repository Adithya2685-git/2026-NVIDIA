"""
Boltzmann-Weighted Population Seeding for Quantum-Enhanced MTS

This module implements our S-Tier innovation: Boltzmann seeding.

The Problem:
The paper uses the BEST quantum sample and replicates it K times.
This creates a homogeneous population with low diversity.

Our Solution:
Instead of P(s) = 1 if s is best, else 0
We use:  P(s) ∝ exp(-β × E(s))

This:
1. Prefers low-energy samples (quality)
2. Still includes higher-energy samples (diversity)
3. β controls the trade-off:
   - β → 0: uniform (max diversity)
   - β → ∞: greedy (always pick best)
   - β ≈ 1: balanced

Why Diversity Matters:
- MTS uses population for recombination
- Diverse population explores more of the landscape
- Low diversity → premature convergence
"""

import numpy as np
from typing import List, Tuple, Union, Optional
from .labs_energy import calculate_energy, Sequence


def boltzmann_weights(energies: Union[List[int], np.ndarray], beta: float = 1.0) -> np.ndarray:
    """
    Calculate Boltzmann weights for given energies.
    
    P(E) ∝ exp(-β × E)
    
    Weights are normalized to sum to 1.
    
    Args:
        energies: List/array of energy values
        beta: Inverse temperature parameter
              - beta → 0: uniform distribution
              - beta → ∞: greedy (all weight on minimum)
              - beta ≈ 1: balanced
    
    Returns:
        np.ndarray: Normalized probability weights
    """
    energies = np.array(energies, dtype=np.float64)
    
    if len(energies) == 0:
        return np.array([])
    
    if len(energies) == 1:
        return np.array([1.0])
    
    # Shift energies to prevent numerical overflow
    # exp(-β × (E - E_min)) = exp(-β × E) / exp(-β × E_min)
    E_min = energies.min()
    shifted = energies - E_min
    
    # Boltzmann factor
    log_weights = -beta * shifted
    
    # Softmax-style normalization for numerical stability
    max_log = log_weights.max()
    weights = np.exp(log_weights - max_log)
    weights /= weights.sum()
    
    return weights


def boltzmann_seeding(
    samples: List[Union[np.ndarray, Tuple[np.ndarray, int]]],
    K: int = 100,
    beta: float = 1.0,
    unique_only: bool = False
) -> List[np.ndarray]:
    """
    Create diverse population using Boltzmann-weighted sampling.
    
    S-Tier Innovation: Instead of replicating the best sample K times,
    we sample K individuals weighted by their energy quality.
    
    Args:
        samples: List of sequences (or (sequence, energy) tuples)
        K: Population size (number of individuals to sample)
        beta: Temperature parameter
              - beta = 0.0: uniform sampling (maximum diversity)
              - beta = 1.0: balanced (recommended)
              - beta > 5.0: very greedy (low diversity)
              - beta = inf: greedy baseline (best sample only)
        unique_only: If True, only sample unique sequences
    
    Returns:
        List of K sequences (as numpy arrays)
    """
    if len(samples) == 0:
        raise ValueError("samples cannot be empty")
    
    # Extract sequences and energies
    if isinstance(samples[0], tuple):
        sequences = [np.array(s[0], dtype=np.int8) for s in samples]
        energies = [s[1] for s in samples]
    else:
        sequences = [np.array(s, dtype=np.int8) for s in samples]
        energies = [calculate_energy(s) for s in sequences]
    
    # Handle edge case: very high beta (greedy)
    if beta > 100 or beta == float('inf'):
        # Just replicate the best sample
        best_idx = np.argmin(energies)
        return [sequences[best_idx].copy() for _ in range(K)]
    
    # Calculate Boltzmann weights
    weights = boltzmann_weights(energies, beta)
    
    # Sample K individuals with replacement
    indices = np.random.choice(len(sequences), size=K, p=weights, replace=True)
    
    # Create population
    population = [sequences[i].copy() for i in indices]
    
    return population


def adaptive_beta_seeding(
    samples: List[Union[np.ndarray, Tuple[np.ndarray, int]]],
    K: int = 100,
    target_diversity: float = 0.5
) -> Tuple[List[np.ndarray], float]:
    """
    Automatically choose β to achieve target diversity.
    
    Diversity = (number of unique sequences) / K
    
    Uses binary search to find optimal β.
    
    Args:
        samples: List of sequences (or (sequence, energy) tuples)
        K: Population size
        target_diversity: Target fraction of unique sequences (0 to 1)
    
    Returns:
        Tuple of (population, beta_used)
    """
    beta_low, beta_high = 0.01, 10.0
    
    # Binary search for optimal beta
    for _ in range(15):  # 15 iterations gives good precision
        beta_mid = (beta_low + beta_high) / 2
        
        # Sample population with current beta
        population = boltzmann_seeding(samples, K, beta_mid)
        
        # Calculate diversity
        unique = len(set(tuple(s.tolist()) for s in population))
        diversity = unique / K
        
        if diversity < target_diversity:
            # Too greedy, reduce beta
            beta_high = beta_mid
        else:
            # Too diverse (or just right), increase beta
            beta_low = beta_mid
    
    # Final sampling with found beta
    final_beta = (beta_low + beta_high) / 2
    population = boltzmann_seeding(samples, K, final_beta)
    
    return population, final_beta


def diversity_metrics(population: List[np.ndarray]) -> dict:
    """
    Calculate diversity metrics for a population.
    
    Args:
        population: List of sequences
    
    Returns:
        Dictionary with diversity metrics
    """
    K = len(population)
    
    if K == 0:
        return {'unique': 0, 'diversity': 0, 'mean_hamming': 0}
    
    # Number of unique sequences
    unique_seqs = set(tuple(s.tolist()) for s in population)
    n_unique = len(unique_seqs)
    
    # Diversity ratio
    diversity = n_unique / K
    
    # Average pairwise Hamming distance (sample if population is large)
    if K <= 100:
        # Compute all pairs
        total_dist = 0
        n_pairs = 0
        for i in range(K):
            for j in range(i + 1, K):
                dist = np.sum(population[i] != population[j])
                total_dist += dist
                n_pairs += 1
        mean_hamming = total_dist / n_pairs if n_pairs > 0 else 0
    else:
        # Sample pairs
        n_sample = 1000
        distances = []
        for _ in range(n_sample):
            i, j = np.random.choice(K, 2, replace=False)
            dist = np.sum(population[i] != population[j])
            distances.append(dist)
        mean_hamming = np.mean(distances)
    
    N = len(population[0]) if K > 0 else 0
    normalized_hamming = mean_hamming / N if N > 0 else 0
    
    return {
        'unique': n_unique,
        'diversity': diversity,
        'mean_hamming': mean_hamming,
        'normalized_hamming': normalized_hamming,
        'K': K,
        'N': N,
    }


def compare_seeding_methods(
    samples: List[np.ndarray],
    K: int = 100,
    beta: float = 1.0
) -> dict:
    """
    Compare Boltzmann seeding vs paper's greedy approach.
    
    Args:
        samples: Quantum samples
        K: Population size
        beta: Boltzmann temperature
    
    Returns:
        Dictionary comparing methods
    """
    # Paper's approach: best sample replicated K times
    energies = [calculate_energy(s) for s in samples]
    best_idx = np.argmin(energies)
    greedy_pop = [samples[best_idx].copy() for _ in range(K)]
    
    # Our approach: Boltzmann seeding
    boltzmann_pop = boltzmann_seeding(
        list(zip(samples, energies)), K, beta
    )
    
    # Metrics
    greedy_metrics = diversity_metrics(greedy_pop)
    boltzmann_metrics = diversity_metrics(boltzmann_pop)
    
    greedy_energies = [calculate_energy(s) for s in greedy_pop]
    boltzmann_energies = [calculate_energy(s) for s in boltzmann_pop]
    
    return {
        'greedy': {
            **greedy_metrics,
            'mean_energy': np.mean(greedy_energies),
            'min_energy': np.min(greedy_energies),
        },
        'boltzmann': {
            **boltzmann_metrics,
            'mean_energy': np.mean(boltzmann_energies),
            'min_energy': np.min(boltzmann_energies),
            'beta': beta,
        },
    }


def visualize_boltzmann(energies: List[int], betas: List[float] = None):
    """
    Visualize Boltzmann weights for different β values.
    
    Args:
        energies: List of sample energies
        betas: List of β values to compare
    
    Returns:
        matplotlib figure or None
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return None
    
    if betas is None:
        betas = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    energies = np.array(energies)
    sorted_idx = np.argsort(energies)
    sorted_energies = energies[sorted_idx]
    
    fig, axes = plt.subplots(1, len(betas), figsize=(3 * len(betas), 4))
    
    for ax, beta in zip(axes, betas):
        weights = boltzmann_weights(sorted_energies, beta)
        ax.bar(range(len(weights)), weights)
        ax.set_title(f'β = {beta}')
        ax.set_xlabel('Sample (sorted by energy)')
        ax.set_ylabel('Probability')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Quick test
    print("Testing Boltzmann Seeding...")
    
    # Create synthetic samples with varying energies
    np.random.seed(42)
    N = 10
    n_samples = 100
    
    # Generate samples
    samples = []
    for _ in range(n_samples):
        s = np.random.choice([-1, 1], N).astype(np.int8)
        samples.append(s)
    
    energies = [calculate_energy(s) for s in samples]
    
    print(f"Samples: {len(samples)}")
    print(f"Energy range: {min(energies)} to {max(energies)}")
    print(f"Mean energy: {np.mean(energies):.1f}")
    
    # Test different betas
    print("\nBeta comparison:")
    for beta in [0.1, 0.5, 1.0, 2.0, 5.0]:
        population = boltzmann_seeding(
            list(zip(samples, energies)),
            K=50,
            beta=beta
        )
        metrics = diversity_metrics(population)
        pop_energies = [calculate_energy(s) for s in population]
        print(f"  β={beta}: diversity={metrics['diversity']:.2%}, "
              f"mean_E={np.mean(pop_energies):.1f}, min_E={min(pop_energies)}")
    
    # Compare methods
    print("\n\nMethod comparison:")
    comparison = compare_seeding_methods(samples, K=50, beta=1.0)
    print(f"Greedy: diversity={comparison['greedy']['diversity']:.2%}, "
          f"mean_E={comparison['greedy']['mean_energy']:.1f}")
    print(f"Boltzmann: diversity={comparison['boltzmann']['diversity']:.2%}, "
          f"mean_E={comparison['boltzmann']['mean_energy']:.1f}")
