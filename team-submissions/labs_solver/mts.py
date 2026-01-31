"""
Memetic Tabu Search (MTS) for LABS Optimization

MTS is a hybrid metaheuristic that combines:
1. Evolutionary algorithms (population, selection, crossover, mutation)
2. Local search (tabu search for intensification)

This is the state-of-the-art classical algorithm for LABS with O(1.34^N) scaling.

Algorithm:
1. Initialize population (random or quantum-seeded)
2. For each generation:
   - Select parents via tournament
   - Combine or select random individual
   - Mutate with probability p_mut = 1/N
   - Apply tabu search to offspring
   - Update best and replace in population
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
import time

from .labs_energy import calculate_energy, Sequence, random_sequence
from .tabu_search import tabu_search


def combine(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Combine two parent sequences using uniform crossover.
    
    Each position is randomly chosen from either parent with 50% probability.
    This is the standard crossover operator used in the paper.
    
    Args:
        p1: First parent sequence
        p2: Second parent sequence
    
    Returns:
        np.ndarray: Child sequence
    """
    N = len(p1)
    mask = np.random.randint(0, 2, N, dtype=np.int8)
    child = np.where(mask, p1, p2).astype(np.int8)
    return child


def mutate(s: np.ndarray, p_mut: float) -> np.ndarray:
    """
    Mutate sequence with probability p_mut per position.
    
    Each bit is independently flipped with probability p_mut.
    Standard value is p_mut = 1/N.
    
    Args:
        s: Input sequence
        p_mut: Per-position mutation probability
    
    Returns:
        np.ndarray: Mutated sequence (copy)
    """
    s = s.copy()
    mask = np.random.random(len(s)) < p_mut
    s[mask] *= -1
    return s


def tournament_select(
    population: List[np.ndarray],
    energies: List[int],
    tournament_size: int = 2
) -> np.ndarray:
    """
    Select an individual via tournament selection.
    
    Randomly choose tournament_size individuals and return the best one.
    
    Args:
        population: List of sequences
        energies: List of corresponding energies
        tournament_size: Number of individuals in tournament
    
    Returns:
        np.ndarray: Selected individual (copy)
    """
    indices = np.random.choice(len(population), tournament_size, replace=False)
    best_idx = indices[np.argmin([energies[i] for i in indices])]
    return population[best_idx].copy()


def memetic_tabu_search(
    N: int,
    population: Optional[List[np.ndarray]] = None,
    population_size: int = 100,
    p_comb: float = 0.9,
    p_mut: Optional[float] = None,
    max_generations: int = 1000,
    target_energy: Optional[int] = None,
    tabu_iterations: int = 100,
    tabu_tenure: int = 7,
    tournament_size: int = 2,
    verbose: bool = False,
    callback: Optional[Callable[[int, int, List[int]], None]] = None
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Memetic Tabu Search for LABS optimization.
    
    This is the main MTS algorithm from the paper. It maintains a population
    of solutions, evolves them through crossover and mutation, and applies
    tabu search for local optimization.
    
    Args:
        N: Sequence length
        population: Initial population (or None for random initialization)
        population_size: K in paper (default 100)
        p_comb: Probability of combination vs random selection (default 0.9)
        p_mut: Per-bit mutation rate (default 1/N)
        max_generations: Maximum number of generations
        target_energy: Stop early if this energy is achieved
        tabu_iterations: Max iterations for each tabu search
        tabu_tenure: Tabu tenure parameter
        tournament_size: Size of tournament for parent selection
        verbose: Print progress information
        callback: Optional callback(generation, best_energy, all_energies)
    
    Returns:
        Tuple of:
        - best_s: Best sequence found
        - best_energy: Energy of best sequence
        - stats: Dictionary with statistics
    """
    if p_mut is None:
        p_mut = 1.0 / N
    
    # Initialize population
    if population is None:
        population = [random_sequence(N) for _ in range(population_size)]
    else:
        # Ensure we have exactly population_size individuals
        population = [np.array(p, dtype=np.int8) for p in population]
        while len(population) < population_size:
            population.append(random_sequence(N))
        population = population[:population_size]
    
    # Calculate initial energies
    energies = [calculate_energy(p) for p in population]
    
    # Track best solution
    best_idx = int(np.argmin(energies))
    best_s = population[best_idx].copy()
    best_energy = energies[best_idx]
    
    # Statistics tracking
    start_time = time.time()
    total_evaluations = len(population)  # Initial evaluation count
    generations_completed = 0
    energy_history = [best_energy]
    
    if verbose:
        print(f"MTS initialized: population_size={population_size}, p_comb={p_comb}, p_mut={p_mut:.4f}")
        print(f"Initial best energy: {best_energy}")
    
    for gen in range(max_generations):
        # Step 1: Select or combine to create offspring
        if np.random.random() < p_comb:
            # Combine two parents
            p1 = tournament_select(population, energies, tournament_size)
            p2 = tournament_select(population, energies, tournament_size)
            child = combine(p1, p2)
        else:
            # Select random individual
            idx = np.random.randint(len(population))
            child = population[idx].copy()
        
        # Step 2: Mutate
        child = mutate(child, p_mut)
        
        # Step 3: Local search with tabu
        child, child_energy = tabu_search(
            child,
            max_iterations=tabu_iterations,
            tabu_tenure=tabu_tenure,
            verbose=False
        )
        total_evaluations += tabu_iterations  # Approximate
        
        # Step 4: Update best if improved
        if child_energy < best_energy:
            best_s = child.copy()
            best_energy = child_energy
            
            if verbose:
                print(f"Generation {gen}: New best energy = {best_energy}")
        
        # Step 5: Replace random individual in population
        replace_idx = np.random.randint(len(population))
        population[replace_idx] = child
        energies[replace_idx] = child_energy
        
        # Bookkeeping
        generations_completed = gen + 1
        energy_history.append(best_energy)
        
        # Callback for external monitoring
        if callback is not None:
            callback(gen, best_energy, energies.copy())
        
        # Early termination if target reached
        if target_energy is not None and best_energy <= target_energy:
            if verbose:
                print(f"Target energy {target_energy} reached at generation {gen}")
            break
    
    elapsed_time = time.time() - start_time
    
    # Compile statistics
    stats = {
        'N': N,
        'population_size': population_size,
        'generations': generations_completed,
        'evaluations': total_evaluations,
        'time_seconds': elapsed_time,
        'initial_best': energy_history[0],
        'final_best': best_energy,
        'energy_history': energy_history,
        'final_population_energies': energies,
        'final_population_mean': np.mean(energies),
        'final_population_std': np.std(energies),
    }
    
    return best_s, best_energy, stats


def quantum_enhanced_mts(
    N: int,
    quantum_samples: List[np.ndarray],
    population_size: int = 100,
    **mts_kwargs
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Quantum-Enhanced MTS: Seed MTS with quantum-generated samples.
    
    This is the QE-MTS algorithm from the paper. Instead of random
    initialization, we use samples from a quantum algorithm.
    
    Args:
        N: Sequence length
        quantum_samples: List of sequences from quantum sampling
        population_size: Size of MTS population
        **mts_kwargs: Additional arguments for memetic_tabu_search
    
    Returns:
        Same as memetic_tabu_search
    """
    if len(quantum_samples) == 0:
        raise ValueError("quantum_samples cannot be empty")
    
    # Use quantum samples as initial population
    # If we have fewer samples than population_size, replicate
    population = []
    for i in range(population_size):
        idx = i % len(quantum_samples)
        population.append(np.array(quantum_samples[idx], dtype=np.int8))
    
    return memetic_tabu_search(N, population=population, population_size=population_size, **mts_kwargs)


def mts_with_monitoring(
    N: int,
    **kwargs
) -> Tuple[np.ndarray, int, Dict[str, Any], List[Dict]]:
    """
    MTS with detailed monitoring of each generation.
    
    Returns additional detailed history for analysis.
    """
    history = []
    
    def monitor_callback(gen, best_energy, energies):
        history.append({
            'generation': gen,
            'best_energy': best_energy,
            'mean_energy': np.mean(energies),
            'min_energy': min(energies),
            'max_energy': max(energies),
            'std_energy': np.std(energies),
        })
    
    result = memetic_tabu_search(N, callback=monitor_callback, **kwargs)
    return result[0], result[1], result[2], history


if __name__ == "__main__":
    # Quick test
    print("Testing Memetic Tabu Search...")
    
    # Test on N=15
    N = 15
    
    print(f"\nRunning MTS on N={N}...")
    best_s, best_energy, stats = memetic_tabu_search(
        N,
        population_size=50,
        max_generations=100,
        verbose=True
    )
    
    print(f"\nFinal Results:")
    print(f"Best energy: {best_energy}")
    print(f"Generations: {stats['generations']}")
    print(f"Time: {stats['time_seconds']:.2f}s")
    print(f"Population mean: {stats['final_population_mean']:.1f}")
    print(f"Population std: {stats['final_population_std']:.1f}")
