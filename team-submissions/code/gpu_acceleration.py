"""
GPU Acceleration Module for LABS Optimization

This module provides CuPy-accelerated implementations of key operations
for the Memetic Tabu Search algorithm, enabling significant speedups
on NVIDIA GPUs.

Key accelerated operations:
1. Batched energy calculation (FFT-based autocorrelation)
2. Batched crossover operations
3. Batched mutation operations
4. Batched neighbor evaluation for tabu search

Usage:
    from gpu_acceleration import GPUAccelerator
    
    gpu = GPUAccelerator()
    energies = gpu.batch_energy(sequences)  # Fast batched energy
"""

import numpy as np
from typing import List, Tuple, Optional, Union

# Try to import CuPy, fall back to NumPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = np  # Use NumPy as fallback
    CUPY_AVAILABLE = False


class GPUAccelerator:
    """
    GPU-accelerated operations for LABS optimization.
    
    Falls back to NumPy if CuPy is not available, maintaining
    API compatibility across platforms.
    
    Example:
        gpu = GPUAccelerator()
        
        # Batch energy calculation
        sequences = [np.array([1, 1, -1, 1, -1], dtype=np.int8) for _ in range(100)]
        energies = gpu.batch_energy(sequences)
        
        # Batch crossover
        children = gpu.batch_crossover(parents1, parents2)
    """
    
    def __init__(self, device_id: int = 0, use_gpu: bool = True):
        """
        Initialize the GPU accelerator.
        
        Args:
            device_id: CUDA device ID to use
            use_gpu: Whether to attempt GPU acceleration
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.device_id = device_id
        
        if self.use_gpu:
            try:
                cp.cuda.Device(device_id).use()
                self.xp = cp
                self.backend = 'cupy'
            except Exception:
                self.xp = np
                self.backend = 'numpy'
                self.use_gpu = False
        else:
            self.xp = np
            self.backend = 'numpy'
    
    @property
    def is_gpu(self) -> bool:
        """Check if GPU acceleration is active."""
        return self.use_gpu and self.backend == 'cupy'
    
    def to_device(self, arr: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
        """Transfer array to GPU if available."""
        if self.is_gpu:
            return cp.asarray(arr)
        return arr
    
    def to_host(self, arr: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
        """Transfer array to CPU."""
        if self.is_gpu and hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)
    
    def batch_energy(self, sequences: List[np.ndarray]) -> np.ndarray:
        """
        Calculate LABS energy for a batch of sequences using FFT.
        
        Uses the autocorrelation theorem: autocorr = ifft(fft(s) * conj(fft(s)))
        This is more efficient for batched computation than the O(N^2) direct method.
        
        Args:
            sequences: List of binary sequences (each with +1/-1 values)
        
        Returns:
            np.ndarray: Array of energy values for each sequence
        """
        if len(sequences) == 0:
            return np.array([], dtype=np.int32)
        
        # Stack sequences into a batch
        N = len(sequences[0])
        batch_size = len(sequences)
        
        # Convert to array and transfer to device
        batch = np.stack([np.asarray(s, dtype=np.float64) for s in sequences])
        batch = self.to_device(batch)
        
        # FFT-based autocorrelation (vectorized across batch)
        # Pad to 2N for linear (not circular) autocorrelation
        f = self.xp.fft.fft(batch, n=2*N, axis=1)
        autocorr = self.xp.fft.ifft(f * self.xp.conj(f), axis=1).real[:, :N]
        
        # Energy = sum of C_k^2 for k=1 to N-1
        # autocorr[:, 0] is trivial (= N), skip it
        energies = self.xp.sum(autocorr[:, 1:] ** 2, axis=1)
        
        # Transfer back and convert to int
        energies = self.to_host(energies)
        return np.round(energies).astype(np.int32)
    
    def batch_energy_direct(self, sequences: List[np.ndarray]) -> np.ndarray:
        """
        Calculate LABS energy using direct O(N^2) method (batched).
        
        This is more accurate for small N and avoids FFT rounding issues.
        
        Args:
            sequences: List of binary sequences
        
        Returns:
            np.ndarray: Array of energy values
        """
        if len(sequences) == 0:
            return np.array([], dtype=np.int32)
        
        N = len(sequences[0])
        batch = np.stack([np.asarray(s, dtype=np.int32) for s in sequences])
        batch = self.to_device(batch)
        
        energies = self.xp.zeros(len(sequences), dtype=self.xp.int32)
        
        for k in range(1, N):
            # C_k = sum of s[i] * s[i+k] for each sequence
            C_k = self.xp.sum(batch[:, :N-k] * batch[:, k:], axis=1)
            energies += C_k * C_k
        
        return self.to_host(energies)
    
    def batch_crossover(
        self,
        parents1: List[np.ndarray],
        parents2: List[np.ndarray],
        mask: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        Perform uniform crossover on batches of parent pairs.
        
        Each position is randomly chosen from either parent with 50% probability.
        
        Args:
            parents1: List of first parents
            parents2: List of second parents
            mask: Optional predefined mask (for reproducibility)
        
        Returns:
            List of child sequences
        """
        if len(parents1) == 0:
            return []
        
        N = len(parents1[0])
        batch_size = len(parents1)
        
        p1 = np.stack([np.asarray(p, dtype=np.int8) for p in parents1])
        p2 = np.stack([np.asarray(p, dtype=np.int8) for p in parents2])
        
        p1 = self.to_device(p1)
        p2 = self.to_device(p2)
        
        # Generate random mask if not provided
        if mask is None:
            mask = self.xp.random.randint(0, 2, (batch_size, N), dtype=self.xp.int8)
        else:
            mask = self.to_device(mask)
        
        # Apply crossover: child[i] = p1[i] if mask[i] else p2[i]
        children = self.xp.where(mask, p1, p2)
        
        # Transfer back and split into list
        children = self.to_host(children).astype(np.int8)
        return [children[i] for i in range(batch_size)]
    
    def batch_mutate(
        self,
        sequences: List[np.ndarray],
        p_mut: float
    ) -> List[np.ndarray]:
        """
        Apply mutation to a batch of sequences.
        
        Each bit is independently flipped with probability p_mut.
        
        Args:
            sequences: List of sequences to mutate
            p_mut: Per-bit mutation probability
        
        Returns:
            List of mutated sequences (copies)
        """
        if len(sequences) == 0:
            return []
        
        N = len(sequences[0])
        batch_size = len(sequences)
        
        batch = np.stack([np.asarray(s, dtype=np.int8) for s in sequences])
        batch = self.to_device(batch)
        
        # Generate mutation mask
        mutation_mask = self.xp.random.random((batch_size, N)) < p_mut
        
        # Apply mutations (flip = multiply by -1)
        batch = self.xp.where(mutation_mask, -batch, batch)
        
        # Transfer back
        batch = self.to_host(batch).astype(np.int8)
        return [batch[i] for i in range(batch_size)]
    
    def batch_neighbor_energies(self, s: np.ndarray) -> np.ndarray:
        """
        Evaluate energies of all single-flip neighbors (GPU-accelerated).
        
        For a sequence of length N, this evaluates N neighbors in parallel
        on the GPU, which is much faster than sequential evaluation.
        
        Args:
            s: Current sequence
        
        Returns:
            np.ndarray: Array of energies for each neighbor (flip at position i)
        """
        N = len(s)
        s = np.asarray(s, dtype=np.int8)
        
        # Create all N neighbors at once
        # neighbors[i] = s with position i flipped
        neighbors = np.tile(s, (N, 1))
        flip_indices = np.arange(N)
        neighbors[flip_indices, flip_indices] *= -1
        
        # Batch energy calculation
        neighbors = self.to_device(neighbors.astype(np.float64))
        
        # FFT-based autocorrelation
        f = self.xp.fft.fft(neighbors, n=2*N, axis=1)
        autocorr = self.xp.fft.ifft(f * self.xp.conj(f), axis=1).real[:, :N]
        energies = self.xp.sum(autocorr[:, 1:] ** 2, axis=1)
        
        energies = self.to_host(energies)
        return np.round(energies).astype(np.int32)
    
    def fast_tabu_step(
        self,
        s: np.ndarray,
        current_energy: int,
        best_energy: int,
        tabu_set: set
    ) -> Tuple[int, int]:
        """
        Perform one step of tabu search with GPU-accelerated neighbor evaluation.
        
        Args:
            s: Current sequence (will be modified in place)
            current_energy: Current energy
            best_energy: Global best energy
            tabu_set: Set of tabu positions
        
        Returns:
            Tuple of (best_move_index, new_energy)
        """
        N = len(s)
        
        # GPU-accelerated neighbor evaluation
        neighbor_energies = self.batch_neighbor_energies(s)
        
        # Apply tabu constraints
        best_move = None
        best_move_energy = float('inf')
        
        for i in range(N):
            energy = neighbor_energies[i]
            
            if i in tabu_set:
                # Tabu move - only if aspiration criterion met
                if energy < best_energy and energy < best_move_energy:
                    best_move = i
                    best_move_energy = energy
            else:
                # Non-tabu move
                if energy < best_move_energy:
                    best_move = i
                    best_move_energy = energy
        
        return best_move, int(best_move_energy) if best_move is not None else current_energy


class GPUPopulation:
    """
    GPU-accelerated population management for MTS.
    
    Keeps the entire population on GPU memory for fast operations.
    """
    
    def __init__(self, sequences: List[np.ndarray], accelerator: GPUAccelerator):
        """
        Initialize GPU population.
        
        Args:
            sequences: Initial population
            accelerator: GPU accelerator instance
        """
        self.gpu = accelerator
        self.N = len(sequences[0])
        self.K = len(sequences)
        
        # Stack population on GPU
        pop = np.stack([np.asarray(s, dtype=np.int8) for s in sequences])
        self._population = self.gpu.to_device(pop)
        
        # Calculate initial energies
        self._energies = None
        self.recalculate_energies()
    
    @property
    def population(self) -> List[np.ndarray]:
        """Get population as list of numpy arrays."""
        pop = self.gpu.to_host(self._population)
        return [pop[i] for i in range(self.K)]
    
    @property
    def energies(self) -> List[int]:
        """Get population energies."""
        return list(self._energies)
    
    def recalculate_energies(self):
        """Recalculate all energies."""
        self._energies = self.gpu.batch_energy(self.population)
    
    def get_best(self) -> Tuple[np.ndarray, int]:
        """Get the best individual and its energy."""
        best_idx = int(np.argmin(self._energies))
        pop = self.gpu.to_host(self._population)
        return pop[best_idx].copy(), int(self._energies[best_idx])
    
    def tournament_select(self, tournament_size: int = 2) -> Tuple[np.ndarray, int]:
        """Tournament selection."""
        indices = np.random.choice(self.K, tournament_size, replace=False)
        best_idx = indices[np.argmin([self._energies[i] for i in indices])]
        pop = self.gpu.to_host(self._population)
        return pop[best_idx].copy(), int(self._energies[best_idx])
    
    def replace(self, idx: int, sequence: np.ndarray, energy: int):
        """Replace individual at index."""
        pop = self.gpu.to_host(self._population)
        pop[idx] = sequence
        self._population = self.gpu.to_device(pop)
        self._energies[idx] = energy


def gpu_accelerated_mts(
    N: int,
    population: Optional[List[np.ndarray]] = None,
    population_size: int = 100,
    p_comb: float = 0.9,
    p_mut: Optional[float] = None,
    max_generations: int = 1000,
    target_energy: Optional[int] = None,
    tabu_iterations: int = 100,
    tabu_tenure: int = 7,
    verbose: bool = False
) -> Tuple[np.ndarray, int, dict]:
    """
    GPU-accelerated Memetic Tabu Search.
    
    This is a drop-in replacement for memetic_tabu_search() that uses
    GPU acceleration for key operations.
    
    Args:
        N: Sequence length
        population: Initial population (or None for random)
        population_size: Population size K
        p_comb: Combination probability
        p_mut: Per-bit mutation probability (default 1/N)
        max_generations: Maximum generations
        target_energy: Early stopping target
        tabu_iterations: Tabu search iterations
        tabu_tenure: Tabu tenure
        verbose: Print progress
    
    Returns:
        Tuple of (best_sequence, best_energy, stats)
    """
    import time
    from collections import deque
    
    if p_mut is None:
        p_mut = 1.0 / N
    
    # Initialize GPU accelerator
    gpu = GPUAccelerator()
    
    if verbose:
        print(f"GPU Accelerated MTS: backend={gpu.backend}")
    
    # Initialize population
    if population is None:
        population = [np.random.choice([-1, 1], N).astype(np.int8) 
                     for _ in range(population_size)]
    else:
        population = [np.array(p, dtype=np.int8) for p in population]
        while len(population) < population_size:
            population.append(np.random.choice([-1, 1], N).astype(np.int8))
        population = population[:population_size]
    
    # Batch calculate initial energies
    energies = list(gpu.batch_energy(population))
    
    # Track best
    best_idx = int(np.argmin(energies))
    best_s = population[best_idx].copy()
    best_energy = energies[best_idx]
    
    start_time = time.time()
    total_evaluations = len(population)
    energy_history = [best_energy]
    
    for gen in range(max_generations):
        # Selection and crossover
        if np.random.random() < p_comb:
            # Tournament select two parents
            idx1 = np.random.choice(population_size, 2, replace=False)
            best1 = idx1[np.argmin([energies[i] for i in idx1])]
            
            idx2 = np.random.choice(population_size, 2, replace=False)
            best2 = idx2[np.argmin([energies[i] for i in idx2])]
            
            # GPU crossover (batch of 1)
            children = gpu.batch_crossover([population[best1]], [population[best2]])
            child = children[0]
        else:
            child = population[np.random.randint(population_size)].copy()
        
        # GPU mutation (batch of 1)
        children = gpu.batch_mutate([child], p_mut)
        child = children[0]
        
        # Tabu search with GPU-accelerated neighbor evaluation
        current_s = child.copy()
        current_energy = int(gpu.batch_energy([current_s])[0])
        local_best_s = current_s.copy()
        local_best_energy = current_energy
        
        tabu_list = deque(maxlen=tabu_tenure)
        tabu_set = set()
        
        for _ in range(tabu_iterations):
            # GPU-accelerated neighbor evaluation
            move_idx, move_energy = gpu.fast_tabu_step(
                current_s, current_energy, local_best_energy, tabu_set
            )
            
            if move_idx is None:
                break
            
            # Make move
            current_s[move_idx] *= -1
            current_energy = move_energy
            
            # Update tabu
            if len(tabu_list) == tabu_tenure:
                oldest = tabu_list[0]
                tabu_set.discard(oldest)
            tabu_list.append(move_idx)
            tabu_set.add(move_idx)
            
            # Update local best
            if current_energy < local_best_energy:
                local_best_s = current_s.copy()
                local_best_energy = current_energy
        
        total_evaluations += tabu_iterations
        child = local_best_s
        child_energy = local_best_energy
        
        # Update global best
        if child_energy < best_energy:
            best_s = child.copy()
            best_energy = child_energy
            if verbose:
                print(f"Generation {gen}: New best E={best_energy}")
        
        # Replace in population
        replace_idx = np.random.randint(population_size)
        population[replace_idx] = child
        energies[replace_idx] = child_energy
        
        energy_history.append(best_energy)
        
        if target_energy is not None and best_energy <= target_energy:
            if verbose:
                print(f"Target reached at generation {gen}")
            break
    
    elapsed = time.time() - start_time
    
    stats = {
        'N': N,
        'population_size': population_size,
        'generations': gen + 1,
        'evaluations': total_evaluations,
        'time_seconds': elapsed,
        'initial_best': energy_history[0],
        'final_best': best_energy,
        'energy_history': energy_history,
        'backend': gpu.backend,
        'gpu_accelerated': gpu.is_gpu,
    }
    
    return best_s, best_energy, stats


def benchmark_gpu_vs_cpu(N: int = 20, batch_size: int = 100, n_trials: int = 5):
    """
    Benchmark GPU vs CPU performance for energy calculation.
    
    Args:
        N: Sequence length
        batch_size: Number of sequences to process
        n_trials: Number of benchmark trials
    
    Returns:
        Dictionary with timing results
    """
    import time
    
    # Generate test data
    sequences = [np.random.choice([-1, 1], N).astype(np.int8) 
                 for _ in range(batch_size)]
    
    results = {
        'N': N,
        'batch_size': batch_size,
        'cupy_available': CUPY_AVAILABLE,
    }
    
    # CPU timing
    gpu_cpu = GPUAccelerator(use_gpu=False)
    cpu_times = []
    for _ in range(n_trials):
        start = time.time()
        _ = gpu_cpu.batch_energy(sequences)
        cpu_times.append(time.time() - start)
    results['cpu_mean'] = np.mean(cpu_times)
    results['cpu_std'] = np.std(cpu_times)
    
    # GPU timing (if available)
    if CUPY_AVAILABLE:
        gpu_gpu = GPUAccelerator(use_gpu=True)
        
        # Warm-up
        _ = gpu_gpu.batch_energy(sequences)
        
        gpu_times = []
        for _ in range(n_trials):
            start = time.time()
            _ = gpu_gpu.batch_energy(sequences)
            gpu_times.append(time.time() - start)
        results['gpu_mean'] = np.mean(gpu_times)
        results['gpu_std'] = np.std(gpu_times)
        results['speedup'] = results['cpu_mean'] / results['gpu_mean']
    
    return results


if __name__ == "__main__":
    print("GPU Acceleration Module Test")
    print("=" * 50)
    print(f"CuPy available: {CUPY_AVAILABLE}")
    
    # Test basic operations
    gpu = GPUAccelerator()
    print(f"Backend: {gpu.backend}")
    
    # Test batch energy
    N = 15
    n_seqs = 50
    sequences = [np.random.choice([-1, 1], N).astype(np.int8) for _ in range(n_seqs)]
    
    print(f"\nTesting batch_energy with {n_seqs} sequences of length {N}...")
    energies = gpu.batch_energy(sequences)
    print(f"Energies range: {min(energies)} to {max(energies)}")
    
    # Verify against direct calculation
    from .labs_energy import calculate_energy
    direct_energies = [calculate_energy(s) for s in sequences]
    match = all(e1 == e2 for e1, e2 in zip(energies, direct_energies))
    print(f"Matches direct calculation: {match}")
    
    # Test GPU MTS
    print("\nTesting GPU-accelerated MTS...")
    best_s, best_e, stats = gpu_accelerated_mts(
        N=12,
        population_size=20,
        max_generations=50,
        verbose=True
    )
    print(f"Best energy: {best_e}")
    print(f"Time: {stats['time_seconds']:.2f}s")
    print(f"GPU accelerated: {stats['gpu_accelerated']}")
    
    # Benchmark
    print("\nBenchmarking GPU vs CPU...")
    bench = benchmark_gpu_vs_cpu(N=20, batch_size=100)
    print(f"CPU time: {bench['cpu_mean']*1000:.2f}ms")
    if 'gpu_mean' in bench:
        print(f"GPU time: {bench['gpu_mean']*1000:.2f}ms")
        print(f"Speedup: {bench['speedup']:.1f}x")
