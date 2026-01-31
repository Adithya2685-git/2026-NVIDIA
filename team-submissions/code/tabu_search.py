"""
Tabu Search Algorithm for LABS Optimization

Tabu search is a local search metaheuristic that uses memory structures
to avoid cycling and escape local minima. It maintains a "tabu list" of
recently visited solutions that are forbidden for a number of iterations.

Key features:
- Tabu tenure: Number of iterations a move stays forbidden
- Aspiration criterion: Allow tabu moves if they improve global best
- Neighborhood: All single-bit flips
"""

import numpy as np
from typing import Tuple, List, Set, Optional
from collections import deque

from .labs_energy import calculate_energy, Sequence


def tabu_search(
    s: Sequence,
    max_iterations: int = 1000,
    tabu_tenure: int = 7,
    verbose: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Perform tabu search starting from a given sequence.
    
    The algorithm:
    1. Evaluate all neighbors (single bit flips)
    2. Select best non-tabu move (or use aspiration if it beats global best)
    3. Add the flipped position to tabu list
    4. Repeat until max_iterations or no improving moves
    
    Args:
        s: Initial sequence (+1/-1 values)
        max_iterations: Maximum number of iterations
        tabu_tenure: How long a position stays tabu after being flipped
        verbose: Print progress information
    
    Returns:
        Tuple of (best_sequence, best_energy)
    """
    s = np.array(s, dtype=np.int8)
    N = len(s)
    
    # Current solution
    current_s = s.copy()
    current_energy = calculate_energy(current_s)
    
    # Best solution found
    best_s = current_s.copy()
    best_energy = current_energy
    
    # Tabu list: stores recently flipped positions
    # Using deque with maxlen for automatic eviction
    tabu_list: deque = deque(maxlen=tabu_tenure)
    tabu_set: Set[int] = set()  # For O(1) lookup
    
    # Track iterations without improvement for early stopping
    no_improvement_count = 0
    max_no_improvement = min(N * 10, 500)  # Adaptive patience
    
    for iteration in range(max_iterations):
        best_move: Optional[int] = None
        best_move_energy = float('inf')
        
        # Evaluate all neighbors (single bit flips)
        for i in range(N):
            # Flip bit i
            current_s[i] *= -1
            new_energy = calculate_energy(current_s)
            current_s[i] *= -1  # Flip back
            
            if i in tabu_set:
                # Tabu move - only consider if aspiration criterion met
                if new_energy < best_energy:  # Aspiration: beats global best
                    if new_energy < best_move_energy:
                        best_move = i
                        best_move_energy = new_energy
            else:
                # Non-tabu move
                if new_energy < best_move_energy:
                    best_move = i
                    best_move_energy = new_energy
        
        # No valid move found (shouldn't happen normally)
        if best_move is None:
            if verbose:
                print(f"Iteration {iteration}: No valid move, stopping")
            break
        
        # Make the best move
        current_s[best_move] *= -1
        current_energy = best_move_energy
        
        # Update tabu list
        if len(tabu_list) == tabu_tenure:
            # Remove oldest from set
            oldest = tabu_list[0]
            tabu_set.discard(oldest)
        tabu_list.append(best_move)
        tabu_set.add(best_move)
        
        # Update best solution
        if current_energy < best_energy:
            best_s = current_s.copy()
            best_energy = current_energy
            no_improvement_count = 0
            
            if verbose:
                print(f"Iteration {iteration}: New best energy = {best_energy}")
        else:
            no_improvement_count += 1
        
        # Early termination if optimal (energy = 0 is impossible for most N)
        if best_energy == 0:
            break
        
        # Early stopping if no improvement for too long
        if no_improvement_count >= max_no_improvement:
            if verbose:
                print(f"Iteration {iteration}: No improvement for {max_no_improvement} iterations, stopping")
            break
    
    return best_s, best_energy


def tabu_search_with_restarts(
    N: int,
    n_restarts: int = 5,
    max_iterations_per_restart: int = 500,
    tabu_tenure: int = 7,
    verbose: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Tabu search with random restarts.
    
    Run tabu search multiple times from different random starting points
    and return the best result.
    
    Args:
        N: Sequence length
        n_restarts: Number of random restarts
        max_iterations_per_restart: Max iterations per restart
        tabu_tenure: Tabu tenure parameter
        verbose: Print progress
    
    Returns:
        Tuple of (best_sequence, best_energy)
    """
    best_s = None
    best_energy = float('inf')
    
    for restart in range(n_restarts):
        # Random starting point
        s = np.random.choice([-1, 1], N).astype(np.int8)
        
        # Run tabu search
        result_s, result_energy = tabu_search(
            s, 
            max_iterations=max_iterations_per_restart,
            tabu_tenure=tabu_tenure,
            verbose=False
        )
        
        if result_energy < best_energy:
            best_s = result_s
            best_energy = result_energy
            
            if verbose:
                print(f"Restart {restart + 1}/{n_restarts}: New best energy = {best_energy}")
    
    return best_s, best_energy


def evaluate_all_neighbors(s: np.ndarray) -> np.ndarray:
    """
    Evaluate energies of all single-flip neighbors.
    
    Args:
        s: Current sequence
    
    Returns:
        np.ndarray: Array of energies for each neighbor (flip position i)
    """
    N = len(s)
    energies = np.zeros(N, dtype=np.int32)
    
    for i in range(N):
        s[i] *= -1
        energies[i] = calculate_energy(s)
        s[i] *= -1
    
    return energies


def is_local_minimum(s: Sequence) -> bool:
    """
    Check if a sequence is a local minimum.
    
    A sequence is a local minimum if no single bit flip reduces the energy.
    
    Args:
        s: Sequence to check
    
    Returns:
        bool: True if s is a local minimum
    """
    s = np.array(s, dtype=np.int8)
    current_energy = calculate_energy(s)
    
    for i in range(len(s)):
        s[i] *= -1
        neighbor_energy = calculate_energy(s)
        s[i] *= -1
        
        if neighbor_energy < current_energy:
            return False
    
    return True


if __name__ == "__main__":
    # Quick test
    import time
    
    print("Testing Tabu Search...")
    
    # Test on N=20
    N = 20
    s = np.random.choice([-1, 1], N).astype(np.int8)
    initial_energy = calculate_energy(s)
    
    start = time.time()
    result_s, result_energy = tabu_search(s, max_iterations=1000, verbose=True)
    elapsed = time.time() - start
    
    print(f"\nN={N}")
    print(f"Initial energy: {initial_energy}")
    print(f"Final energy: {result_energy}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Is local minimum: {is_local_minimum(result_s)}")
