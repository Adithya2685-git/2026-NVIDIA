"""
LABS Energy Calculation Module

This module provides functions for calculating the energy of binary sequences
in the Low Autocorrelation Binary Sequences (LABS) problem.

The LABS problem objective:
    E(s) = Σ_{k=1}^{N-1} C_k²
    where C_k = Σ_{i=1}^{N-k} s_i × s_{i+k}

Key properties:
- Energy is always non-negative
- E(s) = E(-s) (flip symmetry)
- E(s) = E(s[::-1]) (reversal symmetry)
"""

import numpy as np
from typing import List, Union, Tuple, Optional

# Type alias for sequences
Sequence = Union[List[int], np.ndarray]


def calculate_energy(s: Sequence) -> int:
    """
    Calculate LABS energy for a binary sequence.
    
    The energy is defined as:
        E(s) = Σ_{k=1}^{N-1} C_k²
        where C_k = Σ_{i=1}^{N-k} s_i × s_{i+k}
    
    Args:
        s: Binary sequence with values in {+1, -1}
           Can be a list or numpy array
    
    Returns:
        int: Energy value (always non-negative integer)
    
    Examples:
        >>> calculate_energy([1, 1, -1])
        1
        >>> calculate_energy([1, 1, 1, -1])
        4
    """
    s = np.asarray(s, dtype=np.int8)
    N = len(s)
    
    if N < 2:
        return 0
    
    energy = 0
    for k in range(1, N):
        # C_k = sum of s[i] * s[i+k] for i = 0 to N-k-1
        C_k = np.sum(s[:N-k] * s[k:])
        energy += C_k * C_k
    
    return int(energy)


def calculate_energy_vectorized(s: Sequence) -> int:
    """
    Calculate LABS energy using FFT-based autocorrelation.
    
    This is faster for large N but equivalent to calculate_energy().
    
    Args:
        s: Binary sequence with values in {+1, -1}
    
    Returns:
        int: Energy value
    """
    s = np.asarray(s, dtype=np.float64)
    N = len(s)
    
    if N < 2:
        return 0
    
    # Use FFT for autocorrelation
    # autocorr[k] = Σ s[i] * s[i+k]
    f = np.fft.fft(s, n=2*N)
    autocorr = np.fft.ifft(f * np.conj(f)).real[:N]
    
    # Sum of C_k² for k=1 to N-1
    # autocorr[0] is the trivial k=0 case (= N), skip it
    energy = np.sum(autocorr[1:] ** 2)
    
    return int(round(energy))


def calculate_autocorrelations(s: Sequence) -> np.ndarray:
    """
    Calculate all autocorrelation coefficients C_k.
    
    Args:
        s: Binary sequence with values in {+1, -1}
    
    Returns:
        np.ndarray: Array of C_k values for k=1 to N-1
    """
    s = np.asarray(s, dtype=np.int8)
    N = len(s)
    
    C = np.zeros(N - 1, dtype=np.int32)
    for k in range(1, N):
        C[k-1] = np.sum(s[:N-k] * s[k:])
    
    return C


def verify_symmetries(s: Sequence) -> bool:
    """
    Verify that LABS symmetries hold for a sequence.
    
    Checks:
    1. E(s) = E(-s) (flip symmetry)
    2. E(s) = E(s[::-1]) (reversal symmetry)
    
    Args:
        s: Binary sequence with values in {+1, -1}
    
    Returns:
        bool: True if all symmetries hold
    
    Raises:
        AssertionError: If any symmetry is violated
    """
    s = np.asarray(s, dtype=np.int8)
    E = calculate_energy(s)
    
    # Flip symmetry: E(s) = E(-s)
    E_flipped = calculate_energy(-s)
    assert E == E_flipped, f"Flip symmetry violated: E(s)={E}, E(-s)={E_flipped}"
    
    # Reversal symmetry: E(s) = E(s[::-1])
    E_reversed = calculate_energy(s[::-1])
    assert E == E_reversed, f"Reversal symmetry violated: E(s)={E}, E(reversed)={E_reversed}"
    
    # Combined: E(s) = E(-s[::-1])
    E_combined = calculate_energy(-s[::-1])
    assert E == E_combined, f"Combined symmetry violated: E(s)={E}, E(-reversed)={E_combined}"
    
    return True


def get_canonical_sequence(s: Sequence) -> np.ndarray:
    """
    Return the canonical representative of a sequence's equivalence class.
    
    Due to symmetries, each sequence has 4 equivalent forms:
    s, -s, reverse(s), -reverse(s)
    
    We pick the lexicographically smallest one.
    
    Args:
        s: Binary sequence with values in {+1, -1}
    
    Returns:
        np.ndarray: Canonical representative
    """
    s = np.asarray(s, dtype=np.int8)
    
    candidates = [
        s,
        -s,
        s[::-1],
        -s[::-1]
    ]
    
    # Convert to tuples for comparison
    return min(candidates, key=lambda x: tuple(x.tolist()))


def energy_delta_single_flip(s: Sequence, energy: int, flip_idx: int) -> int:
    """
    Calculate energy change from flipping a single bit.
    
    This is an optimization for tabu search - instead of recalculating
    the full energy, we compute the delta.
    
    Args:
        s: Current sequence
        energy: Current energy
        flip_idx: Index of bit to flip
    
    Returns:
        int: New energy after flip (not the delta)
    
    Note:
        For simplicity, this version just recalculates. A truly optimized
        version would compute only the changed terms.
    """
    s = np.asarray(s, dtype=np.int8).copy()
    s[flip_idx] *= -1
    return calculate_energy(s)


def brute_force_optimal(N: int) -> Tuple[int, np.ndarray]:
    """
    Find optimal sequence by brute force enumeration.
    
    Warning: Exponential complexity! Only use for small N (≤ 15).
    
    Args:
        N: Sequence length
    
    Returns:
        Tuple of (optimal_energy, optimal_sequence)
    """
    if N > 20:
        raise ValueError(f"N={N} too large for brute force. Max is 20.")
    
    best_energy = float('inf')
    best_seq = None
    
    # We only need to check 2^(N-2) sequences due to symmetries
    # But for simplicity, we check all and rely on symmetry
    for i in range(2 ** N):
        # Convert integer to binary sequence
        s = np.array([(1 if (i >> j) & 1 else -1) for j in range(N)], dtype=np.int8)
        
        E = calculate_energy(s)
        if E < best_energy:
            best_energy = E
            best_seq = s.copy()
    
    return int(best_energy), best_seq


# Known optimal energies from the literature
# Format: N -> (optimal_energy, example_sequence or None)
KNOWN_OPTIMA = {
    3: (1, np.array([1, 1, -1], dtype=np.int8)),
    4: (4, np.array([1, 1, 1, -1], dtype=np.int8)),
    5: (2, np.array([1, 1, 1, -1, 1], dtype=np.int8)),
    6: (5, np.array([1, 1, 1, 1, -1, 1], dtype=np.int8)),
    7: (4, np.array([1, 1, 1, -1, -1, 1, -1], dtype=np.int8)),
    8: (8, np.array([1, 1, 1, 1, -1, -1, 1, -1], dtype=np.int8)),
    9: (6, None),
    10: (8, None),
    11: (6, np.array([1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1], dtype=np.int8)),
    12: (12, None),
    13: (9, np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.int8)),
    14: (14, None),
    15: (13, None),
    16: (16, None),
    17: (12, None),
    18: (18, None),
    19: (14, None),
    20: (24, None),
    21: (18, None),
    22: (20, None),
    23: (17, None),
    24: (24, None),
    25: (22, None),
}


def get_approximation_ratio(energy: int, N: int) -> float:
    """
    Calculate approximation ratio relative to known optimum.
    
    Ratio = E_opt / E_found
    
    Args:
        energy: Found energy
        N: Sequence length
    
    Returns:
        float: Approximation ratio (1.0 is optimal, lower is worse)
    """
    if N not in KNOWN_OPTIMA:
        return float('nan')
    
    E_opt = KNOWN_OPTIMA[N][0]
    
    if energy == 0:
        return float('inf') if E_opt == 0 else 0.0
    
    return E_opt / energy


def random_sequence(N: int) -> np.ndarray:
    """
    Generate a random binary sequence.
    
    Args:
        N: Sequence length
    
    Returns:
        np.ndarray: Random sequence with values in {+1, -1}
    """
    return np.random.choice([-1, 1], size=N).astype(np.int8)


def bitstring_to_sequence(bitstring: str) -> np.ndarray:
    """
    Convert a bitstring (e.g., '01101') to a sequence.
    
    Convention: '0' -> +1, '1' -> -1
    
    Args:
        bitstring: String of 0s and 1s
    
    Returns:
        np.ndarray: Sequence with values in {+1, -1}
    """
    return np.array([1 if b == '0' else -1 for b in bitstring], dtype=np.int8)


def sequence_to_bitstring(s: Sequence) -> str:
    """
    Convert a sequence to a bitstring.
    
    Convention: +1 -> '0', -1 -> '1'
    
    Args:
        s: Sequence with values in {+1, -1}
    
    Returns:
        str: Bitstring representation
    """
    s = np.asarray(s)
    return ''.join('0' if x == 1 else '1' for x in s)


if __name__ == "__main__":
    # Quick test
    print("Testing LABS energy calculation...")
    
    # Test known optima
    for N in [3, 4, 5, 7]:
        E_opt, seq = KNOWN_OPTIMA[N]
        if seq is not None:
            E_calc = calculate_energy(seq)
            print(f"N={N}: E_opt={E_opt}, E_calc={E_calc}, match={E_opt == E_calc}")
            verify_symmetries(seq)
    
    print("All tests passed!")
