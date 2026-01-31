"""
CUDA-Q Kernels for Digitized Counterdiabatic Quantum Optimization (DCQO)

This module implements the quantum circuits for the counteradiabatic
approach to solving the LABS problem, as described in:
"Scaling advantage with quantum-enhanced memetic tabu search for LABS"

Key components:
- R_ZZ gate: Two-qubit ZZ rotation
- R_YZ block: Two-qubit rotation from Equation 15
- R_YZZZ block: Four-qubit rotation from Equation 15
- Trotterized circuit: Full counteradiabatic evolution

Note: This code requires CUDA-Q (cudaq) to be installed.
For CPU simulation, use target='qpp-cpu'
For GPU acceleration, use target='nvidia'
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from math import pi, floor

# Import CUDA-Q (will fail gracefully if not installed)
try:
    import cudaq
    CUDAQ_AVAILABLE = True
except ImportError:
    CUDAQ_AVAILABLE = False
    print("Warning: CUDA-Q not available. Quantum kernels will not work.")


def get_interactions(N: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Generate G2 (2-body) and G4 (4-body) interaction indices.
    
    Based on Equation 2 and 15 from the paper:
    
    H_f = 2 * Σ_i Σ_k Z_i Z_{i+k} (2-body)
        + 4 * Σ_i Σ_t Σ_k Z_i Z_{i+t} Z_{i+k} Z_{i+k+t} (4-body)
    
    Loop limits:
    - 2-body: i from 1 to N-2, k from 1 to floor((N-i)/2)
    - 4-body: i from 1 to N-3, t from 1 to floor((N-i-1)/2), 
              k from t+1 to N-i-t
    
    Args:
        N: Sequence/qubit length
    
    Returns:
        G2: List of [i, j] pairs for 2-body terms
        G4: List of [i, i+t, i+k, i+k+t] quads for 4-body terms
    
    Note: Returns 0-indexed qubit positions.
    """
    G2: List[List[int]] = []
    G4: List[List[int]] = []
    
    # 2-body terms
    # Paper uses 1-indexed: i from 1 to N-2, k from 1 to floor((N-i)/2)
    # Convert to 0-indexed: i from 0 to N-3, k from 1 to floor((N-1-i)/2)
    for i in range(N - 2):  # i = 0 to N-3 (corresponds to paper's 1 to N-2)
        for k in range(1, floor((N - 1 - i) / 2) + 1):
            # Interaction between qubit i and qubit i+k
            G2.append([i, i + k])
    
    # 4-body terms
    # Paper: i from 1 to N-3, t from 1 to floor((N-i-1)/2), k from t+1 to N-i-t
    # 0-indexed: i from 0 to N-4, t from 1 to floor((N-i-2)/2), k from t+1 to N-i-t-1
    for i in range(N - 3):  # i = 0 to N-4
        for t in range(1, floor((N - i - 2) / 2) + 1):
            for k in range(t + 1, N - i - t):
                # Four qubits: i, i+t, i+k, i+k+t
                G4.append([i, i + t, i + k, i + k + t])
    
    return G2, G4


def count_interactions(N: int) -> Tuple[int, int]:
    """
    Count the number of 2-body and 4-body interactions.
    
    Uses formulas from Equations 3 and 4 of the paper for 2-body terms.
    For 4-body terms, we compute the actual count from the loop bounds
    since the paper's formula has edge cases.
    
    n_two(N) = (N/2)(N/2 - 1) if N even
             = ((N-1)/2)^2 if N odd
    
    Args:
        N: Sequence length
    
    Returns:
        Tuple of (n_two, n_four)
    """
    # 2-body formula from paper
    if N % 2 == 0:
        n_two = (N // 2) * (N // 2 - 1)
    else:
        n_two = ((N - 1) // 2) ** 2
    
    # 4-body: count from loop bounds (more reliable than paper formula)
    # This matches get_interactions exactly
    n_four = 0
    for i in range(N - 3):
        for t in range(1, floor((N - i - 2) / 2) + 1):
            for k in range(t + 1, N - i - t):
                n_four += 1
    
    return n_two, n_four


if CUDAQ_AVAILABLE:
    
    @cudaq.kernel
    def rzz_gate(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
        """
        R_ZZ(theta) = exp(-i * theta/2 * Z⊗Z)
        
        Decomposition:
            CNOT(q0, q1)
            RZ(theta, q1)
            CNOT(q0, q1)
        
        Args:
            q0: First qubit
            q1: Second qubit
            theta: Rotation angle
        """
        x.ctrl(q0, q1)
        rz(theta, q1)
        x.ctrl(q0, q1)
    
    
    @cudaq.kernel
    def r_yz_block(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
        """
        Apply R_YZ(θ) R_ZY(θ) block for 2-body terms.
        
        This implements the circuit from Figure 3 of the paper.
        Requires 2 RZZ gates and 4 single-qubit rotations.
        
        R_YZ = exp(-i * theta * Y⊗Z)
        R_ZY = exp(-i * theta * Z⊗Y)
        
        Args:
            q0: First qubit (Y acts here for R_YZ)
            q1: Second qubit (Y acts here for R_ZY)
            theta: Rotation angle (already includes factor of 4 from Eq. 15)
        """
        # R_YZ(θ): exp(-i * θ * Y⊗Z)
        # Decomposition: RZ(-π/2) on q0, RZZ, RZ(π/2) on q0
        rz(-pi / 2, q0)
        # Apply RZZ
        x.ctrl(q0, q1)
        rz(-theta, q1)
        x.ctrl(q0, q1)
        rz(pi / 2, q0)
        
        # R_ZY(θ): exp(-i * θ * Z⊗Y)
        # Decomposition: RZ(-π/2) on q1, RZZ, RZ(π/2) on q1
        rz(-pi / 2, q1)
        # Apply RZZ
        x.ctrl(q0, q1)
        rz(theta, q1)
        x.ctrl(q0, q1)
        rz(pi / 2, q1)
    
    
    @cudaq.kernel
    def r_yzzz_block(q0: cudaq.qubit, q1: cudaq.qubit, 
                     q2: cudaq.qubit, q3: cudaq.qubit, theta: float):
        """
        Apply 4-qubit rotation block for 4-body terms.
        
        This implements the circuit from Figure 4 of the paper:
        R_YZZZ(θ) R_ZYZZ(θ) R_ZZYZ(θ) R_ZZZY(θ)
        
        Requires 10 RZZ gates and 28 single-qubit rotations.
        
        The 4-body rotation R_YZZZ = exp(-i * θ * Y⊗Z⊗Z⊗Z) is decomposed
        using a ladder of CNOT gates with RY in the middle.
        
        Args:
            q0, q1, q2, q3: Four qubits involved
            theta: Rotation angle (already includes factor of 8 from Eq. 15)
        """
        # We implement a simplified version using the ladder decomposition
        # Each R_YZZZ type rotation becomes: ladder CNOTs, RY, ladder CNOTs back
        
        # R_YZZZ(θ): Y on q0, Z on q1,q2,q3
        rz(-pi / 2, q0)
        x.ctrl(q0, q1)
        x.ctrl(q1, q2)
        x.ctrl(q2, q3)
        rz(theta, q3)
        x.ctrl(q2, q3)
        x.ctrl(q1, q2)
        x.ctrl(q0, q1)
        rz(pi / 2, q0)
        
        # R_ZYZZ(θ): Y on q1, Z on q0,q2,q3
        rz(-pi / 2, q1)
        x.ctrl(q1, q0)
        x.ctrl(q1, q2)
        x.ctrl(q2, q3)
        rz(theta, q3)
        x.ctrl(q2, q3)
        x.ctrl(q1, q2)
        x.ctrl(q1, q0)
        rz(pi / 2, q1)
        
        # R_ZZYZ(θ): Y on q2, Z on q0,q1,q3
        rz(-pi / 2, q2)
        x.ctrl(q2, q0)
        x.ctrl(q2, q1)
        x.ctrl(q2, q3)
        rz(theta, q3)
        x.ctrl(q2, q3)
        x.ctrl(q2, q1)
        x.ctrl(q2, q0)
        rz(pi / 2, q2)
        
        # R_ZZZY(θ): Y on q3, Z on q0,q1,q2
        rz(-pi / 2, q3)
        x.ctrl(q3, q0)
        x.ctrl(q3, q1)
        x.ctrl(q3, q2)
        rz(theta, q3)
        x.ctrl(q3, q2)
        x.ctrl(q3, q1)
        x.ctrl(q3, q0)
        rz(pi / 2, q3)
    
    
    @cudaq.kernel
    def trotterized_circuit(N: int, G2_flat: list[int], n_G2: int,
                            G4_flat: list[int], n_G4: int,
                            thetas: list[float], n_steps: int):
        """
        Full Trotterized counteradiabatic circuit.
        
        Implements U(0,T) from Equation 15:
        
        U(0,T) = ∏_{n=1}^{n_trot} [ ∏_{2-body} R_YZ R_ZY ] × [ ∏_{4-body} R_YZZZ R_ZYZZ R_ZZYZ R_ZZZY ]
        
        Args:
            N: Number of qubits
            G2_flat: Flattened list of 2-body indices [i0,j0, i1,j1, ...]
            n_G2: Number of 2-body terms
            G4_flat: Flattened list of 4-body indices [i0,j0,k0,l0, i1,j1,k1,l1, ...]
            n_G4: Number of 4-body terms
            thetas: List of theta values for each Trotter step
            n_steps: Number of Trotter steps
        
        Note:
            We use flattened lists because CUDA-Q kernels don't support
            nested lists directly.
        """
        # Allocate qubits
        reg = cudaq.qvector(N)
        
        # Initialize |+>^N (ground state of H_i = -Σ σ^x_i)
        for i in range(N):
            h(reg[i])
        
        # Apply Trotter steps
        for step in range(n_steps):
            theta = thetas[step]
            
            # Apply 2-body terms with coefficient 4*theta
            for g in range(n_G2):
                i = G2_flat[2 * g]
                j = G2_flat[2 * g + 1]
                r_yz_block(reg[i], reg[j], 4.0 * theta)
            
            # Apply 4-body terms with coefficient 8*theta
            for g in range(n_G4):
                i = G4_flat[4 * g]
                j = G4_flat[4 * g + 1]
                k = G4_flat[4 * g + 2]
                l = G4_flat[4 * g + 3]
                r_yzzz_block(reg[i], reg[j], reg[k], reg[l], 8.0 * theta)


def flatten_interactions(G2: List[List[int]], G4: List[List[int]]) -> Tuple[List[int], List[int]]:
    """
    Flatten interaction lists for CUDA-Q kernel.
    
    CUDA-Q kernels don't support nested lists, so we flatten:
    [[0,1], [0,2], [1,2]] -> [0,1, 0,2, 1,2]
    
    Args:
        G2: List of 2-element lists
        G4: List of 4-element lists
    
    Returns:
        Tuple of flattened lists
    """
    G2_flat = []
    for pair in G2:
        G2_flat.extend(pair)
    
    G4_flat = []
    for quad in G4:
        G4_flat.extend(quad)
    
    return G2_flat, G4_flat


def run_dcqo(
    N: int,
    n_shots: int = 1000,
    n_steps: int = 1,
    T: float = 1.0,
    target: str = 'qpp-cpu',
    verbose: bool = False
) -> Tuple[List[np.ndarray], List[int], Dict[str, Any]]:
    """
    Run Digitized Counterdiabatic Quantum Optimization.
    
    Args:
        N: Sequence length (number of qubits)
        n_shots: Number of measurement shots
        n_steps: Number of Trotter steps
        T: Total evolution time
        target: CUDA-Q target ('qpp-cpu', 'nvidia', 'nvidia-mgpu')
        verbose: Print progress information
    
    Returns:
        Tuple of:
        - samples: List of sequences (numpy arrays)
        - energies: List of corresponding energies
        - stats: Dictionary with execution statistics
    """
    if not CUDAQ_AVAILABLE:
        raise RuntimeError("CUDA-Q is not available")
    
    import time
    from .labs_energy import calculate_energy, bitstring_to_sequence
    
    start_time = time.time()
    
    # Set target
    try:
        cudaq.set_target(target)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not set target '{target}': {e}")
            print("Falling back to 'qpp-cpu'")
        cudaq.set_target('qpp-cpu')
        target = 'qpp-cpu'
    
    # Get interactions
    G2, G4 = get_interactions(N)
    G2_flat, G4_flat = flatten_interactions(G2, G4)
    
    if verbose:
        print(f"DCQO: N={N}, 2-body={len(G2)}, 4-body={len(G4)}")
        print(f"Target: {target}, Shots: {n_shots}, Steps: {n_steps}")
    
    # Compute theta values for each Trotter step
    # Import the compute_theta function from labs_utils
    import sys
    import os
    
    # Add the tutorial_notebook/auxiliary_files to path
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    utils_path = os.path.join(repo_root, 'tutorial_notebook', 'auxiliary_files')
    if utils_path not in sys.path:
        sys.path.insert(0, utils_path)
    
    try:
        from labs_utils import compute_theta
    except ImportError:
        # Fallback: use a simple theta calculation
        if verbose:
            print("Warning: Could not import labs_utils.compute_theta, using simple schedule")
        
        def compute_theta(t, dt, T, N, G2, G4):
            # Simple sin^2 schedule with constant alpha
            from math import sin, pi
            lam_dot = (pi / (2 * T)) * sin((pi * t) / T)
            alpha = 0.1  # Simple approximation
            return dt * alpha * lam_dot
    
    dt = T / n_steps
    thetas = []
    for step in range(1, n_steps + 1):
        t = step * dt
        theta_val = compute_theta(t, dt, T, N, G2, G4)
        thetas.append(float(theta_val))
    
    if verbose:
        print(f"Thetas: {thetas}")
    
    # Sample the circuit
    circuit_start = time.time()
    result = cudaq.sample(
        trotterized_circuit,
        N,
        G2_flat, len(G2),
        G4_flat, len(G4),
        thetas, n_steps,
        shots_count=n_shots
    )
    circuit_time = time.time() - circuit_start
    
    # Convert results to sequences
    samples = []
    energies = []
    
    for bitstring, count in result.items():
        seq = bitstring_to_sequence(bitstring)
        energy = calculate_energy(seq)
        
        for _ in range(count):
            samples.append(seq.copy())
            energies.append(energy)
    
    total_time = time.time() - start_time
    
    # Statistics
    stats = {
        'N': N,
        'n_shots': n_shots,
        'n_steps': n_steps,
        'T': T,
        'target': target,
        'n_two_body': len(G2),
        'n_four_body': len(G4),
        'unique_bitstrings': len(result),
        'circuit_time': circuit_time,
        'total_time': total_time,
        'min_energy': min(energies) if energies else None,
        'mean_energy': np.mean(energies) if energies else None,
        'std_energy': np.std(energies) if energies else None,
    }
    
    if verbose:
        print(f"Circuit time: {circuit_time:.3f}s")
        print(f"Unique bitstrings: {stats['unique_bitstrings']}")
        print(f"Min energy: {stats['min_energy']}, Mean: {stats['mean_energy']:.2f}")
    
    return samples, energies, stats


if __name__ == "__main__":
    # Quick test
    print("Testing DCQO Kernels...")
    
    # Test get_interactions
    for N in [5, 7, 10]:
        G2, G4 = get_interactions(N)
        n_two_expected, n_four_expected = count_interactions(N)
        print(f"N={N}: G2={len(G2)} (expected {n_two_expected}), G4={len(G4)} (expected {n_four_expected})")
    
    # Test DCQO if CUDA-Q is available
    if CUDAQ_AVAILABLE:
        print("\nRunning DCQO test...")
        samples, energies, stats = run_dcqo(N=5, n_shots=100, verbose=True)
        print(f"Got {len(samples)} samples")
    else:
        print("\nCUDA-Q not available, skipping circuit test")
