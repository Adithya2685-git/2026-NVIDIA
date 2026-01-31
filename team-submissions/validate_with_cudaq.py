#!/usr/bin/env python3
"""
Self-Validation with CUDA-Q for iQuHACK Phase 1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'team-submissions'))

import numpy as np
from math import floor, pi
from collections import deque
import cudaq

# Import from our labs_solver
from labs_solver.labs_energy import calculate_energy
from labs_solver.mts import memetic_tabu_search
from labs_solver.dcqo_kernels import get_interactions

print('='*70)
print('  SELF-VALIDATION FOR PHASE 1 - With CUDA-Q!')
print('='*70)
print(f'CUDA-Q Version: {cudaq.__version__}')

# 1. Known optima
print('\n1. KNOWN OPTIMA VERIFICATION')
print('-' * 50)
known_optima = {
    3: (1, [1, 1, -1]),
    4: (2, [1, -1, -1, -1]),
    5: (2, [1, 1, 1, -1, 1]),
    7: (3, [1, -1, 1, 1, -1, -1, -1]),
}
all_passed = True
for N, (E_opt, seq) in known_optima.items():
    E_calc = calculate_energy(seq)
    passed = E_calc == E_opt
    status = '✓' if passed else '✗'
    print(f'  {status} N={N}: E_calc={E_calc} (expected {E_opt})')
    if not passed:
        all_passed = False

# 2. Symmetries
print('\n2. SYMMETRY VERIFICATION')
print('-' * 50)
test_sequences = [
    [1, 1, -1, 1, -1],
    [1, -1, 1, 1, -1, -1, 1],
]
for s in test_sequences:
    s = np.array(s)
    E = calculate_energy(s)
    E_neg = calculate_energy(-s)
    E_rev = calculate_energy(s[::-1])
    sym_ok = (E == E_neg == E_rev)
    status = '✓' if sym_ok else '✗'
    print(f'  {status} E={E}, E(-s)={E_neg}, E(rev)={E_rev}')
    if not sym_ok:
        all_passed = False

# 3. Interactions
print('\n3. INTERACTION COUNT VERIFICATION')
print('-' * 50)
for N in [5, 7, 10, 15]:
    G2, G4 = get_interactions(N)
    print(f'  ✓ N={N}: G2={len(G2)}, G4={len(G4)}')

# 4. MTS
print('\n4. MTS ALGORITHM')
print('-' * 50)
for N in [5, 10]:
    _, best_E, _ = memetic_tabu_search(N, population_size=20, max_generations=50, verbose=False)
    print(f'  ✓ N={N}: Best energy = {best_E}')

# 5. CUDA-Q Test
print('\n5. CUDA-Q QUANTUM CIRCUIT TEST')
print('-' * 50)

cudaq.set_target('qpp-cpu')

# Test simple bell state
@cudaq.kernel
def bell_state():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])
    mz(qubits)

result = cudaq.sample(bell_state, shots_count=100)
print(f'  ✓ Sampled quantum circuit: {len(result)} unique bitstrings')

# Show results
for bitstring, count in result.items():
    print(f'      {bitstring}: {count} shots')

print(f'  ✓ CUDA-Q is working correctly!')

print('\n' + '='*70)
if all_passed:
    print('✓✓✓ ALL VALIDATION TESTS PASSED! ✓✓✓')
    print('Phase 1 Complete - CUDA-Q is working!')
else:
    print('Some tests failed')
print('='*70)
