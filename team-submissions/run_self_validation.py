#!/usr/bin/env python3
"""
Self-Validation Script for Phase 1

This runs the validation checks from the tutorial notebook.
Usage: python3 run_self_validation.py
"""

import sys
import numpy as np

# Add labs_solver to path
sys.path.insert(0, '/home/adi/Documents/iquhack/2026-NVIDIA/team-submissions')

from labs_solver.labs_energy import calculate_energy, KNOWN_OPTIMA
from labs_solver.dcqo_kernels import get_interactions, count_interactions
from labs_solver.mts import memetic_tabu_search

print("="*60)
print("SELF-VALIDATION FOR PHASE 1")
print("="*60)

# 1. Verify known optimal energies
print("\n1. KNOWN OPTIMA VERIFICATION")
print("-" * 40)

all_passed = True
for N, (E_opt, seq) in KNOWN_OPTIMA.items():
    if seq is not None:
        E_calc = calculate_energy(seq)
        passed = E_calc == E_opt
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  N={N}: E_opt={E_opt}, E_calc={E_calc} [{status}]")
        if not passed:
            all_passed = False

# 2. Verify symmetries
print("\n2. SYMMETRY VERIFICATION")
print("-" * 40)

test_sequences = [
    [1, 1, -1, 1, -1],
    [1, -1, 1, 1, -1, -1, 1],
    [-1, 1, 1, 1, -1, 1, -1, -1],
]

for s in test_sequences:
    s = np.array(s)
    E = calculate_energy(s)
    E_neg = calculate_energy(-s)
    E_rev = calculate_energy(s[::-1])
    E_both = calculate_energy(-s[::-1])
    
    sym_ok = (E == E_neg == E_rev == E_both)
    status = "✓ PASS" if sym_ok else "✗ FAIL"
    print(f"  E={E}, E(-s)={E_neg}, E(rev)={E_rev} [{status}]")
    if not sym_ok:
        all_passed = False

# 3. Brute force verification for small N
print("\n3. BRUTE FORCE VERIFICATION (N=5, N=7)")
print("-" * 40)

for N in [5, 7]:
    min_energy = float('inf')
    best_s = None
    for i in range(2**N):
        s = [1 if (i >> j) & 1 else -1 for j in range(N)]
        E = calculate_energy(s)
        if E < min_energy:
            min_energy = E
            best_s = s
    
    expected = KNOWN_OPTIMA.get(N, (None, None))[0]
    if expected is not None:
        bf_passed = min_energy == expected
        status = "✓ PASS" if bf_passed else "✗ FAIL"
        print(f"  N={N}: brute force min={min_energy}, expected={expected} [{status}]")
        if not bf_passed:
            all_passed = False

# 4. Verify interaction counts
print("\n4. INTERACTION COUNT VERIFICATION")
print("-" * 40)

for N in [5, 7, 10, 12, 15]:
    G2, G4 = get_interactions(N)
    exp_2, exp_4 = count_interactions(N)
    passed = (len(G2) == exp_2) and (len(G4) == exp_4)
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  N={N}: G2={len(G2)}/{exp_2}, G4={len(G4)}/{exp_4} [{status}]")
    if not passed:
        all_passed = False

# 5. MTS finds optima for small N
print("\n5. MTS FINDS KNOWN OPTIMA")
print("-" * 40)

for N in [3, 5]:
    E_opt = KNOWN_OPTIMA.get(N, (None, None))[0]
    if E_opt is not None:
        _, best_E, _ = memetic_tabu_search(N, population_size=30, max_generations=100, verbose=False)
        passed = best_E <= E_opt
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  N={N}: MTS found {best_E}, optimal is {E_opt} [{status}]")
        if not passed:
            all_passed = False

# 6. Annealing schedules
print("\n6. ANNEALING SCHEDULES")
print("-" * 40)

from labs_solver.schedules import ALL_SCHEDULES

for Schedule in ALL_SCHEDULES:
    s = Schedule(T=1.0)
    lam_0 = s.lambda_t(0)
    lam_1 = s.lambda_t(1.0)
    passed = (abs(lam_0) < 1e-10) and (abs(lam_1 - 1.0) < 1e-10)
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {Schedule.__name__}: λ(0)={lam_0:.4f}, λ(1)={lam_1:.4f} [{status}]")
    if not passed:
        all_passed = False

# Final summary
print("\n" + "="*60)
if all_passed:
    print("✓✓✓ ALL 6 VALIDATION CHECKS PASSED! ✓✓✓")
    print("Phase 1 Self-Validation Complete")
else:
    print("✗✗✗ SOME TESTS FAILED - Please review above ✗✗✗")
print("="*60)

sys.exit(0 if all_passed else 1)
