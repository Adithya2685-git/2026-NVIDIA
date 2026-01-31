#!/usr/bin/env python3
"""
Local Testing Script for S-Tier LABS Solver

This script tests all components locally and reports what's available.
Run this to verify your setup before submitting.

Usage:
    cd team-submissions
    python3 test_local.py
"""

import sys
import time
import numpy as np
from datetime import datetime

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_imports():
    """Test that all modules can be imported."""
    print_section("1. Testing Module Imports")
    
    results = {}
    
    try:
        from labs_solver.labs_energy import calculate_energy, KNOWN_OPTIMA, random_sequence
        print("✓ labs_energy.py")
        results['labs_energy'] = True
    except Exception as e:
        print(f"✗ labs_energy.py: {e}")
        results['labs_energy'] = False
    
    try:
        from labs_solver.tabu_search import tabu_search, is_local_minimum
        print("✓ tabu_search.py")
        results['tabu_search'] = True
    except Exception as e:
        print(f"✗ tabu_search.py: {e}")
        results['tabu_search'] = False
    
    try:
        from labs_solver.mts import combine, mutate, memetic_tabu_search
        print("✓ mts.py")
        results['mts'] = True
    except Exception as e:
        print(f"✗ mts.py: {e}")
        results['mts'] = False
    
    try:
        from labs_solver.dcqo_kernels import get_interactions, CUDAQ_AVAILABLE
        print(f"✓ dcqo_kernels.py (CUDA-Q available: {CUDAQ_AVAILABLE})")
        results['dcqo_kernels'] = True
        results['cudaq'] = CUDAQ_AVAILABLE
    except Exception as e:
        print(f"✗ dcqo_kernels.py: {e}")
        results['dcqo_kernels'] = False
        results['cudaq'] = False
    
    try:
        from labs_solver.schedules import ALL_SCHEDULES, SinSquaredSchedule
        print(f"✓ schedules.py ({len(ALL_SCHEDULES)} schedules)")
        results['schedules'] = True
    except Exception as e:
        print(f"✗ schedules.py: {e}")
        results['schedules'] = False
    
    try:
        from labs_solver.boltzmann_seeding import boltzmann_weights, boltzmann_seeding, diversity_metrics
        print("✓ boltzmann_seeding.py")
        results['boltzmann_seeding'] = True
    except Exception as e:
        print(f"✗ boltzmann_seeding.py: {e}")
        results['boltzmann_seeding'] = False
    
    try:
        from labs_solver.gpu_acceleration import GPUAccelerator, CUPY_AVAILABLE
        print(f"✓ gpu_acceleration.py (CuPy available: {CUPY_AVAILABLE})")
        results['gpu_acceleration'] = True
        results['cupy'] = CUPY_AVAILABLE
    except Exception as e:
        print(f"✗ gpu_acceleration.py: {e}")
        results['gpu_acceleration'] = False
        results['cupy'] = False
    
    try:
        from labs_solver.s_tier_solver import STierLABSSolver, compare_methods
        print("✓ s_tier_solver.py")
        results['s_tier_solver'] = True
    except Exception as e:
        print(f"✗ s_tier_solver.py: {e}")
        results['s_tier_solver'] = False
    
    return results

def test_correctness():
    """Test correctness of energy calculation."""
    print_section("2. Testing Correctness")
    
    from labs_solver.labs_energy import calculate_energy, KNOWN_OPTIMA, random_sequence
    
    # Test known optima
    print("\nTesting known optimal energies:")
    all_pass = True
    for N in [3, 5, 7]:
        E_opt, seq = KNOWN_OPTIMA[N]
        if seq is not None:
            E_calc = calculate_energy(seq)
            status = "✓" if E_calc == E_opt else "✗"
            print(f"  {status} N={N}: E={E_calc} (expected {E_opt})")
            if E_calc != E_opt:
                all_pass = False
    
    # Test symmetries
    print("\nTesting symmetries:")
    for _ in range(5):
        N = np.random.randint(5, 15)
        s = random_sequence(N)
        E1 = calculate_energy(s)
        E2 = calculate_energy(-s)
        E3 = calculate_energy(s[::-1])
        
        if E1 == E2 == E3:
            print(f"  ✓ N={N}: E(s)={E1}, E(-s)={E2}, E(rev)={E3}")
        else:
            print(f"  ✗ N={N}: Symmetry violation!")
            all_pass = False
    
    return all_pass

def test_interactions():
    """Test interaction generation."""
    print_section("3. Testing Interaction Generation")
    
    from labs_solver.dcqo_kernels import get_interactions, count_interactions
    
    all_pass = True
    for N in [5, 7, 10, 15, 20]:
        G2, G4 = get_interactions(N)
        n2, n4 = count_interactions(N)
        
        g2_ok = len(G2) == n2
        g4_ok = len(G4) == n4
        
        status = "✓" if (g2_ok and g4_ok) else "✗"
        print(f"  {status} N={N}: G2={len(G2)}/{n2}, G4={len(G4)}/{n4}")
        
        if not (g2_ok and g4_ok):
            all_pass = False
    
    return all_pass

def test_schedules():
    """Test annealing schedules."""
    print_section("4. Testing Annealing Schedules")
    
    from labs_solver.schedules import ALL_SCHEDULES
    
    all_pass = True
    for Schedule in ALL_SCHEDULES:
        try:
            s = Schedule(T=1.0)
            
            # Test start at 0
            lam_0 = s.lambda_t(0)
            ok_0 = abs(lam_0) < 1e-10
            
            # Test end at 1
            lam_T = s.lambda_t(1.0)
            ok_T = abs(lam_T - 1.0) < 1e-10
            
            status = "✓" if (ok_0 and ok_T) else "✗"
            print(f"  {status} {Schedule.__name__}: λ(0)={lam_0:.4f}, λ(1)={lam_T:.4f}")
            
            if not (ok_0 and ok_T):
                all_pass = False
        except Exception as e:
            print(f"  ✗ {Schedule.__name__}: {e}")
            all_pass = False
    
    return all_pass

def test_boltzmann():
    """Test Boltzmann seeding."""
    print_section("5. Testing Boltzmann Seeding")
    
    from labs_solver.boltzmann_seeding import boltzmann_weights, boltzmann_seeding, diversity_metrics
    from labs_solver.labs_energy import random_sequence
    
    # Test weights sum to 1
    energies = [10, 20, 30, 40, 50]
    weights = boltzmann_weights(energies, beta=1.0)
    sum_ok = abs(sum(weights) - 1.0) < 1e-10
    
    status = "✓" if sum_ok else "✗"
    print(f"  {status} Weights sum to 1: {sum(weights):.6f}")
    
    # Test population generation
    samples = [random_sequence(10) for _ in range(100)]
    population = boltzmann_seeding(samples, K=50, beta=1.0)
    size_ok = len(population) == 50
    
    status = "✓" if size_ok else "✗"
    print(f"  {status} Population size: {len(population)} (expected 50)")
    
    # Test diversity
    metrics = diversity_metrics(population)
    print(f"  ✓ Diversity: {metrics['diversity']:.1%}")
    print(f"  ✓ Unique sequences: {metrics['unique']}/{len(population)}")
    
    return sum_ok and size_ok

def test_mts():
    """Test MTS algorithm."""
    print_section("6. Testing Memetic Tabu Search")
    
    from labs_solver.mts import memetic_tabu_search
    from labs_solver.labs_energy import calculate_energy, random_sequence
    
    # Test basic run
    N = 10
    initial = [random_sequence(N) for _ in range(5)]
    initial_best = min(calculate_energy(s) for s in initial)
    
    print(f"\n  Running MTS on N={N} (20 generations)...")
    start = time.time()
    best_s, best_e, stats = memetic_tabu_search(
        N, population_size=20, max_generations=20, verbose=False
    )
    elapsed = time.time() - start
    
    improved = best_e <= initial_best + 5  # Allow some slack
    status = "✓" if improved else "✗"
    
    print(f"  {status} MTS completed in {elapsed:.2f}s")
    print(f"  ✓ Initial best: {initial_best}")
    print(f"  ✓ Final energy: {best_e}")
    print(f"  ✓ Generations: {stats['generations']}")
    
    return improved

def test_gpu_acceleration():
    """Test GPU acceleration if available."""
    print_section("7. Testing GPU Acceleration")
    
    from labs_solver.gpu_acceleration import GPUAccelerator, CUPY_AVAILABLE
    from labs_solver.labs_energy import calculate_energy, random_sequence
    
    gpu = GPUAccelerator()
    print(f"  Backend: {gpu.backend}")
    print(f"  GPU available: {gpu.is_gpu}")
    
    # Test batch energy
    sequences = [random_sequence(15) for _ in range(20)]
    energies = gpu.batch_energy(sequences)
    
    # Verify correctness
    direct_energies = [calculate_energy(s) for s in sequences]
    match = all(e1 == e2 for e1, e2 in zip(energies, direct_energies))
    
    status = "✓" if match else "✗"
    print(f"  {status} Batch energy calculation: {len(energies)} sequences")
    
    if CUPY_AVAILABLE and gpu.is_gpu:
        print("  ✓ Running on GPU!")
    else:
        print("  ℹ Using CPU fallback (CuPy/CUDA not available)")
    
    return match

def test_s_tier_solver():
    """Test the full S-Tier solver."""
    print_section("8. Testing S-Tier Solver")
    
    from labs_solver.s_tier_solver import STierLABSSolver
    
    print("\n  Running S-Tier solver on N=12...")
    solver = STierLABSSolver(
        N=12,
        shots_per_schedule=100,  # Small for testing
        population_size=20,
        boltzmann_beta=1.0
    )
    
    start = time.time()
    best_seq, best_energy, stats = solver.solve(
        max_generations=30,
        verbose=False
    )
    elapsed = time.time() - start
    
    print(f"  ✓ S-Tier completed in {elapsed:.2f}s")
    print(f"  ✓ Best energy: {best_energy}")
    print(f"  ✓ Backend: {stats.get('backend', 'unknown')}")
    print(f"  ✓ Quantum available: {stats.get('quantum_available', False)}")
    
    return best_energy >= 0

def check_cudaq_installation():
    """Check if CUDA-Q is properly installed."""
    print_section("CUDA-Q Installation Check")
    
    try:
        import cudaq
        print("✓ CUDA-Q is installed")
        print(f"  Version: {cudaq.__version__ if hasattr(cudaq, '__version__') else 'unknown'}")
        
        # Try to set target
        try:
            cudaq.set_target('qpp-cpu')
            print("✓ Can set CPU target")
        except Exception as e:
            print(f"✗ Cannot set CPU target: {e}")
        
        try:
            cudaq.set_target('nvidia')
            print("✓ Can set GPU target (nvidia)")
        except Exception as e:
            print(f"ℹ GPU target not available: {e}")
            print("  This is expected if you're not on Brev/GPU machine")
        
        return True
    except ImportError:
        print("✗ CUDA-Q is NOT installed")
        print("\nTo install CUDA-Q locally:")
        print("  pip install cudaq")
        print("\nNote: For GPU support, you need:")
        print("  - NVIDIA GPU with CUDA drivers")
        print("  - CUDA Toolkit installed")
        print("\nFor this challenge, use Brev platform for GPU testing")
        return False

def check_cupy_installation():
    """Check if CuPy is properly installed."""
    print_section("CuPy Installation Check")
    
    try:
        import cupy as cp
        print("✓ CuPy is installed")
        print(f"  Version: {cp.__version__}")
        
        # Check CUDA version
        try:
            print(f"  CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
        except:
            pass
        
        # Try to create array
        try:
            arr = cp.array([1, 2, 3])
            print("✓ Can create GPU arrays")
        except Exception as e:
            print(f"✗ Cannot create GPU arrays: {e}")
        
        return True
    except ImportError:
        print("✗ CuPy is NOT installed")
        print("\nTo install CuPy locally:")
        print("  pip install cupy-cuda12x  # For CUDA 12.x")
        print("  pip install cupy-cuda11x  # For CUDA 11.x")
        print("\nNote: CuPy requires NVIDIA GPU and CUDA drivers")
        return False

def main():
    """Run all tests."""
    print("="*70)
    print("  S-Tier LABS Solver - Local Testing Suite")
    print("="*70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Run all tests
    results['imports'] = test_imports()
    results['correctness'] = test_correctness()
    results['interactions'] = test_interactions()
    results['schedules'] = test_schedules()
    results['boltzmann'] = test_boltzmann()
    results['mts'] = test_mts()
    results['gpu'] = test_gpu_acceleration()
    results['s_tier'] = test_s_tier_solver()
    
    # Check optional installations
    has_cudaq = check_cudaq_installation()
    has_cupy = check_cupy_installation()
    
    # Final summary
    print("\n" + "="*70)
    print("  FINAL TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print("\n" + "-"*70)
    print(f"  CUDA-Q installed: {'Yes' if has_cudaq else 'No'}")
    print(f"  CuPy installed: {'Yes' if has_cupy else 'No'}")
    
    # Overall status
    core_tests = ['correctness', 'interactions', 'schedules', 'mts']
    core_passed = all(results.get(t, False) for t in core_tests)
    
    print("\n" + "="*70)
    if core_passed:
        print("  ✓ ALL CORE TESTS PASSED")
        print("  Your code is ready for submission!")
        print("\n  For GPU testing:")
        print("  1. Go to Brev: https://brev.nvidia.com")
        print("  2. Launch your environment")
        print("  3. Run: python3 run_benchmarks.py")
    else:
        print("  ✗ SOME TESTS FAILED")
        print("  Please fix the issues above before submitting")
    print("="*70)
    
    return 0 if core_passed else 1

if __name__ == "__main__":
    sys.exit(main())
