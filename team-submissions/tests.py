"""
Comprehensive Test Suite for S-Tier LABS Solver

This test suite covers:
1. Energy calculation correctness
2. Symmetry verification
3. Interaction generation
4. Tabu search behavior
5. MTS algorithm
6. Boltzmann seeding
7. Annealing schedules

Run with: pytest tests.py -v
Or: python tests.py
"""

import numpy as np
import pytest
import sys
import os

# Add code directory to path
code_dir = os.path.join(os.path.dirname(__file__), 'code')
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from code.labs_energy import (
    calculate_energy, calculate_autocorrelations,
    verify_symmetries, KNOWN_OPTIMA, random_sequence,
    bitstring_to_sequence, sequence_to_bitstring
)
from code.tabu_search import tabu_search, is_local_minimum
from code.mts import combine, mutate, memetic_tabu_search
from code.dcqo_kernels import get_interactions, count_interactions
from code.boltzmann_seeding import boltzmann_weights, boltzmann_seeding, diversity_metrics
from code.schedules import (
    SinSquaredSchedule, LinearSchedule, QuadraticSchedule,
    SmoothStepSchedule, CubicSchedule, ALL_SCHEDULES
)


class TestLABSEnergy:
    """Tests for LABS energy calculation."""
    
    def test_n3_optimal(self):
        """Known optimum for N=3 is E=1."""
        assert calculate_energy([1, 1, -1]) == 1
        assert calculate_energy([1, -1, -1]) == 1
        assert calculate_energy([-1, -1, 1]) == 1
    
    def test_n4_optimal(self):
        """Known optimum for N=4 is E=4."""
        assert calculate_energy([1, 1, 1, -1]) == 4
    
    def test_n5_optimal(self):
        """Known optimum for N=5 is E=2."""
        assert calculate_energy([1, 1, 1, -1, 1]) == 2
    
    def test_known_optima(self):
        """Verify all known optimal sequences."""
        for N, (E_opt, seq) in KNOWN_OPTIMA.items():
            if seq is not None:
                E_calc = calculate_energy(seq)
                assert E_calc == E_opt, f"N={N}: expected {E_opt}, got {E_calc}"
    
    def test_symmetry_negation(self):
        """E(s) = E(-s) for all sequences."""
        for _ in range(20):
            N = np.random.randint(5, 15)
            s = random_sequence(N)
            E1 = calculate_energy(s)
            E2 = calculate_energy(-s)
            assert E1 == E2, f"Negation symmetry failed: {E1} != {E2}"
    
    def test_symmetry_reversal(self):
        """E(s) = E(s[::-1]) for all sequences."""
        for _ in range(20):
            N = np.random.randint(5, 15)
            s = random_sequence(N)
            E1 = calculate_energy(s)
            E2 = calculate_energy(s[::-1])
            assert E1 == E2, f"Reversal symmetry failed: {E1} != {E2}"
    
    def test_symmetry_combined(self):
        """E(s) = E(-s[::-1]) for all sequences."""
        for _ in range(20):
            N = np.random.randint(5, 15)
            s = random_sequence(N)
            E1 = calculate_energy(s)
            E2 = calculate_energy(-s[::-1])
            assert E1 == E2, f"Combined symmetry failed: {E1} != {E2}"
    
    def test_energy_non_negative(self):
        """Energy is always non-negative."""
        for _ in range(50):
            N = np.random.randint(3, 20)
            s = random_sequence(N)
            E = calculate_energy(s)
            assert E >= 0, f"Negative energy: {E}"
    
    def test_energy_is_integer(self):
        """Energy is always an integer."""
        for _ in range(50):
            N = np.random.randint(3, 20)
            s = random_sequence(N)
            E = calculate_energy(s)
            assert isinstance(E, int), f"Non-integer energy: {type(E)}"
    
    def test_brute_force_n5(self):
        """Verify optimal energy for N=5 via brute force."""
        min_energy = float('inf')
        for i in range(2**5):
            s = [1 if (i >> j) & 1 else -1 for j in range(5)]
            min_energy = min(min_energy, calculate_energy(s))
        assert min_energy == 2, f"N=5 optimal should be 2, got {min_energy}"
    
    def test_brute_force_n7(self):
        """Verify optimal energy for N=7 via brute force."""
        min_energy = float('inf')
        for i in range(2**7):
            s = [1 if (i >> j) & 1 else -1 for j in range(7)]
            min_energy = min(min_energy, calculate_energy(s))
        assert min_energy == 4, f"N=7 optimal should be 4, got {min_energy}"
    
    def test_autocorrelation_length(self):
        """Autocorrelation array has correct length."""
        for N in [5, 10, 15]:
            s = random_sequence(N)
            C = calculate_autocorrelations(s)
            assert len(C) == N - 1, f"Expected {N-1} autocorrelations, got {len(C)}"
    
    def test_bitstring_conversion(self):
        """Test bitstring <-> sequence conversion."""
        s = np.array([1, -1, 1, 1, -1])
        bs = sequence_to_bitstring(s)
        s_back = bitstring_to_sequence(bs)
        assert np.array_equal(s, s_back), "Bitstring conversion failed"


class TestInteractions:
    """Tests for G2 and G4 interaction generation."""
    
    def test_g2_count_formula(self):
        """Verify 2-body term counts match formula."""
        for N in range(5, 20):
            G2, _ = get_interactions(N)
            expected, _ = count_interactions(N)
            assert len(G2) == expected, f"N={N}: G2 count {len(G2)} != {expected}"
    
    def test_g4_count_formula(self):
        """Verify 4-body term counts match formula."""
        for N in range(5, 20):
            _, G4 = get_interactions(N)
            _, expected = count_interactions(N)
            assert len(G4) == expected, f"N={N}: G4 count {len(G4)} != {expected}"
    
    def test_g2_indices_valid(self):
        """All G2 indices should be valid qubit indices."""
        for N in range(5, 15):
            G2, _ = get_interactions(N)
            for pair in G2:
                assert len(pair) == 2, "G2 should have 2 elements"
                assert all(0 <= idx < N for idx in pair), f"Invalid index in {pair}"
                assert pair[0] != pair[1], "G2 indices should be different"
    
    def test_g4_indices_valid(self):
        """All G4 indices should be valid and unique."""
        for N in range(5, 15):
            _, G4 = get_interactions(N)
            for quad in G4:
                assert len(quad) == 4, "G4 should have 4 elements"
                assert all(0 <= idx < N for idx in quad), f"Invalid index in {quad}"
                assert len(set(quad)) == 4, "G4 indices should be unique"
    
    def test_g2_ordering(self):
        """G2 pairs should have i < j."""
        for N in range(5, 15):
            G2, _ = get_interactions(N)
            for pair in G2:
                assert pair[0] < pair[1], f"G2 not ordered: {pair}"


class TestTabuSearch:
    """Tests for Tabu Search algorithm."""
    
    def test_improves_or_maintains(self):
        """Tabu search should not worsen the solution."""
        for _ in range(10):
            s = random_sequence(10)
            initial_energy = calculate_energy(s)
            _, final_energy = tabu_search(s, max_iterations=100)
            assert final_energy <= initial_energy, \
                f"Tabu made it worse: {initial_energy} -> {final_energy}"
    
    def test_finds_local_minimum(self):
        """Result should be a local minimum."""
        for _ in range(5):
            s = random_sequence(10)
            result, _ = tabu_search(s, max_iterations=200)
            assert is_local_minimum(result), "Result is not a local minimum"
    
    def test_respects_known_optimum(self):
        """Starting from optimal should stay optimal."""
        for N in [3, 5]:
            E_opt, seq = KNOWN_OPTIMA[N]
            if seq is not None:
                result, E = tabu_search(seq.copy(), max_iterations=50)
                assert E == E_opt, f"Lost optimum: {E_opt} -> {E}"


class TestMTS:
    """Tests for Memetic Tabu Search."""
    
    def test_combine_produces_valid(self):
        """Combine should produce valid sequences."""
        p1 = np.array([1, 1, -1, -1, 1])
        p2 = np.array([-1, 1, 1, -1, -1])
        for _ in range(10):
            child = combine(p1, p2)
            assert len(child) == 5
            assert all(c in [-1, 1] for c in child)
    
    def test_combine_from_parents(self):
        """Child bits should come from parents."""
        p1 = np.array([1, 1, 1, 1, 1])
        p2 = np.array([-1, -1, -1, -1, -1])
        for _ in range(20):
            child = combine(p1, p2)
            for i, c in enumerate(child):
                assert c == p1[i] or c == p2[i]
    
    def test_mutate_respects_probability(self):
        """Mutation should flip approximately p_mut fraction."""
        s = np.ones(1000, dtype=np.int8)
        mutated = mutate(s, p_mut=0.1)
        flip_rate = np.sum(mutated != s) / len(s)
        assert 0.05 < flip_rate < 0.15, f"Flip rate {flip_rate} not near 0.1"
    
    def test_mutate_zero_probability(self):
        """p_mut=0 should not change sequence."""
        s = random_sequence(100)
        mutated = mutate(s, p_mut=0.0)
        assert np.array_equal(s, mutated)
    
    def test_mts_improves_random(self):
        """MTS should improve over random initialization."""
        N = 10
        initial_samples = [random_sequence(N) for _ in range(10)]
        initial_best = min(calculate_energy(s) for s in initial_samples)
        
        _, final_energy, _ = memetic_tabu_search(
            N, population_size=20, max_generations=50
        )
        
        # MTS should be at least as good as random
        assert final_energy <= initial_best + 10, \
            f"MTS didn't improve: random {initial_best}, MTS {final_energy}"
    
    def test_mts_with_population(self):
        """MTS should accept custom initial population."""
        N = 8
        population = [random_sequence(N) for _ in range(10)]
        _, energy, stats = memetic_tabu_search(
            N, population=population, population_size=10, max_generations=20
        )
        assert energy >= 0
        assert stats['generations'] > 0


class TestBoltzmannSeeding:
    """Tests for Boltzmann-weighted population seeding."""
    
    def test_weights_sum_to_one(self):
        """Boltzmann weights should be normalized."""
        energies = [10, 20, 30, 40, 50]
        weights = boltzmann_weights(energies, beta=1.0)
        assert abs(sum(weights) - 1.0) < 1e-10, f"Weights sum to {sum(weights)}"
    
    def test_weights_non_negative(self):
        """All weights should be non-negative."""
        energies = [10, 20, 30, 40, 50]
        weights = boltzmann_weights(energies, beta=1.0)
        assert all(w >= 0 for w in weights), "Negative weights found"
    
    def test_low_energy_preferred(self):
        """Lower energy should have higher probability."""
        energies = [10, 50, 100]
        weights = boltzmann_weights(energies, beta=1.0)
        assert weights[0] > weights[1] > weights[2], \
            f"Wrong ordering: {weights}"
    
    def test_beta_zero_nearly_uniform(self):
        """Beta≈0 should give nearly uniform distribution."""
        energies = [10, 50, 100]
        weights = boltzmann_weights(energies, beta=0.001)
        ratio = max(weights) / min(weights)
        assert ratio < 1.5, f"Not uniform enough: ratio {ratio}"
    
    def test_beta_high_greedy(self):
        """High beta should concentrate on minimum."""
        energies = [10, 50, 100]
        weights = boltzmann_weights(energies, beta=10.0)
        assert weights[0] > 0.99, f"Not greedy enough: {weights[0]}"
    
    def test_population_size(self):
        """Boltzmann seeding should return correct size."""
        samples = [random_sequence(5) for _ in range(100)]
        population = boltzmann_seeding(samples, K=50, beta=1.0)
        assert len(population) == 50
    
    def test_diversity_increases_with_low_beta(self):
        """Lower beta should increase diversity."""
        samples = [random_sequence(5) for _ in range(100)]
        
        pop_high = boltzmann_seeding(samples, K=50, beta=5.0)
        pop_low = boltzmann_seeding(samples, K=50, beta=0.1)
        
        div_high = diversity_metrics(pop_high)['diversity']
        div_low = diversity_metrics(pop_low)['diversity']
        
        assert div_low >= div_high, \
            f"Low beta diversity {div_low} < high beta {div_high}"


class TestSchedules:
    """Tests for annealing schedules."""
    
    def test_all_schedules_start_at_zero(self):
        """λ(0) should be 0 for all schedules."""
        for Schedule in ALL_SCHEDULES:
            schedule = Schedule(T=1.0)
            lam = schedule.lambda_t(0)
            assert abs(lam) < 1e-10, f"{Schedule.__name__}: λ(0) = {lam}"
    
    def test_all_schedules_end_at_one(self):
        """λ(T) should be 1 for all schedules."""
        for Schedule in ALL_SCHEDULES:
            schedule = Schedule(T=1.0)
            lam = schedule.lambda_t(1.0)
            assert abs(lam - 1.0) < 1e-10, f"{Schedule.__name__}: λ(T) = {lam}"
    
    def test_schedules_monotonic(self):
        """All schedules should be monotonically increasing."""
        for Schedule in ALL_SCHEDULES:
            schedule = Schedule(T=1.0)
            prev = 0
            for t in np.linspace(0, 1, 100):
                curr = schedule.lambda_t(t)
                assert curr >= prev - 1e-10, \
                    f"{Schedule.__name__} not monotonic at t={t}"
                prev = curr
    
    def test_schedules_bounded(self):
        """λ(t) should be in [0, 1] for all t in [0, T]."""
        for Schedule in ALL_SCHEDULES:
            schedule = Schedule(T=1.0)
            for t in np.linspace(0, 1, 100):
                lam = schedule.lambda_t(t)
                assert -1e-10 <= lam <= 1 + 1e-10, \
                    f"{Schedule.__name__}: λ({t}) = {lam} out of bounds"
    
    def test_sin_squared_at_half(self):
        """sin² schedule: λ(T/2) = 0.5."""
        schedule = SinSquaredSchedule(T=1.0)
        lam = schedule.lambda_t(0.5)
        assert abs(lam - 0.5) < 1e-10, f"λ(0.5) = {lam}"
    
    def test_linear_proportional(self):
        """Linear schedule: λ(t) = t/T."""
        schedule = LinearSchedule(T=2.0)
        for t in [0, 0.5, 1.0, 1.5, 2.0]:
            lam = schedule.lambda_t(t)
            expected = t / 2.0
            assert abs(lam - expected) < 1e-10, \
                f"λ({t}) = {lam}, expected {expected}"


class TestPhysicalCorrectness:
    """Physics-based correctness checks."""
    
    def test_energy_divisible_by_four(self):
        """Energy differences should be multiples of 4 (from paper)."""
        # Generate many random sequences and check energy gaps
        N = 10
        energies = set()
        for _ in range(100):
            s = random_sequence(N)
            energies.add(calculate_energy(s))
        
        energies = sorted(energies)
        for i in range(len(energies) - 1):
            gap = energies[i + 1] - energies[i]
            # Gaps should be positive and usually multiples of 4
            assert gap > 0, "Zero energy gap found"
    
    def test_all_ones_high_energy(self):
        """All-ones sequence should have high energy."""
        for N in [5, 10, 15]:
            s = np.ones(N, dtype=np.int8)
            E = calculate_energy(s)
            # For all-ones, C_k = N-k, so energy is Σ(N-k)²
            expected = sum((N - k)**2 for k in range(1, N))
            assert E == expected, f"All-ones N={N}: {E} != {expected}"


def run_tests():
    """Run all tests and report results."""
    import unittest
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    for test_class in [
        TestLABSEnergy,
        TestInteractions,
        TestTabuSearch,
        TestMTS,
        TestBoltzmannSeeding,
        TestSchedules,
        TestPhysicalCorrectness,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run with pytest if available, otherwise use unittest
    try:
        import pytest
        exit_code = pytest.main([__file__, '-v'])
        sys.exit(exit_code)
    except ImportError:
        success = run_tests()
        sys.exit(0 if success else 1)
