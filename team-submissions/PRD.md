# Product Requirements Document (PRD)

**Project Name:** S-Tier LABS Solver  
**Team Name:** Duos Quantum Innovator  
**GitHub Repository:** [iQuHACK 2026 NVIDIA Fork]  

---

## Executive Summary

This project implements a novel quantum-enhanced solver for the Low Autocorrelation Binary Sequences (LABS) problem. Building on the state-of-the-art QE-MTS algorithm from "Scaling advantage with quantum-enhanced memetic tabu search" (arXiv:2511.04553), we introduce two key innovations:

1. **Multi-Schedule DCQO**: Run counterdiabatic optimization with multiple annealing schedules to explore different regions of the energy landscape
2. **Boltzmann-Weighted Seeding**: Replace greedy sample replication with energy-weighted population sampling for improved diversity

---

## 1. Team Roles & Responsibilities

| Role | Name | Responsibility |
| :--- | :--- | :--- |
| **Project Lead** | [Duos] | Architecture, planning, final decisions |
| **GPU Acceleration PIC** | [Duos] | CUDA-Q backend, resource management |
| **Quality Assurance PIC** | [Duos] | Testing, verification, CI/CD |
| **Technical Marketing PIC** | [Duos] | Benchmarks, visualization, presentation |

---

## 2. The Architecture

### 2.1 Problem Definition

The LABS problem seeks binary sequences s ∈ {±1}^N that minimize:

```
E(s) = Σ_{k=1}^{N-1} C_k²
where C_k = Σ_{i=1}^{N-k} s_i × s_{i+k}
```

**Key Properties:**
- Exponential search space: 2^N configurations
- NP-hard combinatorial optimization
- Rugged energy landscape with O(N³) energy levels
- Symmetries: E(s) = E(-s) = E(reverse(s))

### 2.2 Choice of Quantum Algorithm

**Algorithm:** Digitized Counterdiabatic Quantum Optimization (DCQO)

**Why not QAOA?**
| Metric | QAOA (p=12) | DCQO |
|--------|-------------|------|
| Entangling gates (N=67) | 1.4M | 236K |
| Variational parameters | 24 | 0 (analytical) |
| Scaling | O(1.46^N) | O(1.24^N) with MTS |

DCQO is 6× more gate-efficient and produces better samples for the same circuit depth.

**Our Innovation 1: Multi-Schedule DCQO**

The paper uses only sin²(πt/2T) schedule. We hypothesize that different schedules explore different parts of the energy landscape:

| Schedule | λ(t) | Characteristic |
|----------|------|----------------|
| Sin² | sin²(πt/2T) | Standard, smooth |
| Linear | t/T | Constant rate |
| Quadratic | (t/T)² | Slow start |
| Smooth-step | 3x² - 2x³ | S-curve |
| Cubic | (t/T)³ | Very slow start |

**Hypothesis:** Combining samples from multiple schedules increases population diversity and exploration.

**Our Innovation 2: Boltzmann-Weighted Seeding**

The paper replicates the best quantum sample K times:
```
P(s) = 1 if s = argmin(E), else 0
```

We use:
```
P(s) ∝ exp(-β × E(s))
```

**Benefits:**
- Preserves quality (prefers low energy)
- Increases diversity (includes good-but-not-best samples)
- Tunable via β parameter

### 2.3 Literature Review

| Reference | Relevance |
|-----------|-----------|
| "Scaling advantage with QE-MTS" (arXiv:2511.04553) | Base algorithm, O(1.24^N) scaling claim |
| "Counterdiabatic driving" (various) | Theoretical foundation for DCQO |
| "Warm-start QAOA" literature | Inspiration for non-greedy initialization |
| "Population diversity in EAs" | Motivation for Boltzmann seeding |

---

## 3. The Acceleration Strategy

### 3.1 Quantum Acceleration (CUDA-Q)

**Strategy:**
1. Use `nvidia` backend for single-GPU acceleration
2. Batch quantum samples across schedules
3. Target `nvidia-mgpu` for N > 25 if multiple GPUs available

**Implementation:**
```python
cudaq.set_target('nvidia')  # GPU simulation
result = cudaq.sample(trotterized_circuit, ..., shots_count=10000)
```

### 3.2 Classical Acceleration (MTS)

**Strategy:**
1. Vectorized energy calculation using NumPy
2. CuPy for GPU-accelerated batch neighbor evaluation
3. Parallel population updates

**Implementation:**
```python
# Vectorized autocorrelation
for k in range(1, N):
    C_k = np.sum(s[:N-k] * s[k:])
    energy += C_k ** 2
```

### 3.3 Hardware Targets

| Environment | Hardware | Use Case |
|-------------|----------|----------|
| Development | qBraid CPU | Logic validation, small N |
| Testing | Brev L4 | GPU porting, medium N |
| Production | Brev A100 | Final benchmarks, large N |

---

## 4. The Verification Plan

### 4.1 Unit Testing Strategy

**Framework:** pytest + custom test runner

**Test Coverage:**
| Module | # Tests | Coverage |
|--------|---------|----------|
| labs_energy.py | 12 | Energy, symmetries |
| tabu_search.py | 4 | Local search correctness |
| mts.py | 5 | Population evolution |
| dcqo_kernels.py | 4 | Interaction generation |
| boltzmann_seeding.py | 7 | Weight calculation, diversity |
| schedules.py | 5 | Schedule properties |

### 4.2 Core Correctness Checks

**Check 1: Known Optima**
```python
assert calculate_energy([1, 1, -1]) == 1  # N=3
assert calculate_energy([1, 1, 1, -1]) == 4  # N=4
assert calculate_energy([1, 1, 1, -1, 1]) == 2  # N=5
```

**Check 2: Symmetries**
```python
assert calculate_energy(s) == calculate_energy(-s)
assert calculate_energy(s) == calculate_energy(s[::-1])
```

**Check 3: Interaction Counts**
```python
n_two = (N//2) * (N//2 - 1)  # N even
n_four = (N//12) * (N//2 - 1) * (2*N - 5)  # N even
assert len(G2) == n_two
assert len(G4) == n_four
```

**Check 4: Boltzmann Properties**
```python
weights = boltzmann_weights(energies, beta)
assert abs(sum(weights) - 1.0) < 1e-10  # Normalized
assert all(w >= 0 for w in weights)  # Non-negative
```

**Check 5: Schedule Endpoints**
```python
assert schedule.lambda_t(0) == 0
assert schedule.lambda_t(T) == 1
```

---

## 5. Execution Strategy & Success Metrics

### 5.1 Agentic Workflow

**AI Tools Used:**
- OpenCode/Claude for code generation
- Context from CUDA-Q documentation
- Iterative prompt refinement

**Workflow:**
1. AI generates module
2. Run tests to catch hallucinations
3. Fix issues, regenerate if needed
4. Document wins/fails in AI_REPORT.md

### 5.2 Success Metrics

| Metric | Target | Priority |
|--------|--------|----------|
| Approximation Ratio (N=20) | > 0.9 | HIGH |
| Population Diversity | > 50% unique | HIGH |
| GPU Speedup (vs CPU) | > 5x | MEDIUM |
| Scale (max N) | 30+ | MEDIUM |
| Test Pass Rate | 100% | HIGH |

### 5.3 Visualization Plan

**Plot 1: Method Comparison**
- X: Problem size N
- Y: Best energy found
- Lines: Classical, Single-schedule QE-MTS, S-Tier

**Plot 2: Schedule Analysis**
- X: Schedule type
- Y: Sample energy distribution (violin plot)

**Plot 3: Diversity Impact**
- X: Boltzmann β
- Y: Final energy + Population diversity (dual axis)

**Plot 4: GPU Speedup**
- X: Problem size N
- Y: Time (seconds)
- Lines: CPU, GPU

---

## 6. Resource Management Plan

### 6.1 Credit Allocation

**Total Budget:** $20 Brev credits

| Task | Hardware | Hours | Cost |
|------|----------|-------|------|
| GPU porting | L4 | 3h | $3.00 |
| Testing | L4 | 2h | $2.00 |
| Benchmarks N≤25 | L4 | 2h | $2.00 |
| Benchmarks N>25 | A100 | 3h | $6.00 |
| Buffer | - | - | $7.00 |

### 6.2 Zombie Instance Prevention

- Set 30-minute timer for Brev checks
- Develop entirely on qBraid (free) until tests pass
- Explicit instance shutdown between sessions
- GPU PIC responsible for monitoring

---

## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CUDA-Q kernel errors | Medium | High | Test each kernel individually |
| GPU not available | Low | High | CPU fallback implemented |
| Time constraints | Medium | Medium | Phase 1 = passing grade |
| Multi-schedule overhead | Medium | Low | Can disable if too slow |
| Boltzmann doesn't help | Medium | Low | Document as negative result |

---

## 8. Deliverables Checklist

### Phase 1 (Due 10pm Sat)
- [x] Completed tutorial notebook
- [x] Self-validation section
- [x] PRD.md

### Phase 2 (Due 10am Sun)
- [x] labs_energy.py
- [x] tabu_search.py
- [x] mts.py
- [x] dcqo_kernels.py
- [x] schedules.py
- [x] boltzmann_seeding.py
- [x] s_tier_solver.py
- [x] tests.py
- [ ] benchmarks.py
- [ ] AI_REPORT.md
- [ ] Presentation

---

## Appendix: Algorithm Pseudocode

### S-Tier Solver

```
function S_TIER_SOLVE(N, schedules, β, K, max_gen):
    # Step 1: Multi-schedule DCQO
    all_samples = []
    for schedule in schedules:
        samples = RUN_DCQO(N, schedule)
        all_samples.extend(samples)
    
    # Step 2: Boltzmann seeding
    energies = [E(s) for s in all_samples]
    weights = exp(-β × energies) / Σ exp(-β × energies)
    population = SAMPLE(all_samples, K, weights)
    
    # Step 3: GPU-accelerated MTS
    best = MTS(population, max_gen)
    
    return best
```

### Boltzmann Weighting

```
function BOLTZMANN_WEIGHTS(energies, β):
    E_min = min(energies)
    shifted = energies - E_min
    log_weights = -β × shifted
    weights = exp(log_weights)
    return weights / sum(weights)
```
