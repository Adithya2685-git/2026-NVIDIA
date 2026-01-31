"""
Annealing Schedules for Counterdiabatic Quantum Optimization

This module provides multiple annealing schedules for the DCQO algorithm.
The paper uses sin²(πt/2T), but we implement additional schedules to
explore whether different schedules produce better samples.

Key insight: Different schedules may explore different regions of the
energy landscape, leading to more diverse quantum samples.

Available schedules:
1. SinSquared: sin²(πt/2T) - Standard from paper
2. Linear: t/T - Simple linear ramp
3. Quadratic: (t/T)² - Slow start, fast finish
4. SmoothStep: 3(t/T)² - 2(t/T)³ - S-curve
5. Cubic: (t/T)³ - Very slow start
"""

import numpy as np
from math import pi, sin, cos
from typing import Callable, List
from abc import ABC, abstractmethod


class AnnealingSchedule(ABC):
    """
    Base class for annealing schedules.
    
    An annealing schedule defines how the mixing parameter λ(t) evolves
    from 0 to 1 as time goes from 0 to T.
    
    The Hamiltonian evolves as:
        H(t) = (1 - λ(t)) H_i + λ(t) H_f
    
    For counterdiabatic optimization, we need λ̇(t) = dλ/dt.
    """
    
    def __init__(self, T: float = 1.0):
        """
        Initialize schedule.
        
        Args:
            T: Total evolution time
        """
        self.T = T
        self.name = self.__class__.__name__
    
    @abstractmethod
    def lambda_t(self, t: float) -> float:
        """
        Calculate λ(t): annealing parameter from 0 to 1.
        
        Args:
            t: Time (0 ≤ t ≤ T)
        
        Returns:
            float: λ(t) ∈ [0, 1]
        """
        pass
    
    @abstractmethod
    def lambda_dot(self, t: float) -> float:
        """
        Calculate dλ/dt: derivative of annealing parameter.
        
        Args:
            t: Time (0 ≤ t ≤ T)
        
        Returns:
            float: dλ/dt
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.name}(T={self.T})"
    
    def sample_schedule(self, n_points: int = 100) -> np.ndarray:
        """
        Sample the schedule at n_points equally spaced times.
        
        Returns:
            np.ndarray: Array of shape (n_points, 2) with [t, λ(t)]
        """
        times = np.linspace(0, self.T, n_points)
        lambdas = np.array([self.lambda_t(t) for t in times])
        return np.column_stack([times, lambdas])


class SinSquaredSchedule(AnnealingSchedule):
    """
    Standard schedule from the paper: λ(t) = sin²(πt/2T)
    
    Properties:
    - Smooth start and finish (zero derivative at endpoints)
    - Most commonly used in adiabatic optimization
    - λ̇(t) = (π/2T) × sin(πt/T)
    """
    
    def lambda_t(self, t: float) -> float:
        arg = (pi * t) / (2 * self.T)
        return sin(arg) ** 2
    
    def lambda_dot(self, t: float) -> float:
        return (pi / (2 * self.T)) * sin((pi * t) / self.T)


class LinearSchedule(AnnealingSchedule):
    """
    Simple linear schedule: λ(t) = t/T
    
    Properties:
    - Constant rate of change
    - Non-smooth at endpoints
    - λ̇(t) = 1/T
    """
    
    def lambda_t(self, t: float) -> float:
        return t / self.T
    
    def lambda_dot(self, t: float) -> float:
        return 1.0 / self.T


class QuadraticSchedule(AnnealingSchedule):
    """
    Quadratic schedule: λ(t) = (t/T)²
    
    Properties:
    - Slow start, fast finish
    - Zero derivative at t=0
    - λ̇(t) = 2t/T²
    """
    
    def lambda_t(self, t: float) -> float:
        return (t / self.T) ** 2
    
    def lambda_dot(self, t: float) -> float:
        return (2 * t) / (self.T ** 2)


class SmoothStepSchedule(AnnealingSchedule):
    """
    Smooth step (S-curve) schedule: λ(t) = 3(t/T)² - 2(t/T)³
    
    Properties:
    - Zero derivative at both endpoints (very smooth)
    - Faster in the middle, slower at edges
    - λ̇(t) = (6t/T² - 6t²/T³) = 6t(T-t)/T³
    """
    
    def lambda_t(self, t: float) -> float:
        x = t / self.T
        return 3 * x**2 - 2 * x**3
    
    def lambda_dot(self, t: float) -> float:
        x = t / self.T
        return (6 * x - 6 * x**2) / self.T


class CubicSchedule(AnnealingSchedule):
    """
    Cubic schedule: λ(t) = (t/T)³
    
    Properties:
    - Very slow start
    - Zero first and second derivatives at t=0
    - λ̇(t) = 3t²/T³
    """
    
    def lambda_t(self, t: float) -> float:
        return (t / self.T) ** 3
    
    def lambda_dot(self, t: float) -> float:
        return (3 * t**2) / (self.T ** 3)


class InverseCubicSchedule(AnnealingSchedule):
    """
    Inverse cubic: λ(t) = 1 - (1 - t/T)³
    
    Properties:
    - Fast start, slow finish
    - Spends more time near the final Hamiltonian
    - λ̇(t) = 3(1 - t/T)²/T
    """
    
    def lambda_t(self, t: float) -> float:
        x = t / self.T
        return 1 - (1 - x)**3
    
    def lambda_dot(self, t: float) -> float:
        x = t / self.T
        return 3 * (1 - x)**2 / self.T


class ExponentialSchedule(AnnealingSchedule):
    """
    Exponential approach: λ(t) = 1 - exp(-3t/T)
    
    Properties:
    - Fast initial ramp, asymptotic approach to 1
    - λ̇(t) = (3/T) exp(-3t/T)
    """
    
    def lambda_t(self, t: float) -> float:
        return 1 - np.exp(-3 * t / self.T)
    
    def lambda_dot(self, t: float) -> float:
        return (3 / self.T) * np.exp(-3 * t / self.T)


# All available schedules
ALL_SCHEDULES = [
    SinSquaredSchedule,
    LinearSchedule,
    QuadraticSchedule,
    SmoothStepSchedule,
    CubicSchedule,
]

# Extended schedules (includes more options)
EXTENDED_SCHEDULES = ALL_SCHEDULES + [
    InverseCubicSchedule,
    ExponentialSchedule,
]

# Default schedule (paper's choice)
DEFAULT_SCHEDULE = SinSquaredSchedule


def compute_thetas_with_schedule(
    N: int,
    G2: List[List[int]],
    G4: List[List[int]],
    schedule: AnnealingSchedule,
    n_steps: int = 1
) -> List[float]:
    """
    Compute theta values for each Trotter step using a given schedule.
    
    θ(t) = Δt × α(t) × λ̇(t)
    
    Where α(t) is computed from Gamma1/Gamma2 (see labs_utils.py).
    
    Args:
        N: Sequence length
        G2: 2-body interactions
        G4: 4-body interactions
        schedule: Annealing schedule to use
        n_steps: Number of Trotter steps
    
    Returns:
        List of theta values, one per Trotter step
    """
    import sys
    import os
    
    # Try to import the official compute_theta
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    utils_path = os.path.join(repo_root, 'tutorial_notebook', 'auxiliary_files')
    if utils_path not in sys.path:
        sys.path.insert(0, utils_path)
    
    try:
        from labs_utils import compute_theta
        use_official = True
    except ImportError:
        use_official = False
    
    T = schedule.T
    dt = T / n_steps
    thetas = []
    
    for step in range(1, n_steps + 1):
        t = step * dt
        
        if use_official:
            # Use official computation
            theta_val = compute_theta(t, dt, T, N, G2, G4)
        else:
            # Simplified computation
            lam_dot = schedule.lambda_dot(t)
            
            # Simplified alpha (constant approximation)
            # In practice, alpha depends on the Hamiltonian structure
            alpha = 0.1
            
            theta_val = dt * alpha * lam_dot
        
        thetas.append(float(theta_val))
    
    return thetas


def visualize_schedules(schedules: List[type] = None, T: float = 1.0, n_points: int = 100):
    """
    Create a comparison plot of different schedules.
    
    Args:
        schedules: List of schedule classes to compare
        T: Total time
        n_points: Number of points to sample
    
    Returns:
        matplotlib figure (or None if matplotlib not available)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return None
    
    if schedules is None:
        schedules = ALL_SCHEDULES
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    times = np.linspace(0, T, n_points)
    
    for ScheduleClass in schedules:
        schedule = ScheduleClass(T=T)
        lambdas = [schedule.lambda_t(t) for t in times]
        lambda_dots = [schedule.lambda_dot(t) for t in times]
        
        ax1.plot(times, lambdas, label=schedule.name)
        ax2.plot(times, lambda_dots, label=schedule.name)
    
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('λ(t)')
    ax1.set_title('Annealing Schedules')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('dλ/dt')
    ax2.set_title('Schedule Derivatives')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Quick test
    print("Testing Annealing Schedules...")
    
    T = 1.0
    test_times = [0, 0.25, 0.5, 0.75, 1.0]
    
    for ScheduleClass in ALL_SCHEDULES:
        schedule = ScheduleClass(T=T)
        print(f"\n{schedule.name}:")
        for t in test_times:
            lam = schedule.lambda_t(t)
            lam_dot = schedule.lambda_dot(t)
            print(f"  t={t:.2f}: λ={lam:.4f}, λ̇={lam_dot:.4f}")
    
    # Verify endpoints
    print("\n\nEndpoint verification:")
    for ScheduleClass in ALL_SCHEDULES:
        schedule = ScheduleClass(T=1.0)
        l0 = schedule.lambda_t(0)
        l1 = schedule.lambda_t(1.0)
        print(f"{schedule.name}: λ(0)={l0:.6f}, λ(T)={l1:.6f}")
