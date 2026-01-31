# S-Tier LABS Solver
# iQuHACK 2026 NVIDIA Challenge
#
# Innovations:
# 1. Multi-Schedule DCQO - Multiple annealing schedules for diverse sampling
# 2. Boltzmann Seeding - Energy-weighted population initialization
# 3. GPU Acceleration - CUDA-Q + CuPy for full GPU pipeline

from .labs_energy import calculate_energy, verify_symmetries, KNOWN_OPTIMA
from .tabu_search import tabu_search
from .mts import memetic_tabu_search, combine, mutate

__version__ = "1.0.0"
__author__ = "iQuHACK Team"
