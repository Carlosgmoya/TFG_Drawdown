from .initialization import initialize_population
from .evaluation import evaluate, precompute_objectives
from .operators import binary_tournament, crossover, mutation
from .fitness import calculate_total_fitness
from .visualization import plot_pareto_front
from .normalization import normalize_objectives
from .projection import projection_simplex
from .data_loader import load_dataset