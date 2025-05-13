
"""Sparrow Search Algorithm (SSA) for hyper‑parameter optimisation."""
import numpy as np
from typing import Tuple, Callable

class SSA:
    """SSA optimiser for minimising an objective function.

    Attributes
    ----------
    bounds : list[Tuple[float, float]]
        Search bounds for each parameter.
    pop_size : int
        Population size.
    max_iter : int
        Maximum iterations.
    objective : Callable[[np.ndarray], float]
        Objective function to minimise.
    """

    def __init__(self, bounds, pop_size=30, max_iter=50, objective: Callable[[np.ndarray], float] = None,
                 rng: np.random.Generator | None = None):
        self.bounds = np.array(bounds, dtype=float)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.objective = objective
        self.rng = rng or np.random.default_rng()
        self.dim = self.bounds.shape[0]

    def optimise(self, *obj_args, **obj_kwargs):
        lb, ub = self.bounds[:,0], self.bounds[:,1]
        pop = self.rng.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([self.objective(ind, *obj_args, **obj_kwargs) for ind in pop])
        best_idx = np.argmin(fitness)
        global_best = pop[best_idx].copy()
        global_best_fit = fitness[best_idx]

        for t in range(self.max_iter):
            # Discoverer and follower division
            r = self.rng.random(self.pop_size)
            discoverers = pop[r < 0.2]
            followers = pop[r >= 0.2]

            # Update discoverers
            discoverers += self.rng.normal(0, 1, discoverers.shape) * (discoverers - global_best)
            discoverers = np.clip(discoverers, lb, ub)

            # Update followers towards best
            followers += self.rng.random() * (global_best - followers)
            followers = np.clip(followers, lb, ub)

            pop = np.vstack([discoverers, followers])
            fitness = np.array([self.objective(ind, *obj_args, **obj_kwargs) for ind in pop])

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < global_best_fit:
                global_best = pop[best_idx].copy()
                global_best_fit = fitness[best_idx]

        return global_best, global_best_fit
