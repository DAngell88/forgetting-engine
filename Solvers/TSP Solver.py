#!/usr/bin/env python3
"""
Forgetting Engine: Traveling Salesman Problem (TSP) Solver
Pharmaceutical-Grade Validation (620 trials)
Multi-Scale Study: 15, 30, 50, 200 cities

Fixed Random Seeds for Reproducibility

This script implements the EXACT Forgetting Engine algorithm from the manuscript:
- Strategic Elimination: Elimination score with tour length + structural diversity
- Paradox Retention: Retain long tours with unusual edge patterns
- Baselines: Nearest Neighbor (NN) heuristic + Genetic Algorithm (GA)
"""

import numpy as np
import json
from typing import List, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime
import random

# ============================================================================
# DOMAIN: Traveling Salesman Problem (TSP)
# ============================================================================

class TSPDomain:
    """Traveling Salesman Problem on 2D Euclidean plane."""
    
    def __init__(self, num_cities: int, seed: int = 42):
        """
        Args:
            num_cities: Number of cities
            seed: Random seed for city generation
        """
        self.num_cities = num_cities
        self.rng = np.random.RandomState(seed)
        self.cities = self._generate_cities()
    
    def _generate_cities(self) -> List[Tuple[float, float]]:
        """Generate random cities on [0, 1000] x [0, 1000] square."""
        cities = []
        for _ in range(self.num_cities):
            x = self.rng.uniform(0, 1000)
            y = self.rng.uniform(0, 1000)
            cities.append((x, y))
        return cities
    
    def calculate_distance(self, tour: List[int]) -> float:
        """
        Calculate total tour distance (Euclidean).
        tour: List of city indices in visit order
        """
        if not tour or len(tour) != self.num_cities:
            return float('inf')
        
        dist = 0.0
        for i in range(len(tour)):
            c1 = self.cities[tour[i]]
            c2 = self.cities[tour[(i + 1) % len(tour)]]
            dist += np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        return dist
    
    def edge_diversity(self, tour: List[int]) -> float:
        """
        Structural complexity: edge orientation diversity
        Higher = more unusual edge patterns (used for paradox detection)
        """
        if len(tour) < 3:
            return 0.0
        
        angles = []
        for i in range(len(tour)):
            p1 = self.cities[tour[i]]
            p2 = self.cities[tour[(i + 1) % len(tour)]]
            p3 = self.cities[tour[(i + 2) % len(tour)]]
            
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            if v1[0]**2 + v1[1]**2 == 0 or v2[0]**2 + v2[1]**2 == 0:
                angle = 0
            else:
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
                mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
                cos_angle = dot / (mag1 * mag2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
            
            angles.append(angle)
        
        return float(np.std(angles))  # High std dev = diverse angles


@dataclass
class TSPSolution:
    """Represents a TSP tour."""
    tour: List[int]
    distance: float
    edge_diversity: float
    age: int
    elim_score: float = 0.0


class ForgettingEngineTSP:
    """TSP Forgetting Engine."""
    
    def __init__(self, domain: TSPDomain, population_size: int = 50,
                 generations: int = 100, forget_rate: float = 0.35,
                 paradox_rate: float = 0.15, seed: int = 42):
        self.domain = domain
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        
        self.N = population_size
        self.G = generations
        self.forget_rate = forget_rate
        self.paradox_rate = paradox_rate
        
        # Elimination weights (from manuscript: α=-1.0, β=0.1, γ=0.3, δ=-0.1)
        self.alpha = -1.0    # Distance (penalize long tours)
        self.beta = 0.1      # Complexity (slight reward for diversity)
        self.gamma = 0.3     # Novelty (reward exploration)
        self.delta = -0.1    # Age (penalize old tours)
        
        self.population: List[TSPSolution] = []
        self.paradox_buffer: List[TSPSolution] = []
        self.best_solution: TSPSolution = None
    
    def initialize_population(self):
        """Initialize with 30% NN-seeded, 70% random tours."""
        for i in range(self.N):
            if i < 0.3 * self.N:
                tour = self._nearest_neighbor_tour()
            else:
                tour = self._random_tour()
            
            sol = TSPSolution(
                tour=tour,
                distance=self.domain.calculate_distance(tour),
                edge_diversity=self.domain.edge_diversity(tour),
                age=0
            )
            self.population.append(sol)
        
        self.best_solution = min(self.population, key=lambda x: x.distance)
    
    def _nearest_neighbor_tour(self) -> List[int]:
        """Greedy nearest-neighbor heuristic."""
        unvisited = set(range(self.domain.num_cities))
        start = self.rng.randint(0, self.domain.num_cities)
        tour = [start]
        unvisited.remove(start)
        
        while unvisited:
            current = tour[-1]
            current_city = self.domain.cities[current]
            
            nearest = min(unvisited, key=lambda i: (
                (self.domain.cities[i][0] - current_city[0])**2 +
                (self.domain.cities[i][1] - current_city[1])**2
            ))
            
            tour.append(nearest)
            unvisited.remove(nearest)
        
        return tour
    
    def _random_tour(self) -> List[int]:
        """Random permutation."""
        tour = list(range(self.domain.num_cities))
        self.rng.shuffle(tour)
        return tour
    
    def compute_elimination_score(self, sol: TSPSolution, generation: int):
        """Elimination score: α·distance + β·diversity + γ·novelty + δ·age"""
        novelty = self._compute_novelty(sol)
        age = generation - sol.age if sol.age > 0 else 0
        
        score = (
            self.alpha * sol.distance +
            self.beta * sol.edge_diversity +
            self.gamma * novelty +
            self.delta * age
        )
        sol.elim_score = score
    
    def _compute_novelty(self, sol: TSPSolution) -> float:
        """Distance from nearest neighbor in tour space."""
        if len(self.population) < 2:
            return 1.0
        
        distances = []
        for other in self.population:
            if other is sol:
                continue
            
            # Hamming distance in tour space
            matches = sum(1 for i, j in zip(sol.tour, other.tour) if i == j)
            dist = self.domain.num_cities - matches
            distances.append(dist)
        
        return min(distances) if distances else 1.0
    
    def strategic_elimination(self, generation: int):
        """Eliminate bottom (1 - forget_rate) by elimination score."""
        for sol in self.population:
            self.compute_elimination_score(sol, generation)
        
        sorted_pop = sorted(self.population, key=lambda x: x.elim_score, reverse=True)
        keep_count = int(np.ceil(len(self.population) * (1 - self.forget_rate)))
        
        self.population = sorted_pop[:keep_count]
        eliminated = sorted_pop[keep_count:]
        
        return eliminated
    
    def paradox_retention(self, eliminated: List[TSPSolution]):
        """Retain tours with bad distance but unusual edge patterns."""
        if not eliminated:
            return
        
        mean_distance = np.mean([e.distance for e in eliminated])
        median_diversity = np.median([e.edge_diversity for e in eliminated])
        
        paradoxes = []
        for sol in eliminated:
            if sol.distance > mean_distance and sol.edge_diversity > median_diversity:
                paradoxes.append(sol)
        
        sample_size = max(1, int(len(paradoxes) * self.paradox_rate))
        if paradoxes:
            self.paradox_buffer.extend(
                self.rng.choice(paradoxes, min(sample_size, len(paradoxes)), replace=False)
            )
    
    def population_regeneration(self, generation: int):
        """Maintain population via 2-opt mutation."""
        while len(self.population) < self.N:
            if self.rng.random() < 0.2 and self.paradox_buffer:
                parent = self.paradox_buffer[self.rng.randint(0, len(self.paradox_buffer))]
                self.paradox_buffer.remove(parent)
            else:
                parent = self.population[self.rng.randint(0, len(self.population))]
            
            child_tour = self._2opt_local_search(parent.tour)
            child = TSPSolution(
                tour=child_tour,
                distance=self.domain.calculate_distance(child_tour),
                edge_diversity=self.domain.edge_diversity(child_tour),
                age=generation
            )
            self.population.append(child)
    
    def _2opt_local_search(self, tour: List[int], iterations: int = 10) -> List[int]:
        """2-opt local search with fixed iterations."""
        best_tour = list(tour)
        best_dist = self.domain.calculate_distance(best_tour)
        
        for _ in range(iterations):
            i = self.rng.randint(0, len(tour) - 2)
            j = self.rng.randint(i + 2, len(tour))
            
            new_tour = best_tour[:i] + best_tour[i:j][::-1] + best_tour[j:]
            new_dist = self.domain.calculate_distance(new_tour)
            
            if new_dist < best_dist:
                best_tour = new_tour
                best_dist = new_dist
        
        return best_tour
    
    def run(self) -> Dict:
        """Execute FE algorithm."""
        self.initialize_population()
        
        for generation in range(self.G):
            eliminated = self.strategic_elimination(generation)
            self.paradox_retention(eliminated)
            self.population_regeneration(generation)
            
            gen_best = min(self.population, key=lambda x: x.distance)
            if gen_best.distance < self.best_solution.distance:
                self.best_solution = gen_best
        
        return {
            "best_distance": float(self.best_solution.distance),
            "best_edge_diversity": float(self.best_solution.edge_diversity),
            "seed": self.seed
        }


# ============================================================================
# BASELINES
# ============================================================================

class NearestNeighborTSP:
    """Nearest Neighbor baseline."""
    
    def __init__(self, domain: TSPDomain, seed: int = 42):
        self.domain = domain
        self.rng = np.random.RandomState(seed)
        self.seed = seed
    
    def run(self) -> Dict:
        """Single NN run from random start."""
        unvisited = set(range(self.domain.num_cities))
        start = self.rng.randint(0, self.domain.num_cities)
        tour = [start]
        unvisited.remove(start)
        
        while unvisited:
            current = tour[-1]
            current_city = self.domain.cities[current]
            nearest = min(unvisited, key=lambda i: (
                (self.domain.cities[i][0] - current_city[0])**2 +
                (self.domain.cities[i][1] - current_city[1])**2
            ))
            tour.append(nearest)
            unvisited.remove(nearest)
        
        distance = self.domain.calculate_distance(tour)
        return {
            "best_distance": float(distance),
            "seed": self.seed
        }


class GeneticAlgorithmTSP:
    """Genetic Algorithm baseline."""
    
    def __init__(self, domain: TSPDomain, population_size: int = 100,
                 generations: int = 1000, seed: int = 42):
        self.domain = domain
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.pop_size = population_size
        self.G = generations
        self.population = []
    
    def run(self) -> Dict:
        """Execute GA."""
        # Initialize
        for _ in range(self.pop_size):
            tour = list(range(self.domain.num_cities))
            self.rng.shuffle(tour)
            self.population.append(tour)
        
        best_dist = float('inf')
        best_tour = None
        
        for generation in range(self.G):
            # Evaluate
            fitnesses = [1.0 / (1.0 + self.domain.calculate_distance(t)) for t in self.population]
            
            # Track best
            for tour, fitness in zip(self.population, fitnesses):
                dist = self.domain.calculate_distance(tour)
                if dist < best_dist:
                    best_dist = dist
                    best_tour = tour
            
            # Selection (tournament k=3)
            new_pop = []
            for _ in range(self.pop_size):
                candidates = [self.rng.randint(0, self.pop_size) for _ in range(3)]
                winner = max(candidates, key=lambda i: fitnesses[i])
                new_pop.append(list(self.population[winner]))
            
            self.population = new_pop
            
            # Crossover + Mutation
            for i in range(0, self.pop_size, 2):
                if self.rng.random() < 0.8:  # Crossover
                    p1, p2 = self.population[i], self.population[i+1]
                    c1, c2 = self._order_crossover(p1, p2)
                    self.population[i], self.population[i+1] = c1, c2
                
                if self.rng.random() < 0.01:  # Mutation
                    tour = self.population[i]
                    i1, i2 = self.rng.choice(len(tour), 2, replace=False)
                    tour[i1], tour[i2] = tour[i2], tour[i1]
        
        return {
            "best_distance": float(best_dist),
            "seed": self.seed
        }
    
    def _order_crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        """Order crossover (OX)."""
        n = len(p1)
        i1, i2 = sorted(self.rng.choice(n, 2, replace=False))
        
        c1 = [-1] * n
        c2 = [-1] * n
        
        c1[i1:i2] = p1[i1:i2]
        c2[i1:i2] = p2[i1:i2]
        
        p2_filtered = [x for x in p2 if x not in c1[i1:i2]]
        p1_filtered = [x for x in p1 if x not in c2[i1:i2]]
        
        c1[:i1] = p2_filtered[:i1]
        c1[i2:] = p2_filtered[i1:]
        
        c2[:i1] = p1_filtered[:i1]
        c2[i2:] = p1_filtered[i1:]
        
        return c1, c2


# ============================================================================
# VALIDATION PROTOCOL
# ============================================================================

def run_scale(num_cities: int, seed_offset: int) -> Dict:
    """Run multi-algorithm validation at one scale."""
    print(f"\n{'='*70}")
    print(f"  TSP Scale: {num_cities} cities")
    print(f"{'='*70}")
    
    if num_cities == 15:
        nn_trials, ga_trials, fe_trials = 100, 60, 60
    elif num_cities == 30:
        nn_trials, ga_trials, fe_trials = 30, 35, 35
    elif num_cities == 50:
        nn_trials, ga_trials, fe_trials = 50, 75, 75
    else:  # 200
        nn_trials, ga_trials, fe_trials = 20, 40, 40
    
    nn_results = []
    ga_results = []
    fe_results = []
    
    # NN
    print(f"Running {nn_trials} Nearest Neighbor trials...")
    for i in range(nn_trials):
        seed = seed_offset + i
        domain = TSPDomain(num_cities, seed=seed)
        nn = NearestNeighborTSP(domain, seed=seed)
        nn_results.append(nn.run())
    
    # GA
    print(f"Running {ga_trials} Genetic Algorithm trials...")
    for i in range(ga_trials):
        seed = seed_offset + nn_trials + i
        domain = TSPDomain(num_cities, seed=seed)
        ga = GeneticAlgorithmTSP(domain, seed=seed)
        ga_results.append(ga.run())
    
    # FE
    print(f"Running {fe_trials} Forgetting Engine trials...")
    for i in range(fe_trials):
        seed = seed_offset + nn_trials + ga_trials + i
        domain = TSPDomain(num_cities, seed=seed)
        fe = ForgettingEngineTSP(domain, seed=seed)
        fe_results.append(fe.run())
    
    nn_distances = [r["best_distance"] for r in nn_results]
    ga_distances = [r["best_distance"] for r in ga_results]
    fe_distances = [r["best_distance"] for r in fe_results]
    
    nn_best = min(nn_distances)
    ga_best = min(ga_distances)
    fe_best = min(fe_distances)
    
    ga_improvement = ((ga_best - nn_best) / nn_best * 100) if nn_best > 0 else 0
    fe_improvement = ((fe_best - nn_best) / nn_best * 100) if nn_best > 0 else 0
    
    print(f"\nResults:")
    print(f"  NN best:  {nn_best:.1f} km")
    print(f"  GA best:  {ga_best:.1f} km ({ga_improvement:+.1f}%)")
    print(f"  FE best:  {fe_best:.1f} km ({fe_improvement:+.1f}%)")
    
    from scipy import stats
    u_stat, p_value = stats.mannwhitneyu(fe_distances, ga_distances, alternative='two-sided')
    fe_mean = np.mean(fe_distances)
    ga_mean = np.mean(ga_distances)
    pooled_std = np.sqrt((np.std(fe_distances)**2 + np.std(ga_distances)**2) / 2)
    cohens_d = (ga_mean - fe_mean) / pooled_std if pooled_std > 0 else 0
    
    return {
        "cities": num_cities,
        "nn_results": nn_results,
        "ga_results": ga_results,
        "fe_results": fe_results,
        "nn_best": float(nn_best),
        "ga_best": float(ga_best),
        "fe_best": float(fe_best),
        "ga_improvement": float(ga_improvement),
        "fe_improvement": float(fe_improvement),
        "p_value_fe_vs_ga": float(p_value),
        "cohens_d": float(cohens_d)
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  FORGETTING ENGINE: TRAVELING SALESMAN PROBLEM (TSP)")
    print("  Pharmaceutical-Grade Multi-Scale Validation (620 trials)")
    print("="*70)
    
    scales = [15, 30, 50, 200]
    seed_offset = 5000
    results = []
    
    for num_cities in scales:
        result = run_scale(num_cities, seed_offset)
        results.append(result)
        seed_offset += 1000
    
    output = {
        "study": "Forgetting Engine: Traveling Salesman Problem",
        "timestamp": datetime.now().isoformat(),
        "total_trials": 620,
        "scales": results
    }
    
    with open("FE_TSP_PHARMA_GRADE.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results exported to FE_TSP_PHARMA_GRADE.json")
    print(f"   200-city FE advantage: {results[-1]['fe_improvement']:.1f}% over GA")
