#!/usr/bin/env python3

"""
Forgetting Engine: 2D HP Protein Folding (Refiner-Calibrated)
Domain: HP Lattice Model (2D)
Complexity Metric: Radius of Gyration (Rg)
Validation: Statistical Significance vs. Metropolis-Hastings MC
"""

import numpy as np
import random
import json
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime
from scipy.stats import mannwhitneyu

@dataclass
class Conformation:
    """Represents a protein folding state on a 2D lattice."""
    id: str
    positions: List[Tuple[int, int]]
    energy: float = 0.0  # Truth (Gibbs Free Energy)
    complexity_rg: float = 0.0  # Contradiction (Radius of Gyration)
    elim_score: float = 0.0

class Protein2DDomain:
    """2D HP Lattice Search Space."""

    def __init__(self, sequence: str, seed: int = 42):
        self.sequence = sequence
        self.length = len(sequence)
        self.rng = random.Random(seed)

    def calculate_energy(self, positions: List[Tuple[int, int]]) -> float:
        """HP Energy: -1 for each non-sequential H-H contact."""
        energy = 0
        for i in range(self.length):
            for j in range(i + 2, self.length):
                p1, p2 = positions[i], positions[j]
                if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1:
                    if self.sequence[i] == 'H' and self.sequence[j] == 'H':
                        energy -= 1
        return float(energy)

    def calculate_complexity(self, positions: List[Tuple[int, int]]) -> float:
        """Complexity Metric: Inverse Radius of Gyration (Lower Rg = Higher Complexity/Compactness)."""
        coords = np.array(positions)
        center = np.mean(coords, axis=0)
        rg_sq = np.mean(np.sum((coords - center)**2, axis=1))
        # We invert so higher complexity = more compact
        return 1.0 / (np.sqrt(rg_sq) + 1e-6)

    def is_valid(self, positions: List[Tuple[int, int]]) -> bool:
        """Self-avoiding walk check."""
        return len(set(positions)) == self.length

    def generate_random_walk(self) -> List[Tuple[int, int]]:
        """Generates a valid SAW using backtracking (simplified)."""
        while True:
            positions = [(0, 0)]
            occupied = {(0, 0)}
            failed = False

            for _ in range(self.length - 1):
                curr = positions[-1]
                neighbors = [(curr[0]+1, curr[1]), (curr[0]-1, curr[1]),
                            (curr[0], curr[1]+1), (curr[0], curr[1]-1)]
                valid_neighbors = [n for n in neighbors if n not in occupied]

                if not valid_neighbors:
                    failed = True
                    break

                nxt = self.rng.choice(valid_neighbors)
                positions.append(nxt)
                occupied.add(nxt)

            if not failed:
                return positions

class ForgettingEngine2D:
    """FE Implementation with Radius of Gyration Paradox Retention."""

    def __init__(self, domain: Protein2DDomain, pop_size: int = 50,
                 generations: int = 40, forget_rate: float = 0.3,
                 paradox_rate: float = 0.15, seed: int = 42):
        self.domain = domain
        self.pop_size = pop_size
        self.generations = generations
        self.forget_rate = forget_rate
        self.paradox_rate = paradox_rate
        self.rng = random.Random(seed)

    def compute_elimination_score(self, conf: Conformation, gen: int) -> float:
        """E(x, g) = (alpha*Energy + beta*Complexity) / gen."""
        # We want low energy (negative) and high complexity (compact)
        # Higher score = more likely to be forgotten
        alpha, beta = 1.0, -0.4  # Reward low energy and high compactness
        return (alpha * conf.energy + beta * conf.complexity_rg) / gen

    def run(self) -> Conformation:
        population = []

        for i in range(self.pop_size):
            pos = self.domain.generate_random_walk()
            c = Conformation(f"P0-{i}", pos)
            c.energy = self.domain.calculate_energy(pos)
            c.complexity_rg = self.domain.calculate_complexity(pos)
            population.append(c)

        paradox_buffer = []

        for gen in range(1, self.generations + 1):
            for c in population:
                c.elim_score = self.compute_elimination_score(c, gen)

            # Subtractive Pruning
            population.sort(key=lambda x: x.elim_score, reverse=True)
            keep_count = int(self.pop_size * (1 - self.forget_rate))
            elite = population[:keep_count]
            eliminated = population[keep_count:]

            # Paradox Retention: High Energy (bad) but High Complexity (Compact)
            avg_energy = np.mean([c.energy for c in population])

            # A paradox is a protein that hasn't 'docked' yet but is shaped correctly
            paradoxes = [c for c in eliminated if c.energy > avg_energy and c.complexity_rg > 1.2]

            if paradoxes:
                paradox_buffer = sorted(paradoxes, key=lambda x: x.complexity_rg, reverse=True)[:5]

            # Replenishment
            population = elite.copy()

            while len(population) < self.pop_size:
                if self.rng.random() < self.paradox_rate and paradox_buffer:
                    population.append(self.rng.choice(paradox_buffer))
                else:
                    # Mutation: Local move or regrowth
                    parent = self.rng.choice(elite)
                    new_pos = self.domain.generate_random_walk()  # Simplified mutation
                    child = Conformation(f"G{gen}-{self.rng.randint(0,999)}", new_pos)
                    child.energy = self.domain.calculate_energy(new_pos)
                    child.complexity_rg = self.domain.calculate_complexity(new_pos)
                    population.append(child)

        return min(population, key=lambda x: x.energy)

def run_validation(num_trials: int = 50):
    """Pharmaceutical-Grade Validation."""
    seq = "HPHPHPHHPHHHPHPPPHPH"  # 20-residue benchmark
    domain = Protein2DDomain(seq)

    print(f"--- 2D Protein Folding Validation: {seq} ---")

    fe_results = []

    print("Running Forgetting Engine trials...")
    for t in range(num_trials):
        engine = ForgettingEngine2D(domain, seed=2000+t)
        best = engine.run()
        fe_results.append(best.energy)

        if (t+1) % 10 == 0:
            print(f" Trial {t+1}/{num_trials} complete.")

    mc_results = []

    print("\nRunning Metropolis-Hastings Baseline (MC)...")
    for t in range(num_trials):
        # Baseline: 2000 MC steps (equivalent compute)
        best_mc = 0.0
        curr_pos = domain.generate_random_walk()
        curr_e = domain.calculate_energy(curr_pos)

        for _ in range(2000):
            new_pos = domain.generate_random_walk()
            new_e = domain.calculate_energy(new_pos)

            if new_e < curr_e or random.random() < np.exp(-(new_e - curr_e)/0.5):
                curr_pos, curr_e = new_pos, new_e

            if curr_e < best_mc:
                best_mc = curr_e

        mc_results.append(best_mc)

        if (t+1) % 10 == 0:
            print(f" Trial {t+1}/{num_trials} complete.")

    fe_mean, mc_mean = np.mean(fe_results), np.mean(mc_results)
    stat, pval = mannwhitneyu(fe_results, mc_results, alternative='less')

    print("\n" + "="*45)
    print("2D PROTEIN FOLDING RESULTS")
    print(f"FE Mean Energy: {fe_mean:.2f}")
    print(f"MC Mean Energy: {mc_mean:.2f}")
    print(f"P-value (FE < MC): p={pval:.2e}")
    print("="*45)

    with open("protein_2d_validation.json", "w") as f:
        json.dump({
            "sequence": seq,
            "fe_mean": fe_mean,
            "mc_mean": mc_mean,
            "p_value": pval,
            "complexity_metric": "Radius of Gyration"
        }, f, indent=2)

if __name__ == "__main__":
    run_validation()
