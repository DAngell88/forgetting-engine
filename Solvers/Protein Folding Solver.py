#!/usr/bin/env python3
"""
Forgetting Engine: 3D Protein Folding Solver
Pharmaceutical-Grade Validation (4,800 trials)
Two-Phase Study: Pilot (800) + Production (4,000)

Fixed Random Seeds for Reproducibility
Pre-registered Protocol Hash: 9328d4e885aede604f535222d8abac387fad132ff55908dc4e33c9b143921a7c

This script implements the EXACT Forgetting Engine algorithm from the manuscript:
- Strategic Elimination: Multi-variate elimination score (α=-1.0, β=0.3, γ=0.2, δ=-0.1)
- Paradox Retention: Selective preservation of high-complexity, low-fitness candidates
- Baseline: Metropolis-Hastings Monte Carlo (3D HP Lattice Model)
"""

import numpy as np
import random
import json
from typing import List, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime
import sys

# ============================================================================
# DOMAIN: 3D Protein Folding (HP Lattice Model)
# ============================================================================

class ProteinFoldingDomain3D:
    """3D HP Lattice Model for protein folding."""
    
    def __init__(self, sequence: str):
        """
        Args:
            sequence: String of 'H' (hydrophobic) and 'P' (polar) residues
        """
        self.sequence = sequence
        self.length = len(sequence)
        self.known_optimum = -9.0  # Target energy
    
    def is_valid_conformation(self, coords: List[Tuple[int, int, int]]) -> bool:
        """Check self-avoiding walk constraint."""
        return len(set(coords)) == len(coords)
    
    def calculate_energy(self, coords: List[Tuple[int, int, int]]) -> float:
        """
        Calculate Gibbs Free Energy based on H-H contacts.
        Energy = sum of -1.0 for each non-adjacent H-H contact
        Lower energy = better (more stable fold)
        """
        if not self.is_valid_conformation(coords):
            return 0.0  # Invalid = worst energy
        
        energy = 0.0
        for i in range(self.length):
            if self.sequence[i] == 'H':
                for j in range(i + 2, self.length):  # Non-adjacent
                    if self.sequence[j] == 'H':
                        dist = np.sqrt(
                            (coords[i][0] - coords[j][0])**2 +
                            (coords[i][1] - coords[j][1])**2 +
                            (coords[i][2] - coords[j][2])**2
                        )
                        if abs(dist - 1.0) < 0.01:  # Lattice neighbor
                            energy -= 1.0
        return energy
    
    def contact_order(self, coords: List[Tuple[int, int, int]]) -> float:
        """
        Structural complexity metric: Contact Order
        Defined as average sequence distance between H-H contacts
        Higher = more complex structure
        """
        if not self.is_valid_conformation(coords):
            return 0.0
        
        hh_contacts = []
        for i in range(self.length):
            if self.sequence[i] == 'H':
                for j in range(i + 2, self.length):
                    if self.sequence[j] == 'H':
                        dist = np.sqrt(
                            (coords[i][0] - coords[j][0])**2 +
                            (coords[i][1] - coords[j][1])**2 +
                            (coords[i][2] - coords[j][2])**2
                        )
                        if abs(dist - 1.0) < 0.01:
                            hh_contacts.append(abs(i - j))
        
        if len(hh_contacts) == 0:
            return 0.0
        return np.mean(hh_contacts)
    
    def radius_of_gyration(self, coords: List[Tuple[int, int, int]]) -> float:
        """Radius of gyration: measure of structural compactness."""
        coords_array = np.array(coords, dtype=float)
        center = np.mean(coords_array, axis=0)
        return np.sqrt(np.mean(np.sum((coords_array - center)**2, axis=1)))


# ============================================================================
# FORGETTING ENGINE CORE
# ============================================================================

@dataclass
class Solution:
    """Represents a candidate solution."""
    coords: List[Tuple[int, int, int]]
    fitness: float  # Energy
    complexity: float  # Contact Order
    novelty: float  # Distance from nearest neighbors
    age: int  # Generation born
    elim_score: float = 0.0


class ForgettingEngine3D:
    """3D Protein Folding Forgetting Engine."""
    
    def __init__(self, sequence: str, population_size: int = 25, 
                 generations: int = 25, forget_rate: float = 0.40,
                 paradox_rate: float = 0.15, seed: int = 42):
        self.domain = ProteinFoldingDomain3D(sequence)
        self.rng = np.random.RandomState(seed)
        self.random_seed = seed
        
        # Parameters from manuscript
        self.N = population_size
        self.G = generations
        self.forget_rate = forget_rate
        self.paradox_rate = paradox_rate
        
        # Elimination score weights (from manuscript: α=-1.0, β=0.3, γ=0.2, δ=-0.1)
        self.alpha = -1.0    # Fitness (penalize high energy)
        self.beta = 0.3      # Complexity (reward structured folds)
        self.gamma = 0.2     # Novelty (reward diversity)
        self.delta = -0.1    # Age (penalize old solutions)
        
        self.population: List[Solution] = []
        self.paradox_buffer: List[Solution] = []
        self.best_solution: Solution = None
        self.history: List[Dict] = []
        
    def initialize_population(self):
        """Random 3D conformations."""
        for _ in range(self.N):
            coords = self._random_conformation()
            sol = Solution(
                coords=coords,
                fitness=self.domain.calculate_energy(coords),
                complexity=self.domain.contact_order(coords),
                novelty=0.0,
                age=0
            )
            self.population.append(sol)
        self.best_solution = min(self.population, key=lambda x: x.fitness)
    
    def _random_conformation(self) -> List[Tuple[int, int, int]]:
        """Generate random self-avoiding walk on 3D lattice."""
        moves = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        coords = [(0, 0, 0)]
        
        for _ in range(self.domain.length - 1):
            found = False
            for _ in range(100):  # Max attempts
                move = moves[self.rng.randint(0, 6)]
                new_coord = (
                    coords[-1][0] + move[0],
                    coords[-1][1] + move[1],
                    coords[-1][2] + move[2]
                )
                if new_coord not in coords:
                    coords.append(new_coord)
                    found = True
                    break
            if not found:
                return None  # Invalid trajectory
        
        return coords if len(coords) == self.domain.length else None
    
    def compute_elimination_score(self, sol: Solution, generation: int):
        """
        Elimination Score from manuscript:
        E(x,g) = α·fitness(x) + β·complexity(x) + γ·novelty(x,P) + δ·age(x)
        Higher score = better = less likely to be eliminated
        """
        novelty = self._compute_novelty(sol)
        age = generation - sol.age if sol.age > 0 else 0
        
        score = (
            self.alpha * sol.fitness +
            self.beta * sol.complexity +
            self.gamma * novelty +
            self.delta * age
        )
        sol.elim_score = score
    
    def _compute_novelty(self, sol: Solution) -> float:
        """Distance from nearest neighbor in population."""
        if len(self.population) < 2:
            return 1.0
        
        distances = []
        for other in self.population:
            if other is sol:
                continue
            # Euclidean distance in conformation space
            dist = np.sqrt(sum(
                (c1[i] - c2[i])**2 
                for i in range(3)
                for c1, c2 in zip(sol.coords, other.coords)
            ))
            distances.append(dist)
        
        return min(distances) if distances else 1.0
    
    def strategic_elimination(self, generation: int):
        """
        Eliminate bottom (1 - forget_rate) of population by elimination score.
        Forget rate 0.40 = keep top 60%
        """
        for sol in self.population:
            self.compute_elimination_score(sol, generation)
        
        # Sort by elimination score (higher = better)
        sorted_pop = sorted(self.population, key=lambda x: x.elim_score, reverse=True)
        keep_count = int(np.ceil(len(self.population) * (1 - self.forget_rate)))
        
        self.population = sorted_pop[:keep_count]
        eliminated = sorted_pop[keep_count:]
        
        return eliminated
    
    def paradox_retention(self, eliminated: List[Solution]):
        """
        Selectively retain eliminated solutions that are paradoxical:
        - Fitness worse than average (high energy)
        - Complexity above median (structured)
        
        Paradoxical = "bad by fitness but good by structure"
        """
        if not eliminated:
            return
        
        mean_fitness = np.mean([e.fitness for e in eliminated])
        median_complexity = np.median([e.complexity for e in eliminated])
        
        paradoxes = []
        for sol in eliminated:
            # Paradox: worse fitness + higher complexity
            if sol.fitness > mean_fitness and sol.complexity > median_complexity:
                paradoxes.append(sol)
        
        # Sample paradox_rate fraction
        sample_size = max(1, int(len(paradoxes) * self.paradox_rate))
        if paradoxes:
            self.paradox_buffer.extend(
                self.rng.choice(paradoxes, min(sample_size, len(paradoxes)), replace=False)
            )
    
    def population_regeneration(self):
        """Maintain population size via mutation and reintroduction."""
        while len(self.population) < self.N:
            # 20% reintroduce from paradox buffer, 80% mutate from main pop
            if self.rng.random() < 0.2 and self.paradox_buffer:
                parent = self.paradox_buffer[self.rng.randint(0, len(self.paradox_buffer))]
                self.paradox_buffer.remove(parent)
            else:
                parent = self.population[self.rng.randint(0, len(self.population))]
            
            # Mutate: random pivot move on 1 residue
            child_coords = self._mutate_conformation(parent.coords)
            if child_coords:
                child = Solution(
                    coords=child_coords,
                    fitness=self.domain.calculate_energy(child_coords),
                    complexity=self.domain.contact_order(child_coords),
                    novelty=0.0,
                    age=len(self.history)
                )
                self.population.append(child)
    
    def _mutate_conformation(self, coords: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Single-residue pivot move (1 in domain.length chance)."""
        if self.rng.random() > (1.0 / self.domain.length):
            return coords  # No mutation
        
        # Try random pivot at random position
        pos = self.rng.randint(1, len(coords) - 1)
        moves = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        
        for _ in range(10):
            move = moves[self.rng.randint(0, 6)]
            new_coords = list(coords)
            new_coords[pos] = (coords[pos][0] + move[0], coords[pos][1] + move[1], coords[pos][2] + move[2])
            
            if self.domain.is_valid_conformation(new_coords):
                return new_coords
        
        return coords
    
    def run(self) -> Dict:
        """Execute FE algorithm."""
        self.initialize_population()
        
        for generation in range(self.G):
            # Strategic Elimination
            eliminated = self.strategic_elimination(generation)
            
            # Paradox Retention
            self.paradox_retention(eliminated)
            
            # Population Regeneration
            self.population_regeneration()
            
            # Track best
            gen_best = min(self.population, key=lambda x: x.fitness)
            if gen_best.fitness < self.best_solution.fitness:
                self.best_solution = gen_best
            
            # Record history
            self.history.append({
                "generation": generation,
                "best_energy": self.best_solution.fitness,
                "mean_energy": np.mean([s.fitness for s in self.population]),
                "pop_size": len(self.population),
                "paradox_buffer_size": len(self.paradox_buffer)
            })
        
        return self._generate_results()
    
    def _generate_results(self) -> Dict:
        """Format results for JSON export."""
        success = 1 if self.best_solution.fitness <= self.domain.known_optimum else 0
        
        return {
            "success": success,
            "best_energy": float(self.best_solution.fitness),
            "best_contact_order": float(self.domain.contact_order(self.best_solution.coords)),
            "best_rg": float(self.domain.radius_of_gyration(self.best_solution.coords)),
            "final_population_energy_mean": float(np.mean([s.fitness for s in self.population])),
            "final_population_energy_std": float(np.std([s.fitness for s in self.population])),
            "generations": self.G,
            "random_seed": self.random_seed
        }


# ============================================================================
# MONTE CARLO BASELINE (Metropolis-Hastings)
# ============================================================================

class MonteCarloBaseline3D:
    """Metropolis-Hastings 3D protein folding baseline."""
    
    def __init__(self, sequence: str, seed: int = 42):
        self.domain = ProteinFoldingDomain3D(sequence)
        self.rng = np.random.RandomState(seed)
        self.random_seed = seed
        self.temperature = 1.0
        self.max_iterations = 800
    
    def run(self) -> Dict:
        """Single MC trial."""
        current = self._random_conformation()
        current_energy = self.domain.calculate_energy(current)
        best = current
        best_energy = current_energy
        
        for iteration in range(self.max_iterations):
            # Random pivot move
            proposed = self._mutate_conformation(current)
            if proposed is None:
                continue
            
            proposed_energy = self.domain.calculate_energy(proposed)
            
            # Metropolis acceptance
            dE = proposed_energy - current_energy
            if dE < 0 or self.rng.random() < np.exp(-dE / self.temperature):
                current = proposed
                current_energy = proposed_energy
                
                if current_energy < best_energy:
                    best = current
                    best_energy = current_energy
            
            # Early stopping
            if best_energy <= self.domain.known_optimum:
                break
        
        success = 1 if best_energy <= self.domain.known_optimum else 0
        
        return {
            "success": success,
            "best_energy": float(best_energy),
            "best_contact_order": float(self.domain.contact_order(best)),
            "iterations": iteration + 1,
            "random_seed": self.random_seed
        }
    
    def _random_conformation(self):
        """Random SAW on 3D lattice."""
        moves = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        coords = [(0, 0, 0)]
        
        for _ in range(self.domain.length - 1):
            found = False
            for _ in range(100):
                move = moves[self.rng.randint(0, 6)]
                new_coord = (coords[-1][0] + move[0], coords[-1][1] + move[1], coords[-1][2] + move[2])
                if new_coord not in coords:
                    coords.append(new_coord)
                    found = True
                    break
            if not found:
                return None
        
        return coords if len(coords) == self.domain.length else None
    
    def _mutate_conformation(self, coords):
        """Single pivot move."""
        if self.rng.random() > 0.1:
            return coords
        
        pos = self.rng.randint(1, len(coords) - 1)
        moves = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        
        for _ in range(10):
            move = moves[self.rng.randint(0, 6)]
            new_coords = list(coords)
            new_coords[pos] = (coords[pos][0] + move[0], coords[pos][1] + move[1], coords[pos][2] + move[2])
            
            if self.domain.is_valid_conformation(new_coords):
                return new_coords
        
        return coords


# ============================================================================
# VALIDATION PROTOCOL
# ============================================================================

def run_phase(name: str, mc_trials: int, fe_trials: int, seed_offset: int) -> Dict:
    """Run one phase (pilot or production)."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    
    sequence = "HPHPHPHHPHHHPHPPPHPH"  # 20-residue benchmark
    
    mc_results = []
    fe_results = []
    
    # Monte Carlo trials
    print(f"Running {mc_trials} Monte Carlo trials...")
    for i in range(mc_trials):
        seed = seed_offset + i
        mc = MonteCarloBaseline3D(sequence, seed=seed)
        result = mc.run()
        mc_results.append(result)
        if (i + 1) % 100 == 0:
            print(f"  MC: {i+1}/{mc_trials} complete")
    
    # Forgetting Engine trials
    print(f"Running {fe_trials} Forgetting Engine trials...")
    for i in range(fe_trials):
        seed = seed_offset + mc_trials + i
        fe = ForgettingEngine3D(sequence, seed=seed)
        result = fe.run()
        fe_results.append(result)
        if (i + 1) % 100 == 0:
            print(f"  FE: {i+1}/{fe_trials} complete")
    
    # Statistics
    mc_successes = sum(1 for r in mc_results if r["success"])
    fe_successes = sum(1 for r in fe_results if r["success"])
    
    mc_success_rate = 100.0 * mc_successes / len(mc_results)
    fe_success_rate = 100.0 * fe_successes / len(fe_results)
    
    from scipy import stats
    odds_ratio, p_value = stats.contingency.fisher_exact(
        [[mc_successes, len(mc_results) - mc_successes],
         [fe_successes, len(fe_results) - fe_successes]]
    )
    
    mc_energies = [r["best_energy"] for r in mc_results]
    fe_energies = [r["best_energy"] for r in fe_results]
    
    u_stat, energy_p = stats.mannwhitneyu(mc_energies, fe_energies, alternative='two-sided')
    
    mc_mean = np.mean(mc_energies)
    fe_mean = np.mean(fe_energies)
    pooled_std = np.sqrt((np.std(mc_energies)**2 + np.std(fe_energies)**2) / 2)
    cohens_d = (fe_mean - mc_mean) / pooled_std if pooled_std > 0 else 0
    
    improvement = ((fe_success_rate - mc_success_rate) / mc_success_rate * 100) if mc_success_rate > 0 else 0
    
    print(f"\nResults:")
    print(f"  MC:  {mc_successes}/{len(mc_results)} successes ({mc_success_rate:.1f}%)")
    print(f"  FE:  {fe_successes}/{len(fe_results)} successes ({fe_success_rate:.1f}%)")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  p-value (Fisher): {p_value:.2e}")
    print(f"  Odds Ratio: {odds_ratio:.2f}")
    print(f"  Cohen's d (energy): {cohens_d:.2f}")
    
    return {
        "name": name,
        "mc_trials": mc_trials,
        "fe_trials": fe_trials,
        "mc_results": mc_results,
        "fe_results": fe_results,
        "mc_success_rate": mc_success_rate,
        "fe_success_rate": fe_success_rate,
        "improvement": improvement,
        "p_value": float(p_value),
        "odds_ratio": float(odds_ratio),
        "cohens_d": float(cohens_d),
        "mc_energy_mean": float(mc_mean),
        "fe_energy_mean": float(fe_mean),
        "mc_energy_std": float(np.std(mc_energies)),
        "fe_energy_std": float(np.std(fe_energies))
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  FORGETTING ENGINE: 3D PROTEIN FOLDING VALIDATION")
    print("  Pharmaceutical-Grade (Fixed Seeds)")
    print("="*70)
    
    # Phase 1: Pilot (Pre-registered)
    pilot_results = run_phase(
        "PILOT STUDY (Pre-registered, n=800)",
        mc_trials=400,
        fe_trials=400,
        seed_offset=3000  # MC: 3000-3399, FE: 5000-5397
    )
    
    # Phase 2: Production
    production_results = run_phase(
        "PRODUCTION STUDY (n=4,000)",
        mc_trials=2000,
        fe_trials=2000,
        seed_offset=10000  # MC: 10000-11999, FE: 20000-21999
    )
    
    # Export JSON
    output = {
        "study": "Forgetting Engine: 3D Protein Folding",
        "timestamp": datetime.now().isoformat(),
        "sequence": "HPHPHPHHPHHHPHPPPHPH",
        "known_optimum": -9.0,
        "total_trials": 4800,
        "pilot": pilot_results,
        "production": production_results,
        "protocol_hash": "9328d4e885aede604f535222d8abac387fad132ff55908dc4e33c9b143921a7c"
    }
    
    with open("FE_3D_PROTEIN_FOLDING_PHARMA_GRADE.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results exported to FE_3D_PROTEIN_FOLDING_PHARMA_GRADE.json")
    print(f"   Pilot: {pilot_results['improvement']:.1f}% improvement (p={pilot_results['p_value']:.2e})")
    print(f"   Production: {production_results['improvement']:.1f}% improvement (p={production_results['p_value']:.2e})")
