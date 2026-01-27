# Complete Implementation: Forgetting Engine vs Monte Carlo Protein Folding Experiment
import numpy as np
import random
import time
import csv
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

@dataclass
class TrialResult:
    algorithm: str
    sequence: str
    trial_id: int
    success: bool
    steps_to_solution: int
    final_energy: float
    ground_state_energy: float
    runtime_seconds: float
    parameters: dict
    timestamp: datetime
    random_seed: int

# 2D Lattice Protein Model Implementation
class ProteinLattice:
    def __init__(self, sequence: str):
        self.sequence = sequence
        self.length = len(sequence)
        
    def calculate_energy(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate HP model energy: -1 for each H-H contact"""
        energy = 0
        for i in range(self.length):
            for j in range(i + 2, self.length):  # Skip adjacent residues
                if self._are_adjacent(positions[i], positions[j]):
                    if self.sequence[i] == 'H' and self.sequence[j] == 'H':
                        energy -= 1
        return energy
    
    def _are_adjacent(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two positions are adjacent on lattice"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1
    
    def is_valid_conformation(self, positions: List[Tuple[int, int]]) -> bool:
        """Check if conformation is valid (no overlaps, proper connectivity)"""
        if len(positions) != self.length:
            return False
            
        # Check for overlaps
        if len(set(positions)) != len(positions):
            return False
            
        # Check connectivity (adjacent residues must be adjacent on lattice)
        for i in range(len(positions) - 1):
            if not self._are_adjacent(positions[i], positions[i + 1]):
                return False
                
        return True
    
    def generate_random_walk(self) -> List[Tuple[int, int]]:
        """Generate a valid random walk conformation"""
        positions = [(0, 0)]  # Start at origin
        
        for i in range(1, self.length):
            # Try to place next residue
            attempts = 0
            while attempts < 100:  # Prevent infinite loops
                # Choose random direction from current position
                current = positions[-1]
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                direction = random.choice(directions)
                new_pos = (current[0] + direction[0], current[1] + direction[1])
                
                # Check if position is free
                if new_pos not in positions:
                    positions.append(new_pos)
                    break
                attempts += 1
            
            if attempts >= 100:  # Failed to place, restart
                return self.generate_random_walk()
                
        return positions

class MonteCarloSearcher:
    def __init__(self, sequence: str, temperature: float = 1.0, max_steps: int = 10000):
        self.sequence = sequence
        self.temperature = temperature
        self.max_steps = max_steps
        self.lattice = ProteinLattice(sequence)
        self.ground_state_energy = self._calculate_ground_state_energy()
        
    def _calculate_ground_state_energy(self) -> float:
        """Calculate theoretical ground state energy for HP sequence"""
        h_count = self.sequence.count('H')
        # Theoretical minimum: maximum H-H contacts
        if h_count <= 4:
            return -(h_count - 1)
        else:
            # Approximate ground state for longer sequences
            return -min(h_count * (h_count - 1) // 4, (len(self.sequence) - 2) // 2)
    
    def run_simulation(self, random_seed: int) -> Tuple[int, bool, float]:
        """Execute Monte Carlo simulation"""
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize with random conformation
        current_positions = self.lattice.generate_random_walk()
        current_energy = self.lattice.calculate_energy(current_positions)
        best_energy = current_energy
        
        step_count = 0
        
        while step_count < self.max_steps:
            # Generate random move (pivot move)
            new_positions = self._generate_pivot_move(current_positions)
            
            if new_positions and self.lattice.is_valid_conformation(new_positions):
                new_energy = self.lattice.calculate_energy(new_positions)
                
                # Metropolis criterion
                if self._accept_move(current_energy, new_energy):
                    current_positions = new_positions
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        
                    # Check if ground state reached
                    if current_energy <= self.ground_state_energy:
                        return step_count + 1, True, current_energy
                        
            step_count += 1
            
        return self.max_steps, False, best_energy
    
    def _generate_pivot_move(self, positions: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        """Generate a pivot move"""
        if len(positions) < 3:
            return None
            
        # Choose random pivot point (not endpoints)
        pivot_idx = random.randint(1, len(positions) - 2)
        
        # Choose random direction for rotation
        new_positions = positions.copy()
        
        # Simple implementation: try moving one residue
        if pivot_idx < len(positions) - 1:
            current = positions[pivot_idx]
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            direction = random.choice(directions)
            new_pos = (current[0] + direction[0], current[1] + direction[1])
            
            # Check if this creates a valid move
            temp_positions = positions.copy()
            temp_positions[pivot_idx] = new_pos
            
            return temp_positions
            
        return None
    
    def _accept_move(self, current_energy: float, new_energy: float) -> bool:
        """Metropolis acceptance criterion"""
        if new_energy <= current_energy:
            return True
        else:
            delta_e = new_energy - current_energy
            probability = np.exp(-delta_e / self.temperature)
            return random.random() < probability

class ForgettingEngine:
    def __init__(self, sequence: str, population_size: int = 100, forget_rate: float = 0.3):
        self.sequence = sequence
        self.population_size = population_size
        self.forget_rate = forget_rate
        self.lattice = ProteinLattice(sequence)
        self.ground_state_energy = self._calculate_ground_state_energy()
        
    def _calculate_ground_state_energy(self) -> float:
        """Calculate theoretical ground state energy for HP sequence"""
        h_count = self.sequence.count('H')
        if h_count <= 4:
            return -(h_count - 1)
        else:
            return -min(h_count * (h_count - 1) // 4, (len(self.sequence) - 2) // 2)
    
    def run_simulation(self, random_seed: int) -> Tuple[int, bool, float]:
        """Execute Forgetting Engine simulation"""
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            pos = self.lattice.generate_random_walk()
            energy = self.lattice.calculate_energy(pos)
            population.append((pos, energy))
        
        step_count = 0
        max_steps = 10000
        
        while step_count < max_steps:
            # Generate moves for all conformations
            new_candidates = []
            
            for positions, energy in population:
                # Generate multiple moves per conformation
                moves = self._generate_all_moves(positions)
                
                for move in moves:
                    if move and self.lattice.is_valid_conformation(move):
                        move_energy = self.lattice.calculate_energy(move)
                        # FORGETTING: Only keep improvements or equal energy
                        if move_energy <= energy:
                            new_candidates.append((move, move_energy))
            
            if not new_candidates:
                # If stuck, restart with new random population
                population = []
                for _ in range(self.population_size):
                    pos = self.lattice.generate_random_walk()
                    energy = self.lattice.calculate_energy(pos)
                    population.append((pos, energy))
                step_count += 1
                continue
            
            # Sort by energy (best first)
            new_candidates.sort(key=lambda x: x[1])
            
            # Check for ground state
            best_energy = new_candidates[0][1]
            if best_energy <= self.ground_state_energy:
                return step_count + 1, True, best_energy
            
            # FORGET worst performers
            keep_count = max(1, int(len(new_candidates) * (1 - self.forget_rate)))
            population = new_candidates[:keep_count]
            
            # Replenish population by mutating best conformations
            while len(population) < self.population_size:
                # Choose parent from top third
                parent_idx = random.randint(0, min(len(population) // 3, len(population) - 1))
                parent = population[parent_idx][0]
                
                # Mutate parent
                mutated = self._mutate_conformation(parent)
                if mutated and self.lattice.is_valid_conformation(mutated):
                    energy = self.lattice.calculate_energy(mutated)
                    population.append((mutated, energy))
                else:
                    # If mutation failed, add random conformation
                    pos = self.lattice.generate_random_walk()
                    energy = self.lattice.calculate_energy(pos)
                    population.append((pos, energy))
            
            step_count += 1
        
        # Return best energy found
        best_energy = min(pop[1] for pop in population)
        return max_steps, False, best_energy
    
    def _generate_all_moves(self, positions: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """Generate all possible single moves"""
        moves = []
        
        # Try moving each residue (except endpoints if it breaks connectivity)
        for i in range(1, len(positions) - 1):  # Skip endpoints
            current = positions[i]
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
            for direction in directions:
                new_pos = (current[0] + direction[0], current[1] + direction[1])
                
                # Create new conformation
                new_positions = positions.copy()
                new_positions[i] = new_pos
                
                moves.append(new_positions)
        
        return moves
    
    def _mutate_conformation(self, positions: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        """Mutate a conformation by making a small change"""
        if len(positions) < 3:
            return None
            
        # Choose random residue to move (not endpoints)
        idx = random.randint(1, len(positions) - 2)
        current = positions[idx]
        
        # Try random direction
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        direction = random.choice(directions)
        new_pos = (current[0] + direction[0], current[1] + direction[1])
        
        # Create mutated conformation
        mutated = positions.copy()
        mutated[idx] = new_pos
        
        return mutated

# Experiment Runner
class ExperimentRunner:
    def __init__(self):
        self.results = []
        
    def run_experiment(self, sequence: str, num_trials: int = 1000):
        """Run complete experiment comparing both algorithms"""
        print(f"Running experiment on sequence: {sequence}")
        print(f"Trials per algorithm: {num_trials}")
        
        # Test sequence
        test_sequence = sequence
        
        # Run Monte Carlo trials
        print("\nRunning Monte Carlo trials...")
        for trial in range(num_trials):
            if trial % 100 == 0:
                print(f"  Trial {trial}/{num_trials}")
                
            seed = random.randint(1, 1000000)
            mc = MonteCarloSearcher(test_sequence, temperature=1.0)
            
            start_time = time.time()
            steps, success, final_energy = mc.run_simulation(seed)
            runtime = time.time() - start_time
            
            result = TrialResult(
                algorithm="MonteCarlo",
                sequence=test_sequence,
                trial_id=trial,
                success=success,
                steps_to_solution=steps,
                final_energy=final_energy,
                ground_state_energy=mc.ground_state_energy,
                runtime_seconds=runtime,
                parameters={"temperature": 1.0, "max_steps": 10000},
                timestamp=datetime.now(),
                random_seed=seed
            )
            
            self.results.append(result)
        
        # Run Forgetting Engine trials
        print("\nRunning Forgetting Engine trials...")
        for trial in range(num_trials):
            if trial % 100 == 0:
                print(f"  Trial {trial}/{num_trials}")
                
            seed = random.randint(1, 1000000)
            fe = ForgettingEngine(test_sequence, population_size=50, forget_rate=0.3)
            
            start_time = time.time()
            steps, success, final_energy = fe.run_simulation(seed)
            runtime = time.time() - start_time
            
            result = TrialResult(
                algorithm="ForgettingEngine",
                sequence=test_sequence,
                trial_id=trial,
                success=success,
                steps_to_solution=steps,
                final_energy=final_energy,
                ground_state_energy=fe.ground_state_energy,
                runtime_seconds=runtime,
                parameters={"population_size": 50, "forget_rate": 0.3},
                timestamp=datetime.now(),
                random_seed=seed
            )
            
            self.results.append(result)
            
        print(f"\nCompleted {len(self.results)} total trials")
        return self.results
    
    def save_results_csv(self, filename: str = "experiment_results.csv"):
        """Save results to CSV"""
        with open(filename, 'w', newline='') as f:
            if self.results:
                fieldnames = asdict(self.results[0]).keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    row = asdict(result)
                    # Convert parameters dict to string for CSV
                    row['parameters'] = str(row['parameters'])
                    writer.writerow(row)
                    
        print(f"Results saved to {filename}")
    
    def analyze_results(self):
        """Perform statistical analysis"""
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
        
        # Success rates
        success_rates = df.groupby('algorithm')['success'].agg(['count', 'sum', 'mean'])
        print("\nSuccess Rates:")
        print(success_rates)
        
        # Steps to solution (for successful trials only)
        successful_df = df[df['success'] == True]
        if len(successful_df) > 0:
            print("\nSteps to Solution (Successful Trials):")
            steps_stats = successful_df.groupby('algorithm')['steps_to_solution'].agg(['count', 'mean', 'std', 'median'])
            print(steps_stats)
            
            # Mann-Whitney U test
            mc_steps = successful_df[successful_df['algorithm'] == 'MonteCarlo']['steps_to_solution']
            fe_steps = successful_df[successful_df['algorithm'] == 'ForgettingEngine']['steps_to_solution']
            
            if len(mc_steps) > 0 and len(fe_steps) > 0:
                statistic, p_value = stats.mannwhitneyu(mc_steps, fe_steps, alternative='two-sided')
                print(f"\nMann-Whitney U Test:")
                print(f"  Statistic: {statistic}")
                print(f"  P-value: {p_value:.6f}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
                
                # Effect size (speed improvement)
                if len(fe_steps) > 0 and len(mc_steps) > 0:
                    speed_improvement = mc_steps.mean() / fe_steps.mean()
                    print(f"  Speed Improvement: {speed_improvement:.2f}x")
        
        # Runtime analysis
        print("\nRuntime Analysis:")
        runtime_stats = df.groupby('algorithm')['runtime_seconds'].agg(['mean', 'std'])
        print(runtime_stats)
        
        return df

# Main execution
if __name__ == "__main__":
    print("=== FORGETTING ENGINE vs MONTE CARLO EXPERIMENT ===")
    print("Initializing experiment...")
    
    runner = ExperimentRunner()
    test_sequence = "HPHPHPHHPHHHPHPPPHPH"
    print(f"Test sequence: {test_sequence}")
    print(f"Length: {len(test_sequence)} residues")
    print(f"H residues: {test_sequence.count('H')}")
    print(f"P residues: {test_sequence.count('P')}")
    
    # Run the experiment
    results = runner.run_experiment(test_sequence, num_trials=1000)
    runner.save_results_csv()
    
    # Perform analysis
    analysis_df = runner.analyze_results()
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Total trials: {len(results)}")
    print("Results saved to: experiment_results.csv")
    print("Analysis completed.")