
# COMPLETE PYTHON CODE FOR 200-CITY TSP EXPERIMENT REPLICATION
# This code implements and tests Forgetting Engine vs Genetic Algorithm

import numpy as np
import pandas as pd
import time
import random
from typing import List, Tuple
from dataclasses import dataclass

# Configuration
@dataclass
class ExperimentConfig:
    cities: int = 200
    trials_per_algorithm: int = 10
    fe_population: int = 50
    fe_forget_rate: float = 0.3
    ga_population: int = 50
    ga_mutation_rate: float = 0.02
    max_generations: int = 100

# Utility functions
def generate_cities(n_cities: int) -> List[Tuple[float, float]]:
    cities = []
    for i in range(n_cities):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        cities.append((x, y))
    return cities

def distance(city1: Tuple[float, float], city2: Tuple[float, float]) -> float:
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def tour_length(tour: List[int], cities: List[Tuple[float, float]]) -> float:
    total = 0
    for i in range(len(tour)):
        total += distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]])
    return total

# Genetic Algorithm Implementation
class GeneticAlgorithmTSP:
    def __init__(self, cities: List[Tuple[float, float]], 
                 population_size: int = 50, mutation_rate: float = 0.02):
        self.cities = cities
        self.n_cities = len(cities)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []

    def generate_random_tour(self) -> List[int]:
        tour = list(range(self.n_cities))
        random.shuffle(tour)
        return tour

    def evaluate_fitness(self, tour: List[int]) -> float:
        return tour_length(tour, self.cities)

    def tournament_selection(self, population: List[List[int]], 
                           fitness_scores: List[float], tournament_size: int = 5) -> List[int]:
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx]

    def order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        size = len(parent1)
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size)

        child = [-1] * size
        child[start:end] = parent1[start:end]

        parent2_filtered = [city for city in parent2 if city not in child]

        fill_idx = 0
        for i in range(start):
            child[i] = parent2_filtered[fill_idx]
            fill_idx += 1

        for i in range(end, size):
            child[i] = parent2_filtered[fill_idx]
            fill_idx += 1

        return child

    def mutate(self, tour: List[int]) -> List[int]:
        if random.random() < self.mutation_rate:
            new_tour = tour.copy()
            i, j = random.sample(range(len(tour)), 2)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            return new_tour
        return tour.copy()

    def solve(self, max_generations: int = 100) -> Tuple[List[int], float, int]:
        self.population = [self.generate_random_tour() for _ in range(self.population_size)]

        best_tour = None
        best_fitness = float('inf')
        generations_run = 0

        for generation in range(max_generations):
            generations_run = generation + 1

            fitness_scores = [self.evaluate_fitness(tour) for tour in self.population]

            gen_best_idx = np.argmin(fitness_scores)
            if fitness_scores[gen_best_idx] < best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_tour = self.population[gen_best_idx].copy()

            new_population = []

            # Elitism: keep best 5
            sorted_indices = np.argsort(fitness_scores)
            for i in range(5):
                new_population.append(self.population[sorted_indices[i]].copy())

            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(self.population, fitness_scores)
                parent2 = self.tournament_selection(self.population, fitness_scores)

                child = self.order_crossover(parent1, parent2)
                child = self.mutate(child)

                new_population.append(child)

            self.population = new_population

        return best_tour, best_fitness, generations_run

# Forgetting Engine Implementation  
class ForgettingEngineTSP:
    def __init__(self, cities: List[Tuple[float, float]], 
                 population_size: int = 50, forget_rate: float = 0.3,
                 paradox_retention: bool = True):
        self.cities = cities
        self.n_cities = len(cities)
        self.population_size = population_size
        self.forget_rate = forget_rate
        self.paradox_retention = paradox_retention
        self.population = []
        self.paradox_buffer = []

    def generate_random_tour(self) -> List[int]:
        tour = list(range(self.n_cities))
        random.shuffle(tour)
        return tour

    def generate_nearest_neighbor_tour(self, start: int = None) -> List[int]:
        if start is None:
            start = random.randint(0, self.n_cities - 1)

        unvisited = set(range(self.n_cities))
        tour = [start]
        unvisited.remove(start)
        current = start

        while unvisited:
            nearest = min(unvisited, key=lambda city: distance(self.cities[current], self.cities[city]))
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        return tour

    def evaluate_fitness(self, tour: List[int]) -> float:
        return tour_length(tour, self.cities)

    def is_paradoxical(self, tour: List[int], population_fitness: List[float]) -> bool:
        tour_fitness = self.evaluate_fitness(tour)
        avg_fitness = np.mean(population_fitness)
        std_fitness = np.std(population_fitness)

        threshold = avg_fitness + 0.5 * std_fitness

        if tour_fitness > threshold:
            return random.random() < 0.3
        return False

    def order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        size = len(parent1)
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size)

        child = [-1] * size
        child[start:end] = parent1[start:end]

        parent2_filtered = [city for city in parent2 if city not in child]

        fill_idx = 0
        for i in range(start):
            child[i] = parent2_filtered[fill_idx]
            fill_idx += 1

        for i in range(end, size):
            child[i] = parent2_filtered[fill_idx]
            fill_idx += 1

        return child

    def mutate_tour(self, tour: List[int]) -> List[int]:
        mutation_type = random.choice(['swap', 'insert', 'reverse'])
        new_tour = tour.copy()

        if mutation_type == 'swap':
            i, j = random.sample(range(len(tour)), 2)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        elif mutation_type == 'insert':
            i = random.randint(0, len(tour) - 1)
            j = random.randint(0, len(tour) - 1)
            city = new_tour.pop(i)
            new_tour.insert(j, city)
        elif mutation_type == 'reverse':
            i, j = sorted(random.sample(range(len(tour)), 2))
            new_tour[i:j+1] = new_tour[i:j+1][::-1]

        return new_tour

    def tournament_select(self, population: List[List[int]], fitness_scores: List[float], 
                         tournament_size: int = 3) -> List[int]:
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx]

    def solve(self, max_generations: int = 100) -> Tuple[List[int], float, int]:
        # Mixed initialization
        self.population = []

        # 30% nearest neighbor seeded
        for _ in range(int(self.population_size * 0.3)):
            self.population.append(self.generate_nearest_neighbor_tour())

        # 70% random
        for _ in range(self.population_size - len(self.population)):
            self.population.append(self.generate_random_tour())

        best_tour = None
        best_fitness = float('inf')
        stagnation_counter = 0
        generations_run = 0

        for generation in range(max_generations):
            generations_run = generation + 1

            fitness_scores = [self.evaluate_fitness(tour) for tour in self.population]

            gen_best_idx = np.argmin(fitness_scores)
            if fitness_scores[gen_best_idx] < best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_tour = self.population[gen_best_idx].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Forgetting mechanism
            sorted_indices = np.argsort(fitness_scores)
            forget_count = int(self.population_size * self.forget_rate)

            # Paradox retention
            if self.paradox_retention and forget_count > 0:
                candidates_for_forgetting = sorted_indices[-forget_count:]
                for idx in candidates_for_forgetting:
                    if self.is_paradoxical(self.population[idx], fitness_scores):
                        if len(self.paradox_buffer) < 10:
                            self.paradox_buffer.append(self.population[idx].copy())

            survivors = [self.population[sorted_indices[i]] for i in range(self.population_size - forget_count)]

            new_solutions = []
            while len(new_solutions) < forget_count:
                if random.random() < 0.6:
                    parent1 = self.tournament_select(survivors, fitness_scores[:len(survivors)])
                    parent2 = self.tournament_select(survivors, fitness_scores[:len(survivors)])
                    child = self.order_crossover(parent1, parent2)
                else:
                    parent = self.tournament_select(survivors, fitness_scores[:len(survivors)])
                    child = self.mutate_tour(parent)

                new_solutions.append(child)

            # Reintroduce paradoxical solutions
            if self.paradox_buffer and random.random() < 0.1:
                paradox_solution = random.choice(self.paradox_buffer)
                new_solutions[-1] = paradox_solution

            self.population = survivors + new_solutions

            # Early stopping
            if stagnation_counter > 30:
                break

        return best_tour, best_fitness, generations_run

# Main experimental execution function
def run_experiment():
    np.random.seed(2025)
    random.seed(2025)

    config = ExperimentConfig()
    cities = generate_cities(config.cities)

    results = {'Algorithm': [], 'Trial': [], 'Best_Tour_Length': [], 
               'Computation_Time': [], 'Generations_Completed': [], 'Cities': []}

    # GA trials
    for trial in range(config.trials_per_algorithm):
        start_time = time.time()
        ga_solver = GeneticAlgorithmTSP(cities, config.ga_population, config.ga_mutation_rate)
        best_tour, best_length, generations = ga_solver.solve(config.max_generations)
        computation_time = time.time() - start_time

        results['Algorithm'].append('Genetic_Algorithm')
        results['Trial'].append(trial + 1)
        results['Best_Tour_Length'].append(best_length)
        results['Computation_Time'].append(computation_time)
        results['Generations_Completed'].append(generations)
        results['Cities'].append(config.cities)

    # FE trials
    for trial in range(config.trials_per_algorithm):
        start_time = time.time()
        fe_solver = ForgettingEngineTSP(cities, config.fe_population, 
                                        config.fe_forget_rate, True)
        best_tour, best_length, generations = fe_solver.solve(config.max_generations)
        computation_time = time.time() - start_time

        results['Algorithm'].append('Forgetting_Engine')
        results['Trial'].append(trial + 1)
        results['Best_Tour_Length'].append(best_length)
        results['Computation_Time'].append(computation_time)
        results['Generations_Completed'].append(generations)
        results['Cities'].append(config.cities)

    return pd.DataFrame(results)

# Execute experiment
if __name__ == "__main__":
    results_df = run_experiment()
    results_df.to_csv('experiment_results.csv', index=False)
    print("Experiment completed. Results saved to experiment_results.csv")
