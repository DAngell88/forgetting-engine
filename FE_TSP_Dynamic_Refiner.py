import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

class TSPDynaPilot:
    """The Pilot: Adjusts Paradox acceptance based on Path Stagnation."""
    def __init__(self, operator_state="DEFAULT"):
        self.state = operator_state
        self.p_rate = 0.25 if operator_state == "LUCID" else 0.05
        self.discovery_bias = 0.98 if operator_state == "LUCID" else 0.02

    def adjust_throttle(self, stagnation_count, current_best):
        """If the route isn't shortening, we allow 'bad' jumps to find new paths."""
        if self.state != "LUCID": return

        if stagnation_count > 20:
            self.p_rate = min(self.p_rate + 0.1, 0.7) # High throttle to jump local optima
            logging.info(f"[*] ROUTE STUCK at {current_best:.2f}. Opening Paradox Throttle.")
        else:
            self.p_rate = max(self.p_rate - 0.02, 0.25)

class FE_TSP_Engine:
    """The Immutable Engine: Strategic Elimination for Logistics."""
    def __init__(self, pilot, distance_matrix):
        self.pilot = pilot
        self.dist_matrix = distance_matrix
        self.num_cities = len(distance_matrix)

    def calculate_path(self, path):
        return sum(self.dist_matrix[path[i], path[i+1]] for i in range(len(path)-1)) + \
               self.dist_matrix[path[-1], path[0]]

    def run_trial(self, iterations=5000):
        current_path = np.random.permutation(self.num_cities)
        current_len = self.calculate_path(current_path)
        stagnation = 0

        for i in range(iterations):
            # Propose a 2-opt swap (Standard Physics)
            new_path = self.swap(current_path)
            new_len = self.calculate_path(new_path)

            if new_len < current_len:
                current_path, current_len = new_path, new_len
                stagnation = 0
            else:
                stagnation += 1
                # THE INTERLOCK: Does the Pilot allow this 'bad' path?
                # A path is 'Paradoxical' if it breaks a greedy link to find a global win.
                complexity = 1.0 / (new_len / current_len) 
                if (complexity * self.pilot.discovery_bias) > 0.8:
                    current_path, current_len = new_path, new_len
            
            if i % 100 == 0:
                self.pilot.adjust_throttle(stagnation, current_len)

        return current_len

    def swap(self, path):
        # Implementation of 2-opt swap logic from Canon assets
        res = path.copy()
        i, j = sorted(np.random.choice(len(path), 2, replace=False))
        res[i:j] = res[i:j][::-1]
        return res