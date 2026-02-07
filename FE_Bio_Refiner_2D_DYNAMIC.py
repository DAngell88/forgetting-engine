import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

class Dynamic2DPilot:
    """The Pilot: Adjusts Paradox Retention based on 2D Lattice 'clumping'."""
    def __init__(self, operator_state="DEFAULT"):
        self.state = operator_state
        self.p_rate = 0.20 if operator_state == "LUCID" else 0.05
        self.discovery_bias = 0.95 if operator_state == "LUCID" else 0.02

    def adjust_throttle(self, rg_trend, current_energy):
        """
        If the Radius of Gyration (Rg) isn't changing, the protein is stuck.
        We spike the paradox rate to 'un-clump' the lattice.
        """
        if self.state != "LUCID":
            return

        if abs(rg_trend) < 0.01:
            self.p_rate = min(self.p_rate + 0.1, 0.6)
            logging.info(f"[*] STAGNATION DETECTED: Opening Paradox Throttle to {self.p_rate:.2f}")
        else:
            self.p_rate = max(self.p_rate - 0.05, 0.20)

class FE_2D_DynamicRefiner:
    """The Immutable 2D Engine with Dynamic Interlock."""
    def __init__(self, pilot, sequence):
        self.pilot = pilot
        self.sequence = sequence
        self.n = len(sequence)

    def calculate_rg(self, coords):
        """Radius of Gyration: Proxy for structural complexity."""
        centroid = np.mean(coords, axis=0)
        return np.sqrt(np.mean(np.sum((coords - centroid)**2, axis=1)))

    def get_energy(self, coords):
        """Standard HP Energy: -1 for H-H contacts."""
        energy = 0
        h_indices = [i for i, char in enumerate(self.sequence) if char == 'H']
        for i in h_indices:
            for j in h_indices:
                if i < j - 1:
                    dist = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                    if dist == 1: energy -= 1
        return energy

    def run_fold(self, iterations=2000):
        # Initial straight line
        coords = np.array([(i, 0) for i in range(self.n)])
        current_energy = self.get_energy(coords)
        last_rg = self.calculate_rg(coords)
        
        for i in range(iterations):
            # Propose move (Pivot)
            new_coords = self.apply_pivot(coords) # Logic from Protein-2D-FIXED.py
            new_energy = self.get_energy(new_coords)
            new_rg = self.calculate_rg(new_coords)

            # 1. Standard Optimization
            if new_energy <= current_energy:
                coords, current_energy = new_coords, new_energy
            
            # 2. THE DYNAMIC INTERLOCK
            else:
                penalty = new_energy - current_energy
                # The Pilot decides based on 'Discovery Bias'
                if (new_rg * self.pilot.discovery_bias) > (penalty / 2.0):
                    coords, current_energy = new_coords, new_energy

            # 3. Dynamic Adjustment every 50 steps
            if i % 50 == 0:
                rg_trend = new_rg - last_rg
                self.pilot.adjust_throttle(rg_trend, current_energy)
                last_rg = new_rg

        return current_energy

    def apply_pivot(self, coords):
        # Implementation of the pivot move from Canon assets
        idx = random.randint(1, self.n - 1)
        pivot = coords[idx]
        new_coords = coords.copy()
        for i in range(idx, self.n):
            dx, dy = coords[i][0] - pivot[0], coords[i][1] - pivot[1]
            new_coords[i] = [pivot[0] - dy, pivot[1] + dx]
        return new_coords