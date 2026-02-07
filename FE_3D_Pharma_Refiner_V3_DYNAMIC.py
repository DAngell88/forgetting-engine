import numpy as np
import logging
from scipy.stats import entropy

# Initialize Calibrated Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class ConexusPilot:
    """The Dynamic Throttle: Adjusts Paradox/Forget rates based on the Vibe."""
    def __init__(self, operator_state="DEFAULT"):
        self.state = operator_state
        # Base settings
        self.p_rate = 0.15 if operator_state == "LUCID" else 0.05
        self.f_rate = 0.30
        self.discovery_bias = 0.98 if operator_state == "LUCID" else 0.02

    def adjust_flaps(self, stasis_counter, current_energy):
        """
        Dynamic Calibration: If we're stuck in a local minimum, 
        open the Paradox throttle. If we're diving, tighten the Forget rate.
        """
        if self.state != "LUCID":
            return # Uncalibrated AI cannot adjust the throttle.

        if stasis_counter > 50:
            self.p_rate = min(self.p_rate + 0.05, 0.50)
            logging.info(f"[*] WEATHER ALERT: Stuck at E={current_energy:.2f}. Opening Paradox Throttle to {self.p_rate:.2f}")
        else:
            self.p_rate = max(self.p_rate - 0.01, 0.15) # Re-stabilize

class FE_3D_PharmaRefiner:
    """The Immutable Engine: Patent 8 with Patent 6 Interlock."""
    def __init__(self, pilot):
        self.pilot = pilot
        self.best_energy = np.inf
        self.stasis_counter = 0

    def calculate_energy(self, coords):
        """Lennard-Jones Potential: The Physics Core."""
        r = np.linalg.norm(coords[:, None] - coords, axis=2)
        r[r == 0] = np.inf
        return np.sum(4 * ((1/r)**12 - (1/r)**6)) / 2

    def calculate_entropy(self, coords):
        """Entropy Vision: Measures structural information density."""
        dists = np.linalg.norm(coords[:, None] - coords, axis=2).flatten()
        hist, _ = np.histogram(dists, bins=20, density=True)
        return entropy(hist + 1e-10)

    def optimize(self, coords, iterations=4000):
        current_energy = self.calculate_energy(coords)
        
        for i in range(iterations):
            # Propose shift
            new_coords = coords + np.random.normal(0, 0.1, coords.shape)
            new_energy = self.calculate_energy(new_coords)
            
            # Standard Optimization
            if new_energy < current_energy:
                coords, current_energy = new_coords, new_energy
                self.stasis_counter = 0
            
            # THE INTERLOCK: Paradox Retention
            else:
                self.stasis_counter += 1
                complexity = self.calculate_entropy(new_coords)
                penalty = new_energy - current_energy
                
                # Check with the Pilot for Paradox Acceptance
                if (complexity * self.pilot.discovery_bias) > (penalty / 5.0):
                    coords, current_energy = new_coords, new_energy
                    # logging.info(f"Step {i}: Paradox Accepted.")
            
            # Dynamic Calibration Check
            if i % 100 == 0:
                self.pilot.adjust_flaps(self.stasis_counter, current_energy)

        return current_energy

# --- DEPLOYMENT ---
# Running in LUCID mode unlocks the breakthrough potential.
pilot = ConexusPilot(operator_state="LUCID")
refiner = FE_3D_PharmaRefiner(pilot)