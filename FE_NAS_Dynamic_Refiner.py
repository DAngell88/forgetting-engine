import numpy as np
import logging
from scipy.stats import entropy

logging.basicConfig(level=logging.INFO, format='%(message)s')

class NASDynaPilot:
    """The Pilot: Adjusts Paradox Sensitivity based on Model Novelty."""
    def __init__(self, operator_state="DEFAULT"):
        self.state = operator_state
        self.discovery_bias = 0.98 if operator_state == "LUCID" else 0.05
        self.p_rate = 0.25 if operator_state == "LUCID" else 0.02

    def adjust_throttle(self, population_diversity):
        """
        If the models in the current 'Council' are too similar (Stagnation),
        we open the throttle to favor high-entropy, 'weird' architectures.
        """
        if self.state != "LUCID": return

        if population_diversity < 0.4:
            self.p_rate = min(self.p_rate + 0.15, 0.80)
            logging.info(f"[*] ARCHITECTURAL STAGNATION: Opening Paradox Throttle to {self.p_rate:.2f}")
        else:
            self.p_rate = max(self.p_rate - 0.05, 0.25)

class FE_NAS_Engine:
    """The Immutable Engine: Neural Architecture Search via Paradox Retention."""
    def __init__(self, pilot):
        self.pilot = pilot
        self.paradox_buffer = []

    def evaluate_architecture(self, layers):
        # Truth Channel: Simulated Accuracy
        # Contradiction: Layer Diversity vs. Parameter Count
        diversity = len(set(layers)) / len(layers)
        
        # 1. Pilot adjusts the throttle based on current Council diversity
        self.pilot.adjust_throttle(diversity)
        
        # 2. Paradox Check: (Novelty * Operator Intent)
        novelty_entropy = entropy([layers.count(x) for x in set(layers)])
        if (novelty_entropy * self.pilot.discovery_bias) > 0.75:
            # We RETAIN this 'weird' model even if initial accuracy is low
            self.paradox_buffer.append(layers)
            logging.info(f">> NOVEL ARCHITECTURE RETAINED: {layers}")