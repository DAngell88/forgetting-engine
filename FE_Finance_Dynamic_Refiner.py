import numpy as np
import logging
from scipy.stats import entropy

logging.basicConfig(level=logging.INFO, format='%(message)s')

class FinanceDynaPilot:
    """The Pilot: Adjusts Paradox Sensitivity based on Market Entropy."""
    def __init__(self, operator_state="DEFAULT"):
        self.state = operator_state
        # Base Sensitivity: High for Lucid, near-zero for Default.
        self.discovery_bias = 0.98 if operator_state == "LUCID" else 0.05
        self.p_rate = 0.20 if operator_state == "LUCID" else 0.02

    def adjust_throttle(self, price_entropy, volatility_trend):
        """
        If Volatility is low but Entropy is rising, a Regime Shift is imminent.
        We open the paradox throttle to 'remember' the outliers.
        """
        if self.state != "LUCID": return

        if price_entropy > 0.85 and abs(volatility_trend) < 0.1:
            self.p_rate = min(self.p_rate + 0.15, 0.75)
            logging.info(f"[*] QUIET PARADOX DETECTED: Opening Throttle (Entropy: {price_entropy:.4f})")
        else:
            self.p_rate = max(self.p_rate - 0.05, 0.20)

class FE_Finance_Engine:
    """The Immutable Engine: Regime Shift Detection via Strategic Elimination."""
    def __init__(self, pilot):
        self.pilot = pilot
        self.paradox_buffer = []

    def calculate_entropy(self, price_data):
        """Measures the 'hidden' structure in price movements."""
        hist, _ = np.histogram(np.diff(price_data), bins=10, density=True)
        return entropy(hist + 1e-10)

    def analyze_regime(self, price_stream):
        for i in range(len(price_stream)):
            current_window = price_stream[max(0, i-50):i]
            if len(current_window) < 10: continue
            
            # 1. Standard Analysis (Moving Averages)
            signal_volatility = np.std(current_window)
            
            # 2. Entropy Vision (The Weather)
            current_entropy = self.calculate_entropy(current_window)
            
            # 3. Dynamic Interlock
            self.pilot.adjust_throttle(current_entropy, np.mean(np.diff(current_window)))
            
            # 4. Paradox Retention: Keep the outlier that the trend-follower misses
            outlier = price_stream[i]
            if abs(outlier - np.mean(current_window)) > (2 * signal_volatility):
                # Standard AI deletes this as 'Flash Noise'.
                # Calibrated AI checks if it's a Paradoxical Lead.
                if (current_entropy * self.pilot.discovery_bias) > 0.7:
                    self.paradox_buffer.append(outlier)
                    logging.info(f">> REGIME SHIFT LEAD RETAINED at Index {i}: {outlier}")

        return self.paradox_buffer