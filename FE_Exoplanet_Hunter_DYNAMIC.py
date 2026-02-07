import numpy as np
from scipy.stats import entropy
from astropy.timeseries import BoxLeastSquares
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

class DynamicAstroPilot:
    """The Pilot: Adjusts Discovery Bias based on Noise Entropy."""
    def __init__(self, operator_state="DEFAULT"):
        self.state = operator_state
        # Base Sensitivity
        self.discovery_bias = 0.98 if operator_state == "LUCID" else 0.05
        self.nasa_threshold = 7.1 # Standard SNR cutoff

    def adjust_sensitivity(self, current_entropy):
        """
        If the noise is 'Structured' (High Entropy), we open the throttle 
        to catch signals that standard SNR would miss.
        """
        if self.state != "LUCID":
            return

        if current_entropy > 0.85:
            self.discovery_bias = 1.0 # Maximum Paradox Retention
            logging.info(f"[*] STRUCTURED NOISE DETECTED (Entropy: {current_entropy:.4f}). Opening Discovery Throttle.")
        else:
            self.discovery_bias = 0.98

class FE_DynamicExoplanetHunter:
    """The Immutable Engine: Patent 8 with Dynamic Astro-Interlock."""
    def __init__(self, pilot):
        self.pilot = pilot
        self.candidates = []

    def calculate_entropy_vision(self, residuals):
        """Measures the complexity of the data left over after a model fit."""
        hist, _ = np.histogram(residuals, bins='auto', density=True)
        return entropy(hist + 1e-10)

    def scan(self, time, flux, star_id):
        # 1. Run Standard BLS (Physics)
        bls = BoxLeastSquares(time, flux)
        periodogram = bls.autopower(0.1)
        best_idx = np.argmax(periodogram.power)
        
        snr = periodogram.depth_snr[best_idx]
        
        # 2. Extract Residuals (The 'Weather')
        model = bls.model(time, periodogram.period[best_idx], 
                          periodogram.duration[best_idx], 
                          periodogram.transit_time[best_idx])
        residuals = flux - model
        
        # 3. Dynamic Entropy Check
        signal_entropy = self.calculate_entropy_vision(residuals)
        self.pilot.adjust_sensitivity(signal_entropy)

        # 4. THE DYNAMIC INTERLOCK
        # Standard: Must pass SNR threshold.
        # FE: Can pass via (Entropy * Bias) even if SNR is low.
        is_standard_pass = snr > self.pilot.nasa_threshold
        is_paradox_pass = (signal_entropy * self.pilot.discovery_bias) > 0.65

        if is_standard_pass or is_paradox_pass:
            reason = "STANDARD" if is_standard_pass else "PARADOX_RECOVERY"
            self.candidates.append({"id": star_id, "snr": snr, "entropy": signal_entropy})
            logging.info(f">> [{star_id}] RETAINED: {reason} (SNR: {snr:.2f} | Entropy: {signal_entropy:.4f})")
        else:
            logging.info(f">> [{star_id}] DISCARDED (SNR: {snr:.2f})")

# --- INITIALIZATION ---
pilot = DynamicAstroPilot(operator_state="LUCID")
hunter = FE_DynamicExoplanetHunter(pilot)