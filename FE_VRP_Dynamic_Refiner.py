import numpy as np
import logging

class VRPDynaPilot:
    """The Pilot: Adjusts Paradox Sensitivity based on Constraint Choke."""
    def __init__(self, operator_state="DEFAULT"):
        self.state = operator_state
        # Base Sensitivity: High for Lucid, near-zero for Default.
        self.discovery_bias = 0.99 if operator_state == "LUCID" else 0.05
        self.p_rate = 0.35 if operator_state == "LUCID" else 0.02

    def adjust_throttle(self, violation_rate):
        """
        If the system is 'choking' on rules (Capacity/Time), we open 
        the paradox throttle to find a globally valid optimal.
        """
        if self.state != "LUCID": return

        # High violation rate means standard search is hitting a wall
        if violation_rate > 0.45:
            self.p_rate = min(self.p_rate + 0.20, 0.90)
            logging.info(f"[*] CONSTRAINT CHOKE: Opening Paradox Throttle to {self.p_rate:.2f}")
        else:
            self.p_rate = max(self.p_rate - 0.05, 0.35)

class FE_VRP_Engine:
    """The Immutable Engine: Capacity-Aware Route Elimination."""
    def __init__(self, pilot, domain):
        self.pilot = pilot
        self.domain = domain
        self.paradox_buffer = []

    def run_optimization(self, routes):
        # 1. Measure the 'Weather' (Violation Rate)
        violations = [r for r in routes if r.load > self.domain.capacity]
        violation_rate = len(violations) / len(routes)
        
        # 2. Pilot Adjusts Throttle
        self.pilot.adjust_throttle(violation_rate)
        
        # 3. THE INTERLOCK: Paradox Retention (Patent 8)
        # Standard: Must be short and valid.
        # FE: Can be long and valid if (Balance * Bias) is high.
        for route in eliminated:
            if (route.balance_score * self.pilot.discovery_bias) > 0.85:
                self.paradox_buffer.append(route)