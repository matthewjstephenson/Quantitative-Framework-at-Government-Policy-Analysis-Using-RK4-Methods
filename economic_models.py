import numpy as np

class EconomicModel:
    def __init__(self, params):
        self.params = params
    
    def is_model_valid(self):
        """Check if model parameters are valid"""
        return all(p > 0 for p in self.params.values())

def business_cycle_model(t, y, alpha=0.3, beta=0.95, delta=0.1):
    """
    Simple business cycle model
    y[0]: output
    y[1]: capital
    """
    output = y[0]
    capital = y[1]
    
    # Production function (Cobb-Douglas)
    dy_output = alpha * capital**(alpha-1) - delta
    # Capital accumulation
    dy_capital = beta * (output - delta*capital)
    
    return np.array([dy_output, dy_capital])

def monetary_policy_model(t, y, phi_pi=1.5, phi_y=0.5, sigma=1.0):
    """
    Basic New Keynesian model
    y[0]: inflation
    y[1]: output gap
    """
    inflation = y[0]
    output_gap = y[1]
    
    # Phillips curve
    dy_inflation = sigma * output_gap
    # IS curve with Taylor rule
    dy_output = -(1/sigma) * (phi_pi*inflation + phi_y*output_gap)
    
    return np.array([dy_inflation, dy_output])

def fiscal_policy_model(t, y, g=0.2, tau=0.3, theta=0.1):
    """
    Simple fiscal policy model
    y[0]: debt
    y[1]: output
    """
    debt = y[0]
    output = y[1]
    
    # Debt dynamics
    dy_debt = (1-tau)*output - g + theta*debt
    # Output dynamics
    dy_output = g - tau*output
    
    return np.array([dy_debt, dy_output])
