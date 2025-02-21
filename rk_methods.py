import numpy as np

def rk1_step(f, t, y, h):
    """Euler method (RK1)"""
    k1 = h * f(t, y)
    return y + k1

def rk2_step(f, t, y, h):
    """RK2 method"""
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5*h, y + 0.5*k1)
    return y + k2

def rk4_step(f, t, y, h):
    """RK4 method"""
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5*h, y + 0.5*k1)
    k3 = h * f(t + 0.5*h, y + 0.5*k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def solve_ode(f, t_span, y0, h, method='RK4'):
    """Solve ODE using specified RK method"""
    t_start, t_end = t_span
    steps = int((t_end - t_start) / h)
    t = np.linspace(t_start, t_end, steps+1)
    y = np.zeros((steps+1, len(y0)))
    y[0] = y0
    
    step_methods = {
        'RK1': rk1_step,
        'RK2': rk2_step,
        'RK4': rk4_step
    }
    
    step_func = step_methods[method]
    
    for i in range(steps):
        y[i+1] = step_func(f, t[i], y[i], h)
    
    return t, y
