import streamlit as st
import numpy as np
from rk_methods import solve_ode
from economic_models import (
    business_cycle_model, 
    monetary_policy_model,
    fiscal_policy_model
)
from utils import create_simulation_plot, export_results

st.set_page_config(page_title="Economic Policy Analysis", layout="wide")

# Load custom CSS
with open('assets/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("Economic Policy Analysis Tool")
st.markdown("""
This application implements Runge-Kutta methods for analyzing economic policy scenarios
through dynamic system modeling.
""")

# Sidebar for model selection and parameters
st.sidebar.header("Model Configuration")

model_type = st.sidebar.selectbox(
    "Select Economic Model",
    ["Business Cycle", "Monetary Policy", "Fiscal Policy"]
)

# Time parameters
st.sidebar.subheader("Simulation Parameters")
t_start = st.sidebar.number_input("Start Time", value=0.0)
t_end = st.sidebar.number_input("End Time", value=10.0)
step_size = st.sidebar.number_input("Step Size", value=0.1, min_value=0.01)

rk_method = st.sidebar.selectbox(
    "Numerical Method",
    ["RK4", "RK2", "RK1"]
)

# Model-specific parameters
st.sidebar.subheader("Model Parameters")

if model_type == "Business Cycle":
    alpha = st.sidebar.slider("Capital Share (α)", 0.1, 0.9, 0.3)
    beta = st.sidebar.slider("Discount Factor (β)", 0.1, 0.99, 0.95)
    delta = st.sidebar.slider("Depreciation Rate (δ)", 0.01, 0.2, 0.1)
    
    model_func = lambda t, y: business_cycle_model(t, y, alpha, beta, delta)
    initial_conditions = np.array([1.0, 1.0])
    labels = ["Output", "Capital"]
    
elif model_type == "Monetary Policy":
    phi_pi = st.sidebar.slider("Inflation Response (φπ)", 1.0, 3.0, 1.5)
    phi_y = st.sidebar.slider("Output Response (φy)", 0.0, 1.0, 0.5)
    sigma = st.sidebar.slider("Risk Aversion (σ)", 0.5, 2.0, 1.0)
    
    model_func = lambda t, y: monetary_policy_model(t, y, phi_pi, phi_y, sigma)
    initial_conditions = np.array([0.02, 0.01])  # 2% inflation, 1% output gap
    labels = ["Inflation", "Output Gap"]
    
else:  # Fiscal Policy
    g = st.sidebar.slider("Government Spending (G/Y)", 0.1, 0.5, 0.2)
    tau = st.sidebar.slider("Tax Rate (τ)", 0.1, 0.5, 0.3)
    theta = st.sidebar.slider("Debt Response (θ)", 0.0, 0.3, 0.1)
    
    model_func = lambda t, y: fiscal_policy_model(t, y, g, tau, theta)
    initial_conditions = np.array([0.6, 1.0])  # 60% debt-to-GDP, normal output
    labels = ["Debt-to-GDP", "Output"]

# Run simulation
if st.sidebar.button("Run Simulation"):
    try:
        t, y = solve_ode(
            model_func,
            (t_start, t_end),
            initial_conditions,
            step_size,
            rk_method
        )
        
        # Create and display plot
        fig = create_simulation_plot(t, y, labels, f"{model_type} Model Simulation")
        st.plotly_chart(fig, use_container_width=True)
        
        # Export results
        if st.button("Export Results"):
            df = export_results(t, y, labels, "simulation_results.csv")
            st.write("Results exported to simulation_results.csv")
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"An error occurred during simulation: {str(e)}")

# Documentation
with st.expander("Documentation"):
    st.markdown("""
    ### Model Descriptions
    
    #### Business Cycle Model
    - Models the interaction between output and capital
    - Parameters include capital share (α), discount factor (β), and depreciation rate (δ)
    
    #### Monetary Policy Model
    - New Keynesian model with inflation and output gap
    - Features Taylor rule parameters for monetary policy
    
    #### Fiscal Policy Model
    - Analyzes government debt and output dynamics
    - Includes government spending, taxation, and debt response parameters
    
    ### Numerical Methods
    - RK4: Classical 4th order Runge-Kutta method (most accurate)
    - RK2: 2nd order method (balanced accuracy/speed)
    - RK1: Euler method (fastest but least accurate)
    """)
