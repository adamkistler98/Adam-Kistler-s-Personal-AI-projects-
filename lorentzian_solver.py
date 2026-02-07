import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import torch
import pandas as pd
from scipy.integrate import simpson
import plotly.graph_objects as go
import io
import time

# --- 1. UI CONFIGURATION ---
st.set_page_config(page_title="Lorentzian Metric Solver", layout="wide", page_icon="🌌")

st.markdown(r"""
<style>
    .stApp { background-color: #000000; }
    h1, h2, h3, h4 { color: #00ADB5 !important; font-family: 'Consolas', monospace; }
    p, li, label, .stMarkdown, .stCaption { color: #FFFFFF !important; }
    input, select { background-color: #161B22 !important; color: #00FFF5 !important; border: 1px solid #333 !important; }
    div[data-baseweb="input"], div[data-baseweb="select"] { background-color: #161B22 !important; border: 1px solid #00ADB5 !important; }
    div[data-testid="stMetricValue"] { color: #00FF41 !important; font-family: 'Consolas', monospace; }
    section[data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #222; }
    div.stButton > button { border: 1px solid #00ADB5 !important; color: #00ADB5 !important; background-color: #161B22 !important; width: 100%; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 2. PHYSICS CORE ---
class SpacetimeSolver:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def solve_manifold(metric_type, r0, r_max, param, iters, lr):
        geom = dde.geometry.Interval(r0, r_max)
        
        def pde(r, b):
            db_dr = dde.grad.jacobian(b, r)
            if metric_type == "Morris-Thorne Wormhole":
                # standard wormhole stability
                return db_dr - (b / r) * param 
            elif metric_type == "Kerr Black Hole":
                # Rotation param 'a' affects the horizon
                return db_dr - (2 * r / (r**2 + param**2)) * b
            else: # Alcubierre Warp Drive
                # Solving for the velocity 'v' shaped pulse
                return db_dr + (param * b * (1-b))
        
        bc_val = r0 if metric_type != "Alcubierre Warp Drive" else 1.0
        bc = dde.icbc.DirichletBC(geom, lambda x: bc_val, lambda x, on: on and np.isclose(x[0], r0))
        
        data = dde.data.PDE(geom, pde, bc, num_domain=500, num_boundary=50)
        net = dde.nn.FNN([1, 64, 64, 64, 1], "tanh", "Glorot normal")
        model = dde.Model(data, net)
        model.compile("adam", lr=lr)
        loss, _ = model.train(iterations=iters, display_every=1000)
        return model, loss

    @staticmethod
    def extract_telemetry(model, metric_type, r0, r_max, param):
        r_v = np.linspace(r0, r_max, 800).reshape(-1, 1)
        b = model.net(torch.tensor(r_v, dtype=torch.float32)).detach().numpy()
        
        # Physics Analysis (Generalized for the dashboard)
        rho = np.gradient(b.flatten(), r_v.flatten()) / (8 * np.pi * r_v.flatten()**2 + 1e-12)
        
        z = np.zeros_like(r_v)
        dr = r_v[1] - r_v[0]
        for i in range(1, len(r_v)):
            # Kerr and Wormhole use embedding, Warp Drive uses flat space modulation
            val = (r_v[i] / (b[i] + 1e-9)) - 1 if metric_type != "Alcubierre Warp Drive" else 0.1
            z[i] = z[i-1] + (1.0 / np.sqrt(np.abs(val)) if np.abs(val) > 1e-9 else 10.0) * dr
            
        return r_v, b, rho, z

# --- 3. DASHBOARD ---
st.sidebar.markdown(r"### 🛠️ MANIFOLD SELECTION")
metric_type = st.sidebar.selectbox("Spacetime Metric", 
    ["Morris-Thorne Wormhole", "Kerr Black Hole", "Alcubierre Warp Drive"])

st.sidebar.markdown(r"### 🧬 TOPOLOGY PARAMETERS")
if metric_type == "Morris-Thorne Wormhole":
    r0 = st.sidebar.number_input("Throat Radius ($r_0$)", 0.1, 100.0, 2.0)
    param = st.sidebar.slider("Curvature ($\kappa$)", 0.1, 0.9, 0.5)
    
elif metric_type == "Kerr Black Hole":
    r0 = st.sidebar.number_input("Event Horizon ($r_h$)", 1.0, 100.0, 5.0)
    param = st.sidebar.slider("Angular Momentum ($a$)", 0.0, 5.0, 1.0)
    
else:
    r0 = st.sidebar.number_input("Bubble Origin", 1.0, 50.0, 5.0)
    param = st.sidebar.slider("Velocity ($v/c$)", 0.1, 5.0, 1.0)
    

st.sidebar.markdown(r"### ⚙️ NUMERICAL KERNEL")
lr_val = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
epochs = st.sidebar.select_slider("Epochs", options=[1000, 2500, 5000], value=2500)

pause = st.sidebar.toggle("HALT SIMULATION", value=False)

# Execution
model, hist = SpacetimeSolver.solve_manifold(metric_type, r0, r0 * 10, param, epochs, lr_val)
r, b, rho, z = SpacetimeSolver.extract_telemetry(model, metric_type, r0, r0 * 10, param)

# Metrics
m1, m2, m3 = st.columns(3)
m1.metric("KERNEL CONVERGENCE", f"{hist.loss_train[-1][0]:.2e}")
m2.metric("METRIC TYPE", metric_type.split()[0])
m3.metric("LOCAL STRESS", f"{np.max(rho):.4f}")

st.markdown("---")
v_col, d_col = st.columns([2, 1])

with v_col:
    # Dynamic 3D Visualization based on Metric
    th = np.linspace(0, 2*np.pi, 60)
    R, T = np.meshgrid(r.flatten(), th)
    Z = np.tile(z.flatten(), (60, 1))
    
    # 3D Mesh
    fig = go.Figure(data=[
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=Z, colorscale='Viridis', showscale=False),
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=-Z, colorscale='Viridis', showscale=False)
    ])
    fig.update_layout(template="plotly_dark", scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='cube'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig, use_container_width=True)

with d_col:
    tabs = st.tabs(["📊 STRESS-ENERGY", "📈 TENSOR PROFILES"])
    with tabs[0]:
        st.subheader("Energy Density Distribution")
        fig, ax = plt.subplots(facecolor='black')
        ax.set_facecolor('black')
        ax.plot(r, rho, color='#FF2E63', lw=2)
        ax.tick_params(colors='white'); ax.grid(alpha=0.1)
        st.pyplot(fig)
        

    with tabs[1]:
        st.subheader("Metric Shape $b(r)$")
        fig2, ax2 = plt.subplots(facecolor='black')
        ax2.set_facecolor('black')
        ax2.plot(r, b, color='#00ADB5', lw=2)
        ax2.tick_params(colors='white'); ax2.grid(alpha=0.1)
        st.pyplot(fig2)

if not pause:
    time.sleep(0.01)
    st.rerun()
