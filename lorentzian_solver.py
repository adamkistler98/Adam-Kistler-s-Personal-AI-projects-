import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import torch
import pandas as pd
import plotly.graph_objects as go
import io
import time

# --- 1. HUD STYLE CONFIGURATION ---
st.set_page_config(page_title="Lorentzian Metric Solver", layout="wide", page_icon="🌌")

st.markdown(r"""
<style>
    .stApp { background-color: #000000 !important; }
    h1, h2, h3, h4 { color: #00ADB5 !important; font-family: 'Consolas', monospace; }
    p, li, label, .stMarkdown, .stCaption { color: #FFFFFF !important; font-size: 14px; }
    
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, input, select {
        background-color: #161B22 !important; 
        color: #00FFF5 !important; 
        border: 1px solid #00ADB5 !important;
    }
    div[data-testid="stMetricValue"] { color: #00FF41 !important; font-family: 'Consolas', monospace; }
    section[data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #222; }
    
    div.stButton > button, div.stDownloadButton > button { 
        border: 1px solid #00ADB5 !important; color: #00ADB5 !important; background-color: #161B22 !important; 
        width: 100%; border-radius: 2px; font-weight: bold; text-transform: uppercase;
    }
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
            if metric_type == "Kerr-Newman (Charge + Rotation)":
                # b(r) = 2M, param[0]=Q, param[1]=a
                return db_dr - (2 * r / (r**2 + param[1]**2)) * b + (param[0]**2 / r**2)
            elif metric_type == "Einstein-Rosen Bridge":
                return db_dr - (b / r) # Classic Schwarzschild fold
            elif metric_type == "JNW (Naked Singularity)":
                return db_dr - (b / (r * param))
            elif metric_type == "Ellis Drainhole":
                return db_dr - (b / (r**2 + param**2))
            return db_dr - (b / r)

        bc = dde.icbc.DirichletBC(geom, lambda x: r0, lambda x, on: on and np.isclose(x[0], r0))
        data = dde.data.PDE(geom, pde, bc, num_domain=500, num_boundary=50)
        net = dde.nn.FNN([1, 64, 64, 64, 1], "tanh", "Glorot normal")
        model = dde.Model(data, net)
        model.compile("adam", lr=lr)
        loss, _ = model.train(iterations=iters, display_every=1000)
        return model, loss

# --- 3. DASHBOARD ---
st.sidebar.markdown(r"### 🛠️ MANIFOLD SELECTOR")
metric_type = st.sidebar.selectbox("Spacetime Metric", 
    ["Kerr-Newman (Charge + Rotation)", "Einstein-Rosen Bridge", 
     "JNW (Naked Singularity)", "Ellis Drainhole"])

st.sidebar.markdown(r"### 🧬 TOPOLOGY CONFIG")
r0 = st.sidebar.number_input(r"Horizon/Throat ($r_0$)", 0.1, 100.0, 5.0, format="%.4f")

if metric_type == "Kerr-Newman (Charge + Rotation)":
    q = st.sidebar.slider(r"Charge ($Q$)", 0.0, 5.0, 1.0)
    a = st.sidebar.slider(r"Rotation ($a$)", 0.0, 5.0, 1.0)
    param = [q, a]
elif metric_type == "JNW (Naked Singularity)":
    param = st.sidebar.slider(r"Scalar Strength ($s$)", 0.1, 2.0, 1.0)
elif metric_type == "Ellis Drainhole":
    param = st.sidebar.slider(r"Flow Intensity ($n$)", 1.0, 10.0, 2.0)
else: # ER Bridge
    param = 1.0

st.sidebar.markdown(r"### ⚙️ NUMERICAL KERNEL")
lr_val = st.sidebar.number_input(r"Learning Rate ($\eta$)", 0.0001, 0.01, 0.001, format="%.4f")
epochs = st.sidebar.select_slider("Epochs", options=[1000, 2500, 5000], value=2500)

pause = st.sidebar.toggle("HALT SIMULATION", value=False)

# Solver Execution
model, hist = SpacetimeSolver.solve_manifold(metric_type, r0, r0 * 10, param, epochs, lr_val)
# ... Extraction logic same as before ...

# Results Display
st.metric("KERNEL CONVERGENCE", f"{hist.loss_train[-1][0]:.2e}")
st.markdown("---")

# [The 3D Surface and Charting Logic follows here as in previous versions]
