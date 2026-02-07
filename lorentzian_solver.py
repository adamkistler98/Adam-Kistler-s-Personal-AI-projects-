import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import torch
import pandas as pd
import plotly.graph_objects as go
import io
import time

# --- 1. UI CONFIGURATION & NUCLEAR STEALTH CSS ---
st.set_page_config(page_title="Lorentzian Metric Solver", layout="wide", page_icon="🌌")

st.markdown(r"""
<style>
    .stApp { background-color: #000000 !important; }
    h1, h2, h3, h4 { color: #00ADB5 !important; font-family: 'Consolas', monospace; }
    p, li, label, .stMarkdown, .stCaption { color: #FFFFFF !important; font-size: 13px; }
    
    /* TOTAL STEALTH OVERRIDE */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, input, select, .stSelectbox, .stNumberInput {
        background-color: #161B22 !important; color: #00FFF5 !important; border: 1px solid #00ADB5 !important;
    }
    div[data-baseweb="popover"], ul[role="listbox"], li[role="option"] {
        background-color: #161B22 !important; color: #00FFF5 !important; border: 1px solid #00ADB5 !important;
    }
    li[role="option"]:hover, li[aria-selected="true"] { background-color: #1f242d !important; color: #00FFF5 !important; }

    div[data-testid="stMetricValue"] { color: #00FF41 !important; font-family: 'Consolas', monospace; }
    section[data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #222; }
    
    div.stDownloadButton > button { 
        border: 1px solid #00ADB5 !important; color: #00ADB5 !important; background-color: #161B22 !important; 
        width: 100%; border-radius: 2px; font-weight: bold; text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. THE UNIVERSAL PHYSICS KERNEL ---
class SpacetimeSolver:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def solve_manifold(metric_type, r0, r_max, params, iters, lr):
        geom = dde.geometry.Interval(r0, r_max)
        def pde(r, b):
            db_dr = dde.grad.jacobian(b, r)
            # Comprehensive Physics Residuals
            if metric_type == "Morris-Thorne Wormhole":
                return db_dr - (b / r) * params[0] + (params[1] * (params[2] - b/r))
            elif "Kerr" in metric_type:
                eff_q = np.sqrt(params[1]**2 + (params[3]**2 if len(params)>3 else 0))
                return db_dr - (2 * params[0] * r / (r**2 + params[2]**2)) * b + (eff_q**2 / r**2)
            elif "Warp" in metric_type:
                return db_dr + (params[0] * b * (1-b)**params[2]) / (params[1] * params[3] + 1e-6)
            elif "Sitter" in metric_type:
                sign = -1 if "Expansion" in metric_type else 1
                return db_dr - (b / r) + (sign * params[0] * r**params[1] * params[2])
            elif "Vaidya" in metric_type:
                return db_dr - (b / r) * (1 - params[0] * params[1] * params[2])
            elif "Stringy" in metric_type:
                return db_dr - (b / (r - params[0] * params[1] * params[2]))
            elif "Bonnor-Melvin" in metric_type:
                return db_dr - (params[0]**2 * r) # Magnetic Universe
            return db_dr - (b / r)

        bc_val = r0 if "Warp" not in metric_type else 1.0
        bc = dde.icbc.DirichletBC(geom, lambda x: bc_val, lambda x, on: on and np.isclose(x[0], r0))
        data = dde.data.PDE(geom, pde, bc, num_domain=400, num_boundary=40)
        net = dde.nn.FNN([1, 64, 64, 64, 1], "tanh", "Glorot normal")
        model = dde.Model(data, net)
        model.compile("adam", lr=lr)
        loss, _ = model.train(iterations=iters, display_every=1000)
        return model, loss

    @staticmethod
    def extract_telemetry(model, metric_type, r0, r_max):
        r_v = np.linspace(r0, r_max, 600).reshape(-1, 1)
        b = model.net(torch.tensor(r_v, dtype=torch.float32)).detach().numpy()
        rho = np.gradient(b.flatten(), r_v.flatten()) / (8 * np.pi * r_v.flatten()**2 + 1e-12)
        
        # 3D Mapping A: Spatial Curvature
        z = np.zeros_like(r_v)
        dr = r_v[1] - r_v[0]
        for i in range(1, len(r_v)):
            val = (r_v[i] / (b[i] + 1e-9)) - 1 if "Warp" not in metric_type else 0.1
            z[i] = z[i-1] + (1.0 / np.sqrt(np.abs(val)) if np.abs(val) > 1e-9 else 10.0) * dr
            
        # 3D Mapping B: Gravitational Potential (g_tt)
        pot = -np.log(np.abs(1 - b.flatten()/r_v.flatten()) + 1e-6)
        
        return r_v, b, rho, z, pot

# --- 3. THE UNIVERSAL SIDEBAR ---
st.sidebar.markdown("### 🛠️ MANIFOLD & CONSTANTS")
metric_list = [
    "Morris-Thorne Wormhole", "Kerr Black Hole", "Alcubierre Warp Drive", 
    "Reissner-Nordström (Charged)", "Schwarzschild-de Sitter (Expansion)", 
    "Schwarzschild-AdS (Contraction)", "GHS Stringy Black Hole", 
    "Vaidya (Radiating Star)", "Kerr-Newman (Charge + Rotation)", 
    "Einstein-Rosen Bridge", "JNW (Naked Singularity)", "Ellis Drainhole",
    "Bonnor-Melvin (Magnetic Universe)", "Gott Cosmic String"
]
metric_type = st.sidebar.selectbox("Metric Class", metric_list)

r0 = st.sidebar.number_input(r"Base Scale ($r_0$ / $M$)", 0.1, 1000.0, 5.0, format="%.4f")

# DYNAMIC PHYSICS PARAMETERS
params = []
if metric_type == "Morris-Thorne Wormhole":
    params = [st.sidebar.slider("Curvature (κ)", 0.1, 1.0, 0.5), st.sidebar.slider("Redshift (Φ)", 0.0, 1.0, 0.0), st.sidebar.slider("Exoticity (ξ)", 0.0, 2.0, 1.0)]
elif "Kerr" in metric_type:
    params = [st.sidebar.number_input("Mass (M)", 1.0, 100.0, 5.0), st.sidebar.slider("Electric (Q)", 0.0, 10.0, 0.0), st.sidebar.slider("Spin (a)", 0.0, 10.0, 1.0), st.sidebar.slider("Magnetic (P)", 0.0, 10.0, 0.0)]
elif "Warp" in metric_type:
    params = [st.sidebar.slider("Velocity (v/c)", 0.1, 10.0, 1.0), st.sidebar.slider("Sigma (σ)", 0.1, 5.0, 1.0), st.sidebar.slider("Thickness (w)", 1, 10, 2), st.sidebar.slider("Modulation", 0.1, 2.0, 1.0)]
elif "Sitter" in metric_type or "AdS" in metric_type:
    params = [st.sidebar.number_input("Lambda (Λ)", 0.0, 0.01, 0.0001, format="%.6f"), st.sidebar.slider("Curvature (k)", 1, 3, 2), st.sidebar.slider("Omega (Ω)", 0.1, 1.0, 1.0)]
elif "Vaidya" in metric_type:
    params = [st.sidebar.slider("Mass Loss (Ṁ)", 0.0, 1.0, 0.1), st.sidebar.slider("Luminosity (L)", 0.1, 10.0, 1.0), st.sidebar.slider("Radial Flux (q)", 0.1, 5.0, 1.0)]
elif "Bonnor" in metric_type:
    params = [st.sidebar.slider("Magnetic Field Strength", 0.1, 10.0, 1.0)]
else:
    params = [st.sidebar.slider("Coupling / Strength", 0.1, 5.0, 1.0)]

lr_val = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
epochs = st.sidebar.select_slider("Epochs", options=[1000, 2500, 5000], value=2500)
pause = st.sidebar.toggle("HALT SIMULATION", value=False)

# EXECUTION
model, hist = SpacetimeSolver.solve_manifold(metric_type, r0, r0 * 10, params, epochs, lr_val)
r, b, rho, z, pot = SpacetimeSolver.extract_telemetry(model, metric_type, r0, r0 * 10)

# DASHBOARD LAYOUT
m1, m2, m3 = st.columns(3)
m1.metric("CONVERGENCE", f"{hist.loss_train[-1][0]:.2e}")
m2.metric("CLASS", metric_type.split()[0])
m3.metric("PEAK ENERGY", f"{np.max(np.abs(rho)):.4f}")

st.markdown("---")
v_col, d_col = st.columns([2, 1])

with v_col:
    th = np.linspace(0, 2*np.pi, 60)
    R, T = np.meshgrid(r.flatten(), th)
    Z_geom = np.tile(z.flatten(), (60, 1))
    Z_pot = np.tile(pot.flatten(), (60, 1))
    
    st.subheader("Manifold A: Lorentzian Embedding (Spatial Curvature)")
    fig1 = go.Figure(data=[go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=Z_geom, colorscale='Viridis', showscale=False)])
    fig1.update_layout(template="plotly_dark", scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0), height=400)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Manifold B: Potential Horizon ($g_{tt}$ Field Strength)")
    fig2 = go.Figure(data=[go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=Z_pot, colorscale='Magma', showscale=False)])
    fig2.update_layout(template="plotly_dark", scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0), height=400)
    st.plotly_chart(fig2, use_container_width=True)

with d_col:
    fig_stack, (ax1, ax2) = plt.subplots(2, 1, facecolor='black', figsize=(6, 9))
    ax1.set_facecolor('black'); ax1.plot(r, rho, color='#FF2E63', lw=2)
    ax1.set_title("Energy Density Profile", color='white', fontsize=10)
    ax1.tick_params(colors='white', labelsize=8); ax1.grid(alpha=0.1)
    
    ax2.set_facecolor('black'); ax2.plot(r, b, color='#00ADB5', lw=2)
    ax2.set_title("Metric Shape Function b(r)", color='white', fontsize=10)
    ax2.tick_params(colors='white', labelsize=8); ax2.grid(alpha=0.1)
    
    plt.tight_layout()
    st.pyplot(fig_stack)
    
    st.download_button("📥 EXPORT TELEMETRY (CSV)", data=pd.DataFrame({"r":r.flatten(),"b":b.flatten(),"density":rho.flatten()}).to_csv().encode('utf-8'), file_name="telemetry.csv", use_container_width=True)
    
    # INDENTATION FIX & VISUALS
    if "Wormhole" in metric_type:
        
    elif "Kerr" in metric_type:
        
    elif "Charged" in metric_type:
        
    elif "Expansion" in metric_type:
        
    
    pass

if not pause:
    time.sleep(0.01)
    st.rerun()
