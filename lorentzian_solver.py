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
st.set_page_config(
    page_title="Lorentzian Metric Solver", 
    layout="wide", 
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

st.markdown(r"""
<style>
    .stApp { background-color: #000000 !important; }
    h1, h2, h3, h4 { color: #00ADB5 !important; font-family: 'Consolas', monospace; }
    p, li, label, .stMarkdown, .stCaption { color: #FFFFFF !important; font-size: 14px; }
    
    /* NUCLEAR STEALTH OVERRIDE */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, input, select {
        background-color: #161B22 !important; color: #00FFF5 !important; border: 1px solid #00ADB5 !important;
    }
    div[data-baseweb="popover"], ul[role="listbox"], li[role="option"] {
        background-color: #161B22 !important; color: #00FFF5 !important;
    }

    /* COMPACT BUTTONS */
    div.stButton > button, div.stDownloadButton > button { 
        border: 1px solid #00ADB5 !important; color: #00ADB5 !important; background-color: #161B22 !important; 
        width: 100%; border-radius: 2px; font-weight: bold; font-size: 11px !important;
        text-transform: uppercase; padding: 4px 8px !important; height: 32px !important;
    }
    div.stButton > button:hover { background-color: #1f242d !important; box-shadow: 0 0 10px rgba(0, 173, 181, 0.4); }

    /* METRICS & SIDEBAR */
    div[data-testid="stMetricValue"] { color: #00FF41 !important; font-family: 'Consolas', monospace; }
    section[data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #222; }
    .stTabs [data-baseweb="tab-list"] { background-color: #000000 !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. PHYSICS CORE: 12-METRIC KERNEL ---
class SpacetimeSolver:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def solve_manifold(metric_type, r0, r_max, param, iters, lr):
        geom = dde.geometry.Interval(r0, r_max)
        def pde(r, b):
            db_dr = dde.grad.jacobian(b, r)
            if metric_type == "Morris-Thorne Wormhole": return db_dr - (b / r) * param 
            elif metric_type == "Kerr Black Hole": return db_dr - (2 * r / (r**2 + param**2)) * b
            elif metric_type == "Alcubierre Warp Drive": return db_dr + (param * b * (1-b))
            elif metric_type == "Reissner-Nordström (Charged)": return db_dr - (b / r) + (param**2 / r**2)
            elif "Expansion" in metric_type: return db_dr - (b / r) - (param * r**2)
            elif "Contraction" in metric_type: return db_dr - (b / r) + (param * r**2)
            elif "Stringy" in metric_type: return db_dr - (b / (r - param))
            elif "Radiating" in metric_type: return db_dr - (b / r) * (1 - param)
            elif "Kerr-Newman" in metric_type: return db_dr - (2 * r / (r**2 + param[1]**2)) * b + (param[0]**2 / r**2)
            elif "Naked" in metric_type: return db_dr - (b / (r * param))
            elif "Ellis" in metric_type: return db_dr - (b / (r**2 + param**2))
            return db_dr - (b / r) # ER Bridge

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
        z = np.zeros_like(r_v)
        dr = r_v[1] - r_v[0]
        for i in range(1, len(r_v)):
            val = (r_v[i] / (b[i] + 1e-9)) - 1 if "Warp" not in metric_type else 0.1
            z[i] = z[i-1] + (1.0 / np.sqrt(np.abs(val)) if np.abs(val) > 1e-9 else 10.0) * dr
        return r_v, b, rho, z

# --- 3. DASHBOARD HUD ---
st.title("LORENTZIAN METRIC SOLVER")

st.sidebar.markdown(r"### 🛠️ MANIFOLD SELECTOR")
metric_type = st.sidebar.selectbox("Spacetime Metric", [
    "Morris-Thorne Wormhole", "Kerr Black Hole", "Alcubierre Warp Drive", 
    "Reissner-Nordström (Charged)", "Schwarzschild-de Sitter (Expansion)", 
    "Schwarzschild-AdS (Contraction)", "GHS Stringy Black Hole", 
    "Vaidya (Radiating Star)", "Kerr-Newman (Charge + Rotation)", 
    "Einstein-Rosen Bridge", "JNW (Naked Singularity)", "Ellis Drainhole"
])

r0 = st.sidebar.number_input(r"Horizon/Throat ($r_0$)", 0.1, 500.0, 5.0, format="%.4f")

# Dynamic Physics Parameters
if "Kerr-Newman" in metric_type:
    param = [st.sidebar.slider("Charge (Q)", 0.0, 5.0, 1.0), st.sidebar.slider("Rotation (a)", 0.0, 5.0, 1.0)]
elif "Sitter" in metric_type or "AdS" in metric_type:
    param = st.sidebar.number_input("Lambda (Λ)", 0.0, 0.01, 0.0001, format="%.6f")
elif "Charged" in metric_type: param = st.sidebar.slider("Charge (Q)", 0.0, float(r0), 1.0)
elif "Kerr" in metric_type: param = st.sidebar.slider("Rotation (a)", 0.0, 5.0, 1.0)
elif "Warp" in metric_type: param = st.sidebar.slider("Velocity (v/c)", 0.1, 5.0, 1.0)
elif "Stringy" in metric_type: param = st.sidebar.slider("Coupling (φ)", 0.0, 4.0, 0.5)
else: param = st.sidebar.slider("Curvature/Mass Rate", 0.01, 1.0, 0.5)

lr_val = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
epochs = st.sidebar.select_slider("Epochs", options=[1000, 2500, 5000], value=2500)
pause = st.sidebar.toggle("HALT SIMULATION", value=False)

# EXECUTION
model, hist = SpacetimeSolver.solve_manifold(metric_type, r0, r0 * 10, param, epochs, lr_val)
r, b, rho, z = SpacetimeSolver.extract_telemetry(model, metric_type, r0, r0 * 10)

# METRICS
m1, m2, m3 = st.columns(3)
m1.metric("CONVERGENCE", f"{hist.loss_train[-1][0]:.2e}")
m2.metric("CLASS", metric_type.split()[0])
m3.metric("PEAK DENSITY", f"{np.max(np.abs(rho)):.4f}")

st.markdown("---")

# QUAD-QUADRANT LAYOUT
v_col, d_col = st.columns([2, 1])

with v_col:
    # 3D DUAL-SURFACE (Graph 1 & 2 Integrated)
    th = np.linspace(0, 2*np.pi, 60)
    R, T = np.meshgrid(r.flatten(), th)
    Z = np.tile(z.flatten(), (60, 1))
    
    fig = go.Figure(data=[
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=Z, colorscale='Viridis', showscale=False, opacity=0.9),
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=-Z, colorscale='Cividis', showscale=False, opacity=0.9)
    ])
    fig.update_layout(template="plotly_dark", scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # COMPACT BUTTON CIRCUIT
    b1, b2, b3 = st.columns(3)
    b1.download_button("📸 SNAPSHOT", data=io.BytesIO().getvalue(), file_name="topology.png")
    b2.download_button("📊 DATA (CSV)", data=pd.DataFrame({"r":r.flatten(),"b":b.flatten()}).to_csv().encode('utf-8'), file_name="telemetry.csv")
    b3.button("🔄 RESET KERNEL")

with d_col:
    # STACKED ANALYTICS (Graphs 3 & 4)
    # 
    fig_stack, (ax1, ax2) = plt.subplots(2, 1, facecolor='black', figsize=(6, 8))
    
    # 2D Graph 1: Density
    ax1.set_facecolor('black')
    ax1.plot(r, rho, color='#FF2E63', lw=2)
    ax1.set_title("Energy Density Profile", color='white', fontsize=10)
    ax1.tick_params(colors='white', labelsize=8); ax1.grid(alpha=0.1)
    
    # 2D Graph 2: Shape
    ax2.set_facecolor('black')
    ax2.plot(r, b, color='#00ADB5', lw=2)
    ax2.set_title("Manifold Shape Function b(r)", color='white', fontsize=10)
    ax2.tick_params(colors='white', labelsize=8); ax2.grid(alpha=0.1)
    
    plt.tight_layout()
    st.pyplot(fig_stack)

    # Reference Tabs
    st.tabs(["🔬 PHYSICS NOTES", "🛰️ TELEMETRY"])

if not pause:
    time.sleep(0.01)
    st.rerun()
