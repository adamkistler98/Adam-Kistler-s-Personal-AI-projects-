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

# --- 1. UI CONFIGURATION & PROFESSIONAL STYLING ---
st.set_page_config(
    page_title="Lorentzian Metric Solver", 
    layout="wide", 
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

# HIGH-CONTRAST "RESEARCH HUD" THEME
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #000000; }
    
    /* Headers - Neon Cyan */
    h1, h2, h3, h4 { color: #00ADB5 !important; font-family: 'Consolas', monospace; }
    p, li, label, .stMarkdown, .stCaption { color: #E0E0E0 !important; }
    
    /* Metrics - Neon Green */
    div[data-testid="stMetricValue"] { color: #00FF41 !important; font-family: 'Consolas', monospace; text-shadow: 0 0 10px rgba(0,255,65,0.3); }
    div[data-testid="stMetricLabel"] { color: #AAAAAA !important; font-weight: bold; }
    
    /* Sidebar Background */
    section[data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #222; }
    
    /* Muted Export Buttons (Dark Gray with Cyan Text) */
    div.stButton > button { 
        border: 1px solid #00ADB5; 
        color: #00ADB5; 
        background: #1A1C22; 
        width: 100%; 
        border-radius: 4px;
        font-weight: bold;
    }
    div.stButton > button:hover { 
        background: #00ADB5; 
        color: #000; 
    }
</style>
""", unsafe_allow_html=True)

# --- 2. PHYSICS ENGINE (PINN) ---
class WormholeSolver:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def train_metric(r0, r_max, curvature, iterations, lr):
        geom = dde.geometry.Interval(r0, r_max)
        def pde(r, b):
            db_dr = dde.grad.jacobian(b, r)
            return db_dr - (b / r) * curvature 
        def boundary_throat(x, on_boundary):
            return on_boundary and np.isclose(x[0], r0)
        bc = dde.icbc.DirichletBC(geom, lambda x: r0, boundary_throat)
        data = dde.data.PDE(geom, pde, bc, num_domain=300, num_boundary=50)
        net = dde.nn.FNN([1, 64, 64, 64, 1], "tanh", "Glorot normal")
        model = dde.Model(data, net)
        model.compile("adam", lr=lr)
        loss, _ = model.train(iterations=iterations, display_every=1000)
        return model, loss

    @staticmethod
    def extract_physics(model, r0, r_max, b_impact):
        r_val = np.linspace(r0, r_max, 500).reshape(-1, 1)
        r_tensor = torch.tensor(r_val, dtype=torch.float32, requires_grad=True)
        b_tensor = model.net(r_tensor)
        db_dr = torch.autograd.grad(b_tensor, r_tensor, grad_outputs=torch.ones_like(b_tensor))[0].detach().numpy()
        b = b_tensor.detach().numpy()
        
        rho = db_dr / (8 * np.pi * r_val**2 + 1e-9)
        tidal = np.abs((1 - (b / r_val)) / (r_val**2 + 1e-9))
        
        # Geodesic Light Deflection approximation
        deflection = (b_impact / (r_val + 1e-6)) * (b / r_val)
        
        # Embedding Z Coordinate
        z = np.zeros_like(r_val)
        dr = r_val[1] - r_val[0]
        for i in range(1, len(r_val)):
            val = (r_val[i] / (b[i] + 1e-6)) - 1
            z[i] = z[i-1] + (1.0 / np.sqrt(val) if val > 1e-6 else 10.0) * dr
            
        return r_val, b, rho, tidal, z, deflection

# --- 3. UI LAYOUT ---
st.title("🌌 LORENTZIAN METRIC SOLVER")

# SIDEBAR
st.sidebar.title("⏯️ Simulation State")
is_paused = st.sidebar.toggle("⏸️ Pause Simulation", value=False)

st.sidebar.markdown("## 📐 MANIFOLD GEOMETRY")
r0 = st.sidebar.slider("Throat Radius (r0)", 1.0, 10.0, 2.0)
curvature = st.sidebar.slider("Curvature Factor", 0.1, 0.9, 0.5)
b_impact = st.sidebar.slider("Photon Impact Parameter", 1.0, 20.0, 5.0)

st.sidebar.markdown("## 🧠 NEURAL HYPERPARAMETERS")
iters = st.sidebar.select_slider("Epochs", options=[1000, 2500, 5000], value=2500)
lr = st.sidebar.select_slider("Learning Rate", options=[1e-2, 1e-3], value=1e-3)

if st.sidebar.button("⚠️ Reset Topology", type="primary"):
    st.session_state.sim_result = None
    st.rerun()

# --- 4. EXECUTION ---
# Always train/load model initially
model, history = WormholeSolver.train_metric(r0, 40.0, curvature, iters, lr)
r, b, rho, tidal, z, lensing = WormholeSolver.extract_physics(model, r0, 40.0, b_impact)

# --- 5. MAIN DASHBOARD ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Convergence", f"{history.loss_train[-1][0]:.1e}")
c2.metric("NEC Violation", f"{simpson(rho.flatten(), x=r.flatten()):.3f}")
c3.metric("Max Tidal Shear", f"{np.max(tidal):.3f} g")

status_color = "#00FF41" if np.max(tidal) < 0.5 else "#FF2E63"
c4.markdown(f"<div style='text-align:center'><span style='color:#888;font-size:12px'>METRIC INTEGRITY</span><br><span style='color:{status_color};font-size:20px;font-weight:bold'>{'STABLE' if np.max(tidal) < 0.5 else 'DISRUPTED'}</span></div>", unsafe_allow_html=True)

st.markdown("---")

col_viz, col_data = st.columns([2, 1])

with col_viz:
    # 3D Plotly Surface
    theta = np.linspace(0, 2*np.pi, 50)
    R, T = np.meshgrid(r.flatten(), theta)
    Z = np.tile(z.flatten(), (50, 1))
    
    fig = go.Figure(data=[
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=Z, surfacecolor=np.tile(tidal.flatten(), (50, 1)), colorscale='Plasma', showscale=False),
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=-Z, surfacecolor=np.tile(tidal.flatten(), (50, 1)), colorscale='Plasma', showscale=False)
    ])
    fig.update_layout(template="plotly_dark", scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='cube'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig, use_container_width=True)

    # EXPORT BUTTONS (Directly below visualization)
    btn_a, btn_b = st.columns(2)
    df_export = pd.DataFrame({"Radius": r.flatten(), "Shape": b.flatten(), "Density": rho.flatten(), "Tidal": tidal.flatten()})
    
    btn_a.download_button(
        label="📸 Snapshot Manifold", 
        data=io.BytesIO().getvalue(), 
        file_name="manifold_snapshot.png", 
        use_container_width=True
    )
    btn_b.download_button(
        label="📊 Export Telemetry (CSV)", 
        data=df_export.to_csv(index=False).encode('utf-8'), 
        file_name="metric_data.csv", 
        use_container_width=True
    )

with col_data:
    tabs = st.tabs(["🔭 LIGHT DEFLECTION", "📈 TENSOR PROFILES"])
    
    with tabs[0]:
        st.subheader("Photon Deflection Analysis")
        fig_l, ax_l = plt.subplots(facecolor='black')
        ax_l.set_facecolor('black')
        ax_l.plot(r, lensing, color='#00FFF5', lw=2)
        ax_l.set_title("Light Bending Angle (α)", color='white')
        ax_l.set_xlabel("Radius r", color='#888')
        ax_l.tick_params(colors='#888')
        ax_l.grid(color='#222')
        st.pyplot(fig_l)
        st.caption("Visualizing the deflection of null-geodesics passing near the throat.")

    with tabs[1]:
        st.subheader("Metric Distribution")
        fig_t, ax_t = plt.subplots(2, 1, facecolor='black', figsize=(6, 8))
        ax_t[0].plot(r, b, color='#00ADB5'); ax_t[0].set_title("Shape Function b(r)", color='white')
        ax_t[1].plot(r, rho, color='#FF2E63'); ax_t[1].set_title("Energy Density ρ", color='white')
        for a in ax_t: 
            a.set_facecolor('black')
            a.tick_params(colors='#888')
        plt.tight_layout()
        st.pyplot(fig_t)

# --- 6. AUTO-LOOP LOGIC ---
if not is_paused:
    time.sleep(0.01)
    st.rerun()
