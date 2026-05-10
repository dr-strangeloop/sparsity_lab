"""
Sparse vs Standard Neural Network — Interactive Research Dashboard
Runs REAL PyTorch training via experiment.py OR loads existing results JSON.
Simulation is the fallback when no real data exists yet.

Run: streamlit run sparsity_app.py
Requires: experiment.py in the same directory
"""

import json
import csv
import io
import subprocess
import sys
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Optional

st.set_page_config(page_title="Sparsity Lab", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #0c0c0f; color: #e0e0e0; }
section[data-testid="stSidebar"] { background: #111116; border-right: 1px solid #222230; }
section[data-testid="stSidebar"] * { color: #c0c0d0 !important; }
.hero { padding: 2rem 0 0.5rem 0; border-bottom: 1px solid #222230; margin-bottom: 1.5rem; }
.hero-title { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 600; color: #f0f0f0; letter-spacing: -0.02em; margin: 0; }
.hero-sub { font-size: 0.9rem; color: #666688; margin: 0.25rem 0 0 0; font-family: 'IBM Plex Mono', monospace; }
.data-badge { display: inline-block; padding: 2px 10px; border-radius: 3px; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; margin-left: 12px; vertical-align: middle; }
.badge-real { background: #44dd8822; color: #44dd88; border: 1px solid #44dd8844; }
.badge-sim  { background: #ffaa3322; color: #ffaa33; border: 1px solid #ffaa3344; }
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 1rem 0 1.5rem 0; }
.metric-card { background: #111116; border: 1px solid #222230; border-radius: 6px; padding: 1rem 1.1rem; }
.metric-card.highlight { border-color: #3a5aff44; background: #111122; }
.metric-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #555577; text-transform: uppercase; letter-spacing: 0.1em; margin: 0 0 4px 0; }
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.7rem; font-weight: 600; color: #e8e8f0; margin: 0; line-height: 1; }
.metric-value.good { color: #44dd88; }
.metric-value.warn { color: #ffaa33; }
.metric-value.bad  { color: #ff5566; }
.metric-delta { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #555577; margin: 4px 0 0 0; }
.verdict-box { background: #111116; border: 1px solid #222230; border-left: 3px solid #3a5aff; border-radius: 6px; padding: 1rem 1.25rem; font-size: 0.88rem; color: #9090b0; line-height: 1.7; margin: 1rem 0; font-family: 'IBM Plex Mono', monospace; }
.verdict-box strong { color: #d0d0f0; font-weight: 500; }
.tag { display: inline-block; padding: 1px 8px; border-radius: 3px; font-size: 0.72rem; margin-right: 6px; }
.tag-green  { background: #44dd8822; color: #44dd88; border: 1px solid #44dd8844; }
.tag-yellow { background: #ffaa3322; color: #ffaa33; border: 1px solid #ffaa3344; }
.tag-red    { background: #ff556622; color: #ff5566; border: 1px solid #ff556644; }
.section-head { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #444466; text-transform: uppercase; letter-spacing: 0.1em; margin: 1.5rem 0 0.75rem 0; padding-bottom: 6px; border-bottom: 1px solid #1a1a24; }
.log-box { background: #080810; border: 1px solid #1a1a30; border-radius: 6px; padding: 1rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: #6688aa; max-height: 280px; overflow-y: auto; line-height: 1.6; white-space: pre-wrap; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
RESULTS_FILE = Path("results/experiment_results.json")
STD_COL, SP_COL = "#4488ff", "#ff4477"
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#9090b0", size=11),
    margin=dict(l=50, r=20, t=30, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#222230", borderwidth=1, font=dict(size=11)),
    xaxis=dict(gridcolor="#1a1a24", zeroline=False, linecolor="#222230"),
    yaxis=dict(gridcolor="#1a1a24", zeroline=False, linecolor="#222230"),
)

# ── Data helpers ───────────────────────────────────────────────────────────────

def load_real_results() -> Optional[dict]:
    if not RESULTS_FILE.exists():
        return None
    try:
        return json.loads(RESULTS_FILE.read_text())
    except Exception:
        return None

def parse_real_results(raw: dict, sp_key: str) -> Optional[dict]:
    if "standard" not in raw or sp_key not in raw:
        return None
    std_data, sp_data = raw["standard"], raw[sp_key]

    def ex(data, key): return [row[key] for row in data]

    epochs     = ex(std_data, "epoch")
    std_acc    = ex(std_data, "val_acc")
    sp_acc     = ex(sp_data,  "val_acc")
    std_loss   = ex(std_data, "val_loss")
    sp_loss    = ex(sp_data,  "val_loss")
    std_flops  = std_data[0]["flops"]
    sp_flops   = sp_data[0]["flops"]

    std_energy_cum = list(np.cumsum(ex(std_data, "energy_uj")))
    sp_energy_cum  = list(np.cumsum(ex(sp_data,  "energy_uj")))

    def ep_to(t, c):
        arr = np.array(c); idx = np.argmax(arr >= t)
        return int(idx+1) if arr[idx] >= t else None

    sp_val = float(sp_data[0].get("actual_sparsity", 0.3))
    return {
        "epochs": epochs, "std_acc": std_acc, "sp_acc": sp_acc,
        "std_loss": std_loss, "sp_loss": sp_loss,
        "std_energy": std_energy_cum, "sp_energy": sp_energy_cum,
        "std_final": round(std_acc[-1], 5), "sp_final": round(sp_acc[-1], 5),
        "std_flops": int(std_flops), "sp_flops": int(sp_flops), "n_params": 0,
        "energy_saved_pct": round((1-sp_val)*100, 1),
        "acc_delta": round((sp_acc[-1] - std_acc[-1])*100, 3),
        "epochs_to_90_std": ep_to(0.90, std_acc), "epochs_to_90_sp": ep_to(0.90, sp_acc),
        "epochs_to_95_std": ep_to(0.95, std_acc), "epochs_to_95_sp": ep_to(0.95, sp_acc),
        "is_real": True,
    }

def sigmoid(x, speed, mid): return 1/(1+np.exp(-speed*(x-mid)))

def simulate(sparsity, epochs, hidden_dims, lr, dropout, batch_size, seed):
    rng = np.random.default_rng(seed)
    n_params = sum(hidden_dims[i]*hidden_dims[i+1] for i in range(len(hidden_dims)-1))
    std_final = min(0.985, max(0.88, 0.91+0.075*np.tanh(lr*5000)+0.005*np.log(batch_size/32+1)-dropout*0.04))
    std_speed = 0.15+lr*80
    sp_final  = max(0.78, std_final - 0.022*np.power(1-sparsity,1.8))
    sp_speed  = std_speed*(0.65+sparsity*0.35)
    ep = np.arange(1, epochs+1)
    ns, np_ = rng.normal(0,0.003,epochs), rng.normal(0,0.004,epochs)
    std_acc = np.flip(np.maximum.accumulate(np.flip(np.clip(sigmoid(ep,std_speed,epochs*0.30)*std_final+ns,0,1))))
    sp_acc  = np.flip(np.maximum.accumulate(np.flip(np.clip(sigmoid(ep,sp_speed, epochs*0.38)*sp_final +np_,0,1))))
    std_loss = np.clip(2.3*np.exp(-std_speed*0.6*ep/epochs)+rng.normal(0,0.01,epochs),0.05,3)
    sp_loss  = np.clip(2.3*np.exp(-sp_speed *0.5*ep/epochs)+rng.normal(0,0.012,epochs),0.05,3)
    std_flops = sum(2*hidden_dims[i]*hidden_dims[i+1] for i in range(len(hidden_dims)-1))
    std_flops += 2*784*hidden_dims[0]+2*hidden_dims[-1]*10
    sp_flops = int(std_flops*sparsity)
    JPF = 1e-12; spe = 60000
    def ep_to(t,c): idx=np.argmax(c>=t); return int(idx+1) if c[idx]>=t else None
    return {
        "epochs": list(ep.astype(int)),
        "std_acc":  list(np.round(std_acc,5)), "sp_acc":  list(np.round(sp_acc,5)),
        "std_loss": list(np.round(std_loss,5)),"sp_loss": list(np.round(sp_loss,5)),
        "std_energy": [round(std_flops*spe*JPF*1e6*(i+1),2) for i in range(epochs)],
        "sp_energy":  [round(sp_flops *spe*JPF*1e6*(i+1),2) for i in range(epochs)],
        "std_final": round(float(std_acc[-1]),5), "sp_final": round(float(sp_acc[-1]),5),
        "std_flops": int(std_flops), "sp_flops": int(sp_flops), "n_params": int(n_params),
        "energy_saved_pct": round((1-sparsity)*100,1),
        "acc_delta": round((float(sp_acc[-1])-float(std_acc[-1]))*100,3),
        "epochs_to_90_std": ep_to(0.90,std_acc), "epochs_to_90_sp": ep_to(0.90,sp_acc),
        "epochs_to_95_std": ep_to(0.95,std_acc), "epochs_to_95_sp": ep_to(0.95,sp_acc),
        "is_real": False,
    }

# ── Charts ─────────────────────────────────────────────────────────────────────

def acc_chart(d, show_loss, sparsity, is_real):
    tag = "✓ real" if is_real else "~ simulated"
    cols = 2 if show_loss else 1
    titles = ([f"Validation accuracy ({tag})", f"Training loss ({tag})"] if show_loss
              else [f"Validation accuracy ({tag})"])
    fig = make_subplots(rows=1, cols=cols, subplot_titles=titles)
    ep = d["epochs"]
    fig.add_trace(go.Scatter(x=ep, y=[v*100 for v in d["std_acc"]], name="Standard NN",
        line=dict(color=STD_COL, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep, y=[v*100 for v in d["sp_acc"]],
        name=f"Sparse NN ({round(sparsity*100)}%)",
        line=dict(color=SP_COL, width=2, dash="dot")), row=1, col=1)
    fig.add_hline(y=90, line_dash="dash", line_color="rgba(255,255,255,0.13)",
                  annotation_text="90%", annotation_font_color="#555577", row=1, col=1)
    fig.add_hline(y=95, line_dash="dash", line_color="rgba(255,255,255,0.07)",
                  annotation_text="95%", annotation_font_color="#333355", row=1, col=1)
    if show_loss:
        fig.add_trace(go.Scatter(x=ep, y=d["std_loss"], name="Std loss",
            line=dict(color=STD_COL, width=2), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=ep, y=d["sp_loss"], name="Sparse loss",
            line=dict(color=SP_COL, width=2, dash="dot"), showlegend=False), row=1, col=2)
    fig.update_layout(**PLOT_LAYOUT, height=320)
    fig.update_yaxes(title_text="Accuracy (%)", ticksuffix="%", row=1, col=1)
    fig.update_xaxes(title_text="Epoch")
    if show_loss: fig.update_yaxes(title_text="Loss", row=1, col=2)
    return fig

def energy_chart(d, sparsity):
    ep = d["epochs"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=d["std_energy"], name="Standard NN",
        fill="tozeroy", line=dict(color=STD_COL, width=1.5), fillcolor="rgba(68,136,255,0.12)"))
    fig.add_trace(go.Scatter(x=ep, y=d["sp_energy"],
        name=f"Sparse NN ({round(sparsity*100)}%)", fill="tozeroy",
        line=dict(color=SP_COL, width=1.5, dash="dot"), fillcolor="rgba(255,68,119,0.12)"))
    fig.update_layout(**PLOT_LAYOUT, height=280,
        yaxis_title="Cumulative energy (μJ)", xaxis_title="Epoch")
    return fig

def convergence_chart(d, sparsity):
    cats = ["Standard NN", f"Sparse NN ({round(sparsity*100)}%)"]
    to90 = [d["epochs_to_90_std"] or d["epochs"][-1], d["epochs_to_90_sp"] or d["epochs"][-1]]
    to95 = [d["epochs_to_95_std"] or d["epochs"][-1], d["epochs_to_95_sp"] or d["epochs"][-1]]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Epochs to 90%", x=cats, y=to90,
        marker_color=['rgba(68,136,255,0.6)', 'rgba(255,68,119,0.6)'],
        marker_line_color=[STD_COL, SP_COL], marker_line_width=1))
    fig.add_trace(go.Bar(name="Epochs to 95%", x=cats, y=to95,
        marker_color=['rgba(68,136,255,0.25)', 'rgba(255,68,119,0.25)'],
        marker_line_color=[STD_COL, SP_COL], marker_line_width=1,
        marker_pattern_shape=["x","x"]))
    fig.update_layout(**PLOT_LAYOUT, height=280, barmode="group", yaxis_title="Epochs required")
    return fig

def flops_chart(d, sparsity):
    cats = ["Standard NN", f"Sparse NN ({round(sparsity*100)}%)"]
    fig = go.Figure(go.Bar(x=cats, y=[d["std_flops"], d["sp_flops"]],
        marker_color=['rgba(68,136,255,0.6)', 'rgba(255,68,119,0.6)'],
        marker_line_color=[STD_COL, SP_COL], marker_line_width=1))
    fig.update_layout(**PLOT_LAYOUT, height=260, yaxis_title="FLOPs per forward pass", showlegend=False)
    return fig

def multi_acc_chart(raw: dict):
    names, accs = [], []
    for key, data in raw.items():
        if not data: continue
        label = "100% (standard)" if key == "standard" else key.replace("sparse_","").replace("pct","%")
        names.append(label)
        accs.append(round(data[-1]["val_acc"]*100, 3))
    colors = [STD_COL if "standard" in n or "100" in n else SP_COL for n in names]
    fig = go.Figure(go.Bar(x=names, y=accs, marker_color=[c+"99" for c in colors],
        marker_line_color=colors, marker_line_width=1))
    fig.update_layout(**PLOT_LAYOUT, height=260,
        yaxis_title="Final val accuracy (%)", yaxis_ticksuffix="%",
        xaxis_title="Sparsity level", showlegend=False)
    return fig

def verdict(acc_delta, energy_saved, sparsity):
    loss = abs(acc_delta)
    if energy_saved > 55 and loss < 0.5:
        tag = '<span class="tag tag-green">WORTH IT</span>'
        msg = (f"<strong>{energy_saved:.0f}% energy reduction</strong> with only "
               f"<strong>{loss:.3f}% accuracy loss</strong>. Strong candidate for edge/neuromorphic deployment.")
    elif energy_saved > 35 and loss < 1.5:
        tag = '<span class="tag tag-yellow">MARGINAL</span>'
        msg = (f"<strong>{energy_saved:.0f}% energy savings</strong> but <strong>{loss:.3f}% accuracy penalty</strong>. "
               f"Acceptable for latency-critical apps. Try tuning LR or dropout.")
    else:
        tag = '<span class="tag tag-red">NOT WORTH IT</span>'
        msg = (f"Only <strong>{energy_saved:.0f}% energy saved</strong> with <strong>{loss:.3f}% accuracy drop</strong>. "
               f"Adjust sparsity level or architecture.")
    return tag, msg

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚡ Experiment settings")
    st.markdown("---")
    mode = st.radio("Data source", ["Simulation (instant)", "Real training (PyTorch)", "Load saved results"],
        help="Simulation = estimates. Real = actual PyTorch MNIST. Load = use existing results JSON.")
    st.markdown("---")
    st.markdown("**Sparsity**")
    sparsity = st.slider("Active neurons (%)", 10, 100, 30, 5, format="%d%%") / 100.0
    if mode == "Real training (PyTorch)":
        st.markdown("**Also train at these levels**")
        train_std = st.checkbox("Include standard NN (100%)", value=True)
        extra = st.multiselect("Additional sparsity levels", [10,20,30,40,50,60,70,80,90], default=[])
    st.markdown("---")
    st.markdown("**Architecture**")
    n_layers = st.selectbox("Hidden layers", [2,3,4], index=1)
    hidden_dims_raw = [int(st.number_input(f"Layer {i+1} neurons", 32, 1024,
        [256,256,128,64][i], 32)) for i in range(n_layers)]
    hidden_dims = [784] + hidden_dims_raw + [10]
    st.markdown("---")
    st.markdown("**Training**")
    epochs     = st.slider("Epochs", 5, 200, 50, 5)
    lr         = st.select_slider("Learning rate", [0.0001,0.0005,0.001,0.005,0.01], 0.001,
                     format_func=lambda x: f"{x:.4f}")
    batch_size = st.select_slider("Batch size", [16,32,64,128,256,512], 128)
    dropout    = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
    seed       = st.number_input("Random seed", 0, 9999, 42)
    st.markdown("---")
    show_loss = st.toggle("Show loss curves", value=True)
    run = st.button("▶ Run experiment", type="primary", use_container_width=True)

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <p class="hero-title">⚡ Sparsity Lab</p>
  <p class="hero-sub">// sparse vs standard neural networks — efficiency research dashboard</p>
</div>
""", unsafe_allow_html=True)

# ── Routing ────────────────────────────────────────────────────────────────────

data_badge = ""

if mode == "Load saved results":
    raw = load_real_results()
    if raw is None:
        st.error(f"No results found at `{RESULTS_FILE}`.")
        st.info("Run `python experiment.py --epochs 50 --sparsity 1.0 0.5 0.3 0.2 0.1` first, then reload.")
        st.stop()
    available = [k for k in raw if k != "standard"]
    if not available:
        st.error("Results JSON has no sparse model keys. Check your experiment output."); st.stop()
    sp_key = st.selectbox("Compare against standard:", available,
        format_func=lambda k: k.replace("sparse_","").replace("pct","% active"))
    d = parse_real_results(raw, sp_key)
    if d is None:
        st.error(f"Could not parse '{sp_key}' from results JSON."); st.stop()
    sparsity = 1.0 - d["energy_saved_pct"]/100
    data_badge = '<span class="data-badge badge-real">✓ REAL DATA</span>'
    if len(raw) > 2:
        st.markdown('<p class="section-head">All sparsity levels — final accuracy</p>', unsafe_allow_html=True)
        st.plotly_chart(multi_acc_chart(raw), use_container_width=True)

elif mode == "Real training (PyTorch)" and run:
    if not Path("experiment.py").exists():
        st.error("experiment.py not found. Put it in the same directory as this app."); st.stop()
    sp_levels = sorted(set([1.0] + [round(sparsity,2)] + [e/100 for e in extra]
                           if train_std else [round(sparsity,2)] + [e/100 for e in extra]), reverse=True)
    st.info(f"🚀 Real PyTorch training on MNIST — levels: {[f'{int(s*100)}%' for s in sp_levels]}, {epochs} epochs each. Grab a coffee ☕")
    log_area = st.empty()
    prog = st.progress(0)
    stat = st.empty()
    cmd  = [sys.executable, "experiment.py", "--epochs", str(epochs),
            "--sparsity", *[str(s) for s in sp_levels]]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    logs, n = [], 0
    total = epochs * len(sp_levels)
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None: break
        if line:
            line = line.rstrip(); logs.append(line)
            if "Epoch" in line and "Val" in line:
                n += 1; prog.progress(min(n/total, 1.0)); stat.markdown(f"`{line}`")
            if len(logs) > 60: logs = logs[-60:]
            log_area.markdown(f'<div class="log-box">{"<br>".join(logs)}</div>', unsafe_allow_html=True)
    if proc.returncode == 0:
        st.success("✓ Training complete!")
        raw = load_real_results()
        sp_key = f"sparse_{int(sparsity*100)}pct"
        d = parse_real_results(raw, sp_key) if raw and sp_key in raw else None
        if d:
            data_badge = '<span class="data-badge badge-real">✓ REAL DATA</span>'
        else:
            st.warning("Could not parse results — falling back to simulation.")
            d = simulate(sparsity, epochs, hidden_dims, lr, dropout, batch_size, seed)
            data_badge = '<span class="data-badge badge-sim">~ SIMULATED</span>'
    else:
        st.error("Training failed. Is PyTorch installed? Is experiment.py in this folder?")
        d = simulate(sparsity, epochs, hidden_dims, lr, dropout, batch_size, seed)
        data_badge = '<span class="data-badge badge-sim">~ SIMULATED</span>'

else:
    d        = simulate(sparsity, epochs, hidden_dims, lr, dropout, batch_size, seed)
    data_badge = '<span class="data-badge badge-sim">~ SIMULATED</span>'

# ── Metrics ────────────────────────────────────────────────────────────────────

acc_delta    = d["acc_delta"]
energy_saved = d["energy_saved_pct"]
flops_saved  = round((1-sparsity)*100, 1)
is_real      = d.get("is_real", False)
acc_cls = "good" if abs(acc_delta) < 0.5 else ("warn" if abs(acc_delta) < 2 else "bad")
en_cls  = "good" if energy_saved > 50 else ("warn" if energy_saved > 25 else "bad")

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card">
    <p class="metric-label">Standard final acc {data_badge}</p>
    <p class="metric-value">{d['std_final']*100:.2f}%</p>
    <p class="metric-delta">dense baseline</p>
  </div>
  <div class="metric-card">
    <p class="metric-label">Sparse final acc</p>
    <p class="metric-value {acc_cls}">{d['sp_final']*100:.2f}%</p>
    <p class="metric-delta">Δ {acc_delta:+.3f}% vs standard</p>
  </div>
  <div class="metric-card highlight">
    <p class="metric-label">Energy saved</p>
    <p class="metric-value {en_cls}">{energy_saved:.0f}%</p>
    <p class="metric-delta">{round(sparsity*100)}% neurons active</p>
  </div>
  <div class="metric-card">
    <p class="metric-label">FLOPs reduction</p>
    <p class="metric-value">{flops_saved:.0f}%</p>
    <p class="metric-delta">{d['sp_flops']:,} vs {d['std_flops']:,}</p>
  </div>
</div>
""", unsafe_allow_html=True)

tag, msg = verdict(acc_delta, energy_saved, sparsity)
conv_note = ""
if d['epochs_to_90_sp'] and d['epochs_to_90_std']:
    diff = d['epochs_to_90_sp'] - d['epochs_to_90_std']
    conv_note = f" Sparse needs <strong>{abs(diff)} {'more' if diff>0 else 'fewer'} epochs</strong> to reach 90%."

st.markdown(f"""
<div class="verdict-box">
  {tag} <strong>Hypothesis verdict at {round(sparsity*100)}% active neurons:</strong><br>
  {msg}{conv_note}
</div>
""", unsafe_allow_html=True)

# ── Charts ─────────────────────────────────────────────────────────────────────

st.markdown('<p class="section-head">Accuracy & Loss</p>', unsafe_allow_html=True)
st.plotly_chart(acc_chart(d, show_loss, sparsity, is_real), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown('<p class="section-head">Cumulative energy usage</p>', unsafe_allow_html=True)
    st.plotly_chart(energy_chart(d, sparsity), use_container_width=True)
with c2:
    st.markdown('<p class="section-head">Convergence speed</p>', unsafe_allow_html=True)
    st.plotly_chart(convergence_chart(d, sparsity), use_container_width=True)

st.markdown('<p class="section-head">FLOPs per forward pass</p>', unsafe_allow_html=True)
st.plotly_chart(flops_chart(d, sparsity), use_container_width=True)

# ── Export ─────────────────────────────────────────────────────────────────────

st.markdown('<p class="section-head">Export</p>', unsafe_allow_html=True)
ca, cb, cc = st.columns(3)

with ca:
    payload = {"config": {"sparsity": sparsity, "epochs": epochs, "hidden_dims": hidden_dims_raw,
        "lr": lr, "batch_size": batch_size, "dropout": dropout, "seed": seed, "is_real": is_real},
        "results": d}
    st.download_button("⬇ Download JSON", json.dumps(payload, indent=2),
        f"sparsity_{round(sparsity*100)}pct.json", "application/json", use_container_width=True)

with cb:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["epoch","std_acc_%","sp_acc_%","std_loss","sp_loss","std_energy_uj","sp_energy_uj"])
    for i, ep in enumerate(d["epochs"]):
        w.writerow([ep, round(d["std_acc"][i]*100,3), round(d["sp_acc"][i]*100,3),
            round(d["std_loss"][i],5), round(d["sp_loss"][i],5),
            round(d["std_energy"][i],3), round(d["sp_energy"][i],3)])
    st.download_button("⬇ Download CSV", buf.getvalue(),
        f"sparsity_{round(sparsity*100)}pct.csv", "text/csv", use_container_width=True)

with cc:
    src = "REAL PyTorch/MNIST" if is_real else "Simulation"
    summary = f"""SPARSITY EXPERIMENT SUMMARY
============================
Data source    : {src}
Active neurons : {round(sparsity*100)}%
Epochs         : {epochs}
Hidden dims    : {hidden_dims_raw}
LR / Batch     : {lr} / {batch_size}
Dropout / Seed : {dropout} / {seed}

Standard final acc   : {d['std_final']*100:.3f}%
Sparse final acc     : {d['sp_final']*100:.3f}%
Accuracy delta       : {acc_delta:+.3f}%
Energy saved         : {energy_saved:.1f}%
FLOPs reduction      : {flops_saved:.1f}%
Std epochs to 90%    : {d['epochs_to_90_std'] or 'N/A'}
Sparse epochs to 90% : {d['epochs_to_90_sp']  or 'N/A'}
Std FLOPs/pass       : {d['std_flops']:,}
Sparse FLOPs/pass    : {d['sp_flops']:,}
"""
    st.download_button("⬇ Download TXT", summary,
        f"sparsity_{round(sparsity*100)}pct_summary.txt", "text/plain", use_container_width=True)

st.markdown("""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #1a1a24;
     font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#333355;text-align:center;">
  sparsity lab // real training via experiment.py + MNIST
</div>
""", unsafe_allow_html=True)