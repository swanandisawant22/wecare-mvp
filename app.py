import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import streamlit.components.v1 as components
from datetime import datetime
import time

st.set_page_config(
    page_title="WeCare | Intelligent Safety Platform",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "model/wecare_activity_model.pkl"
FEATURE_PATH = "model/wecare_feature_columns.pkl"
DATA_PATH = "data/test.csv"

# -------------------- MODEL + DATA --------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(FEATURE_PATH)

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

try:
    model, feature_columns = load_model()
except Exception:
    model, feature_columns = None, []

try:
    dataset = load_dataset()
except Exception:
    dataset = None

def create_features(df, columns):
    features = {}
    for sensor_type in ["amp", "phase"]:
        for i in range(30):
            col = f"{sensor_type}_sc{i}"
            if col in df.columns:
                v = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(v) == 0:
                    v = pd.Series([0.0])
                features[f"{col}_mean"] = v.mean()
                features[f"{col}_std"] = v.std()
                features[f"{col}_min"] = v.min()
                features[f"{col}_max"] = v.max()
    out = pd.DataFrame([features])
    for col in columns:
        if col not in out.columns:
            out[col] = 0
    return out[list(columns)]

def interpret(activity):
    a = str(activity).lower()
    if a in ("walking", "standing", "sit_down"):
        return "NORMAL", "Normal movement pattern", "#047857"
    if a == "static":
        return "MONITOR", "Low-movement state detected", "#B45309"
    return "MONITOR", "Review activity", "#B45309"

# -------------------- STATE --------------------
defaults = {
    "page": "Overview",
    "live_mode": False,
    "live_sample_index": 0,
    "activity_history": [],
    "last_marker": None,
    "active_person": "Person 1",
    "twin_view": "Front",
    "twin_layer": "Sensors",
    "auto_rotate": False,
    "selected_region": "Right Hip",
    "pain_points": {},
    "alerts": [
        {"time":"15:34:10","event":"Fall Risk Detected","zone":"Bedroom","risk":"High","status":"Open"},
        {"time":"14:22:31","event":"Prolonged Inactivity","zone":"Living Room","risk":"Moderate","status":"Open"},
        {"time":"11:05:12","event":"Unusual Movement","zone":"Kitchen","risk":"Low","status":"Resolved"},
    ],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def live_state():
    if dataset is None or model is None or not feature_columns or "sample_id" not in dataset.columns:
        return None
    ids = dataset["sample_id"].unique()
    idx = st.session_state.live_sample_index % len(ids)
    sid = ids[idx]
    sample = dataset[dataset["sample_id"] == sid].copy()
    X = create_features(sample, feature_columns)
    pred = str(model.predict(X)[0])
    probs = model.predict_proba(X)[0]
    conf = float(np.max(probs) * 100)
    status, message, color = interpret(pred)
    amp_cols = [c for c in sample.columns if c.startswith("amp_sc")]
    if amp_cols:
        raw = pd.to_numeric(sample[amp_cols[0]], errors="coerce").dropna().to_numpy()
        raw = raw[-64:] if len(raw) else np.zeros(64)
        raw = raw - np.nanmean(raw)
        sd = np.nanstd(raw) or 1
        wave = raw / sd
    else:
        wave = np.sin(np.linspace(0, 6*np.pi, 64))
    return {"sample_id":sid,"prediction":pred,"confidence":conf,"status":status,
            "message":message,"color":color,"wave":wave,"index":idx,"total":len(ids)}

def record_event(s):
    if not s:
        return
    marker = (int(s["sample_id"]), s["prediction"])
    if st.session_state.last_marker == marker:
        return
    st.session_state.activity_history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Sample": int(s["sample_id"]),
        "Activity": s["prediction"].replace("_"," ").title(),
        "Confidence": f'{s["confidence"]:.1f}%',
        "Status": s["status"],
    })
    st.session_state.activity_history = st.session_state.activity_history[-30:]
    st.session_state.last_marker = marker

state = live_state()
if st.session_state.live_mode and state:
    record_event(state)

# -------------------- STYLE --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:Inter,sans-serif;}
.stApp {background:#F8FAFC;color:#0F172A;}
#MainMenu, footer {visibility:hidden;}
/* Keep Streamlit's header alive: the native sidebar collapse/reopen control lives here. */
header[data-testid="stHeader"] {
  visibility: visible !important;
  display: flex !important;
  background: rgba(248,250,252,0) !important;
  height: 3rem !important;
  min-height: 3rem !important;
}
/* Hide only nonessential chrome. Do NOT hide stToolbar itself. */
[data-testid="stDecoration"] {display:none !important;}
[data-testid="stAppDeployButton"] {display:none !important;}
.block-container {padding-top:.35rem;padding-bottom:2.5rem;max-width:1500px;}
/* FINAL SIDEBAR FIX — navigation is permanently expanded. */
section[data-testid="stSidebar"] {
  background:#FFFFFF !important;
  border-right:1px solid #CBD5E1 !important;
  display:block !important;
  visibility:visible !important;
  transform:none !important;
  min-width:330px !important;
  width:330px !important;
  max-width:330px !important;
  left:0 !important;
}
section[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
  display:block !important;
  visibility:visible !important;
  width:330px !important;
}
/* No fragile native arrow: WeCare navigation remains visible. */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
  display:none !important;
}
section[data-testid="stSidebar"] * {color:#0F172A;}
[data-testid="stSidebar"] .stButton button {text-align:left;justify-content:flex-start;}
h1,h2,h3 {color:#0F172A!important;}
p, label, .stCaption {color:#64748B;}
.stButton>button {
  background:#FFFFFF;color:#0F172A;border:1px solid #CBD5E1;border-radius:10px;
  min-height:42px;font-weight:600;box-shadow:none;
}
.stButton>button:hover {border-color:#1D4ED8;color:#1D4ED8;background:#EFF6FF;}
div[data-testid="stMetric"] {
  background:#FFFFFF;border:1px solid #CBD5E1;border-radius:14px;padding:16px;
}
div[data-testid="stMetric"] label {color:#64748B!important;}
div[data-testid="stMetricValue"] {color:#0F172A!important;}
div[data-testid="stDataFrame"] {border:1px solid #CBD5E1;border-radius:12px;overflow:hidden;}
.stTabs [data-baseweb="tab-list"] {gap:8px;}
.stTabs [data-baseweb="tab"] {background:#FFFFFF;border:1px solid #CBD5E1;border-radius:9px;padding:8px 14px;}
.stTabs [aria-selected="true"] {color:#1D4ED8!important;border-color:#1D4ED8!important;}
.stProgress > div > div > div > div {background:#1D4ED8;}
.wc-logo {display:flex;align-items:center;gap:11px;margin:4px 0 16px;}
.logo-mark {width:52px;height:52px;position:relative;flex:0 0 52px;}
.logo-mark .roof {position:absolute;left:10px;top:17px;width:30px;height:30px;border:3px solid #0F766E;border-top:0;border-radius:3px;}
.logo-mark .roof:before,.logo-mark .roof:after {content:"";position:absolute;top:-10px;width:25px;height:3px;background:#0F766E;border-radius:3px;}
.logo-mark .roof:before {left:-5px;transform:rotate(-38deg);}
.logo-mark .roof:after {right:-5px;transform:rotate(38deg);}
.logo-mark .heart {position:absolute;left:17px;top:25px;color:#1D4ED8;font-size:20px;line-height:1;}
.logo-mark .wifi1,.logo-mark .wifi2 {position:absolute;border:3px solid #0F766E;border-left-color:transparent;border-right-color:transparent;border-bottom-color:transparent;border-radius:50%;}
.logo-mark .wifi1 {width:38px;height:20px;left:7px;top:1px;}
.logo-mark .wifi2 {width:24px;height:13px;left:14px;top:8px;}
.logo-text {font-size:22px;font-weight:800;color:#0F172A;line-height:1;}
.logo-sub {font-size:9px;color:#64748B;letter-spacing:1px;margin-top:5px;}
.badge {display:inline-flex;align-items:center;gap:7px;border-radius:999px;padding:7px 11px;font-size:11px;font-weight:700;}
.ok {background:#ECFDF5;color:#047857;border:1px solid #A7F3D0;}
.warn {background:#FFFBEB;color:#B45309;border:1px solid #FDE68A;}
.risk {background:#FEF2F2;color:#B91C1C;border:1px solid #FECACA;}
.blue {background:#EFF6FF;color:#1D4ED8;border:1px solid #BFDBFE;}
.card {background:#FFFFFF;border:1px solid #CBD5E1;border-radius:16px;padding:18px;margin-bottom:14px;}
.card-title {font-size:15px;font-weight:800;color:#0F172A;margin-bottom:7px;}
.muted {color:#64748B;font-size:12px;}
.big {font-size:29px;font-weight:800;color:#0F172A;}
.page-title {font-size:29px;font-weight:800;color:#0F172A;margin-bottom:3px;}
.page-sub {color:#64748B;font-size:13px;margin-bottom:18px;}
.alert-high {background:#FEF2F2;border:1px solid #FECACA;border-left:4px solid #B91C1C;border-radius:12px;padding:14px;}
.alert-warn {background:#FFFBEB;border:1px solid #FDE68A;border-left:4px solid #B45309;border-radius:12px;padding:14px;}
.system-strip {background:#FFFFFF;border:1px solid #CBD5E1;border-radius:14px;padding:12px 16px;font-size:11px;color:#64748B;}
</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("""<div class="wc-logo"><div class="logo-mark"><span class="wifi1"></span><span class="wifi2"></span><span class="roof"></span><span class="heart">♥</span></div><div>
    <div class="logo-text">We<span style="color:#0F766E">Care</span></div><div class="logo-sub">INTELLIGENT SAFETY PLATFORM</div>
    </div></div>""", unsafe_allow_html=True)
    st.markdown('<span class="badge ok">● SYSTEM ONLINE</span>', unsafe_allow_html=True)
    st.write("")
    st.session_state.live_mode = st.toggle("Live Monitoring", value=st.session_state.live_mode)
    if st.session_state.live_mode:
        st.success("AUTO • CSI stream active")
    else:
        st.caption("Manual CSI processing")

    nav = [
        ("⌂","Overview"),("◉","Live Sensing"),("♙","Digital Human Twin"),
        ("♡","Sleep & Vitals"),("⚡","Mobility & Fall Risk"),("◈","Biometrics"),
        ("◎","Attribution"),("▣","Safety Log"),("☷","Simple Summary"),("▤","Reports"),("⚙","Settings")
    ]
    st.write("")
    for icon, name in nav:
        if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True):
            st.session_state.page = name
            st.rerun()
    st.divider()
    st.caption("WC-PUNE-01 • CSI NODE")
    st.caption("Prototype telemetry environment")

def header(title, subtitle):
    a,b = st.columns([7,2])
    with a:
        st.markdown(f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div style="text-align:right"><span class="badge ok">● LIVE CONNECTION</span></div>', unsafe_allow_html=True)

def activity_card(s):
    if not s:
        st.warning("CSI model/data unavailable.")
        return
    st.markdown(f"""<div class="card"><div class="card-title">Current AI Activity</div>
    <div class="big">{s["prediction"].replace("_"," ").title()}</div>
    <div class="muted">{s["confidence"]:.1f}% confidence • Sample {s["sample_id"]}</div><br>
    <span class="badge {'ok' if s['status']=='NORMAL' else 'warn'}">{s["status"]}</span></div>""", unsafe_allow_html=True)

def wave_chart(s, height=230):
    y = s["wave"] if s else np.sin(np.linspace(0,8*np.pi,64))
    fig = go.Figure(go.Scatter(y=y, mode="lines", line=dict(color="#0E7490", width=2)))
    fig.update_layout(height=height, margin=dict(l=5,r=5,t=5,b=5), paper_bgcolor="#FFFFFF",
                      plot_bgcolor="#FFFFFF", showlegend=False, xaxis=dict(showgrid=False, zeroline=False),
                      yaxis=dict(gridcolor="#E2E8F0", zeroline=False))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# -------------------- PAGES --------------------
page = st.session_state.page

if page == "Overview":
    header("Overview", "Real-time insights for safer, healthier and independent living.")
    c1,c2,c3 = st.columns([1.15,1,1.25])
    with c1:
        st.markdown("""<div class="card"><div class="card-title">Patient / Resident</div>
        <div class="big" style="font-size:21px">Person 1</div><div class="muted">Patient ID: WC-2048-A<br>Bedroom • Zone A</div></div>""", unsafe_allow_html=True)
    with c2: activity_card(state)
    with c3:
        st.markdown('<div class="card"><div class="card-title">Safety Status</div><span class="badge ok">● NORMAL</span><br><br><div class="muted">All monitored signals within current prototype thresholds.</div></div>', unsafe_allow_html=True)

    st.subheader("Tracked Occupants")
    cols = st.columns(4)
    for i,col in enumerate(cols,1):
        with col:
            p=f"Person {i}"
            if st.button(("● " if st.session_state.active_person==p else "○ ")+p, key=f"occ_{i}", use_container_width=True):
                st.session_state.active_person=p; st.rerun()

    a,b = st.columns([1.2,1])
    with a:
        st.markdown('<div class="card-title">Live Sensing Snapshot</div>', unsafe_allow_html=True); wave_chart(state,240)
    with b:
        st.markdown('<div class="card-title">Latest Safety Event</div>', unsafe_allow_html=True)
        st.markdown("""<div class="alert-warn"><b>Motion Active</b><br><span class="muted">Bedroom • Moderate risk • Review prolonged movement pattern</span></div>""", unsafe_allow_html=True)
        st.write("")
        if st.button("Open Safety Log →", use_container_width=True):
            st.session_state.page="Safety Log"; st.rerun()

elif page == "Live Sensing":
    header("Live Sensing", "Real-time CSI monitoring and AI activity recognition.")
    a,b = st.columns([2.2,1])
    with a:
        st.markdown('<div class="card-title">Live CSI Waveform</div>', unsafe_allow_html=True); wave_chart(state,320)
    with b:
        activity_card(state)
        if state:
            st.metric("CSI Sample", f'{state["index"]+1} / {state["total"]}')
            if not st.session_state.live_mode and st.button("▶ Process Next CSI Sample", use_container_width=True):
                record_event(state); st.session_state.live_sample_index += 1; st.rerun()
    st.subheader("Recent Activity Stream")
    if st.session_state.activity_history:
        st.dataframe(pd.DataFrame(st.session_state.activity_history[::-1]), use_container_width=True, hide_index=True)
    else:
        st.info("No events recorded yet. Enable Live Monitoring or process a sample.")
    if dataset is not None:
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Rows", f"{len(dataset):,}"); m2.metric("Columns", len(dataset.columns))
        m3.metric("Unique Samples", dataset["sample_id"].nunique() if "sample_id" in dataset else "—")
        m4.metric("Model Features", len(feature_columns))

elif page == "Digital Human Twin":
    header("Digital Human Twin", "Interactive body map • orientation views • sensing, risk and reported-pain layers.")

    top1,top2,top3,top4,top5 = st.columns([1,1,1,1,1.2])
    for label,col in zip(["Front","Right","Back","Left"],[top1,top2,top3,top4]):
        with col:
            if st.button(label+" View", key=f"view_{label}", use_container_width=True):
                st.session_state.twin_view=label; st.rerun()
    with top5:
        st.session_state.auto_rotate=st.toggle("Auto Rotate", value=st.session_state.auto_rotate)

    left,center,right = st.columns([1,2.4,1.25])
    with left:
        st.markdown('<div class="card-title">Body Layers</div>', unsafe_allow_html=True)
        layer = st.radio("Layer", ["Sensors","Risk","Reported Pain","Skeleton"], index=["Sensors","Risk","Reported Pain","Skeleton"].index(st.session_state.twin_layer), label_visibility="collapsed")
        st.session_state.twin_layer=layer
        st.markdown("""<div class="card"><div class="muted">
        <b style="color:#047857">●</b> Normal<br><br>
        <b style="color:#1D4ED8">●</b> Active sensing<br><br>
        <b style="color:#B45309">●</b> Attention<br><br>
        <b style="color:#B91C1C">●</b> High risk<br><br>
        <b style="color:#7C3AED">●</b> Reported pain
        </div></div>""", unsafe_allow_html=True)

    with center:
        view=st.session_state.twin_view
        # Safe isolated HTML/SVG: renders the twin, never prints markup in Streamlit.
        flip = "-1" if view=="Back" else "1"
        side = view in ("Left","Right")
        torso_rx = 42 if side else 68
        shoulder = 42 if side else 78
        layer=st.session_state.twin_layer
        markers = {
            "Sensors":[("#047857",300,105),("#1D4ED8",300,190),("#0E7490",235,245),("#B45309",355,330),("#1D4ED8",265,500),("#047857",335,500)],
            "Risk":[("#047857",300,105),("#B45309",355,330),("#B91C1C",265,500)],
            "Skeleton":[("#1D4ED8",300,190),("#0E7490",300,330),("#1D4ED8",265,500),("#1D4ED8",335,500)],
            "Reported Pain":[("#7C3AED",355,330)] if st.session_state.pain_points else []
        }[layer]
        dots="".join([f'<circle cx="{x}" cy="{y}" r="9" fill="{c}" stroke="white" stroke-width="4"/><circle cx="{x}" cy="{y}" r="18" fill="none" stroke="{c}" stroke-opacity=".25" stroke-width="8"/>' for c,x,y in markers])
        html=f"""
        <div style="height:650px;background:#fff;border:1px solid #CBD5E1;border-radius:18px;display:flex;align-items:center;justify-content:center;font-family:Inter,sans-serif;position:relative;overflow:hidden">
        <div style="position:absolute;top:16px;left:18px;color:#64748B;font-size:12px">{view.upper()} • {layer.upper()} LAYER</div>
        <svg viewBox="0 0 600 650" width="100%" height="610">
          <defs>
            <linearGradient id="body" x1="0" x2="1"><stop stop-color="#E2E8F0"/><stop offset=".5" stop-color="#F8FAFC"/><stop offset="1" stop-color="#CBD5E1"/></linearGradient>
            <filter id="shadow"><feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="#0E7490" flood-opacity=".15"/></filter>
          </defs>
          <ellipse cx="300" cy="612" rx="110" ry="16" fill="#E2E8F0"/>
          <g transform="translate(300 0) scale({flip} 1) translate(-300 0)" filter="url(#shadow)">
            <circle cx="300" cy="85" r="48" fill="url(#body)" stroke="#94A3B8" stroke-width="2"/>
            <rect x="282" y="125" width="36" height="34" rx="14" fill="url(#body)" stroke="#94A3B8"/>
            <path d="M{300-shoulder} 175 Q300 140 {300+shoulder} 175 L{300+torso_rx-10} 345 Q300 385 {300-torso_rx+10} 345 Z" fill="url(#body)" stroke="#94A3B8" stroke-width="2"/>
            <path d="M{300-shoulder} 180 Q205 250 205 365 Q205 390 225 380 L255 245" fill="url(#body)" stroke="#94A3B8" stroke-width="2"/>
            <path d="M{300+shoulder} 180 Q395 250 395 365 Q395 390 375 380 L345 245" fill="url(#body)" stroke="#94A3B8" stroke-width="2"/>
            <path d="M260 345 Q250 430 260 585 Q275 605 288 580 L300 365" fill="url(#body)" stroke="#94A3B8" stroke-width="2"/>
            <path d="M340 345 Q350 430 340 585 Q325 605 312 580 L300 365" fill="url(#body)" stroke="#94A3B8" stroke-width="2"/>
            <path d="M260 590 L235 610 Q230 622 260 622 L286 610 L286 580" fill="url(#body)" stroke="#94A3B8"/>
            <path d="M340 590 L365 610 Q370 622 340 622 L314 610 L314 580" fill="url(#body)" stroke="#94A3B8"/>
            <path d="M270 190 Q300 210 330 190 M270 270 Q300 290 330 270 M300 160 L300 350" fill="none" stroke="#0E7490" stroke-opacity=".28" stroke-width="2"/>
            {dots}
          </g>
          <circle cx="300" cy="300" r="145" fill="none" stroke="#0E7490" stroke-opacity=".10" stroke-width="2" stroke-dasharray="7 8"/>
          <circle cx="300" cy="300" r="195" fill="none" stroke="#1D4ED8" stroke-opacity=".08" stroke-width="2" stroke-dasharray="7 10"/>
        </svg></div>"""
        components.html(html, height=665)

        r1,r2,r3 = st.columns(3)
        with r1:
            if st.button("↶ Rotate Left", use_container_width=True):
                views=["Front","Left","Back","Right"]; st.session_state.twin_view=views[(views.index(view)+1)%4]; st.rerun()
        with r2:
            if st.button("↺ Reset View", use_container_width=True):
                st.session_state.twin_view="Front"; st.session_state.twin_layer="Sensors"; st.rerun()
        with r3:
            if st.button("Rotate Right ↷", use_container_width=True):
                views=["Front","Right","Back","Left"]; st.session_state.twin_view=views[(views.index(view)+1)%4]; st.rerun()

    with right:
        st.markdown('<div class="card-title">Selected Region</div>', unsafe_allow_html=True)
        regions=["Head","Chest","Abdomen","Left Shoulder","Right Shoulder","Left Elbow","Right Elbow","Left Wrist","Right Wrist","Left Hip","Right Hip","Left Knee","Right Knee","Left Ankle","Right Ankle"]
        region=st.selectbox("Body region", regions, index=regions.index(st.session_state.selected_region) if st.session_state.selected_region in regions else 0)
        st.session_state.selected_region=region
        activity = state["prediction"].replace("_"," ").title() if state else "Unavailable"
        conf = f'{state["confidence"]:.1f}%' if state else "—"
        st.markdown(f"""<div class="card"><div class="big" style="font-size:20px">{region}</div><br>
        <div class="muted">Sensor State</div><b style="color:#047857">Active</b><hr>
        <div class="muted">Current Activity</div><b>{activity}</b><hr>
        <div class="muted">AI Confidence</div><b>{conf}</b><hr>
        <div class="muted">Risk Level</div><b style="color:#B45309">Moderate</b></div>""", unsafe_allow_html=True)
        st.markdown('<div class="card-title">Report Pain / Discomfort</div>', unsafe_allow_html=True)
        severity=st.slider("Severity",1,10, st.session_state.pain_points.get(region,5))
        if st.button("Add / Update Pain Point", use_container_width=True):
            st.session_state.pain_points[region]=severity; st.session_state.twin_layer="Reported Pain"; st.success(f"{region}: {severity}/10 recorded")
        if region in st.session_state.pain_points and st.button("Clear Selected Pain Point", use_container_width=True):
            del st.session_state.pain_points[region]; st.rerun()

elif page == "Sleep & Vitals":
    header("Sleep & Vitals", "Trend vital signs and sleep patterns. Demo / simulated values in this software MVP.")
    st.warning("Demo / simulated values — not direct clinical measurements from the current CSI activity model.")
    vals=[("Heart Rate","72 BPM"),("Breathing Rate","16 /min"),("Blood Pressure","118/76 mmHg"),("HRV","42 ms")]
    cols=st.columns(4)
    for c,(n,v) in zip(cols,vals):
        with c: st.metric(n,v,"Stable")
    a,b=st.columns([1.5,1])
    with a:
        x=np.arange(48); fig=go.Figure()
        fig.add_trace(go.Scatter(x=x,y=70+4*np.sin(x/3),name="Heart Rate",line=dict(color="#1D4ED8")))
        fig.add_trace(go.Scatter(x=x,y=16+1.5*np.sin(x/5),name="Breathing",line=dict(color="#0E7490")))
        fig.update_layout(height=330,paper_bgcolor="#fff",plot_bgcolor="#fff",margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig,use_container_width=True)
    with b:
        st.metric("Sleep Duration","7h 24m"); st.metric("Sleep Quality","86%","Good")
        st.progress(.86)

elif page == "Mobility & Fall Risk":
    header("Mobility & Fall Risk", "Analyse movement patterns, inactivity and potential safety risks.")
    c1,c2,c3=st.columns(3)
    with c1: activity_card(state)
    with c2: st.metric("Fall Risk Indicator","41%","Moderate")
    with c3: st.metric("Inactivity Duration","12 min","Last movement recently")
    st.subheader("Movement Pattern")
    hist=pd.DataFrame(st.session_state.activity_history)
    if not hist.empty:
        counts=hist["Activity"].value_counts()
        fig=go.Figure(go.Bar(x=counts.index,y=counts.values,marker_color="#1D4ED8"))
        fig.update_layout(height=300,paper_bgcolor="#fff",plot_bgcolor="#fff")
        st.plotly_chart(fig,use_container_width=True)
    else: st.info("Live activity history will populate this analysis.")
    st.markdown('<div class="alert-warn"><b>Moderate attention</b><br><span class="muted">Prototype rule: static/low-movement states are flagged for monitoring.</span></div>',unsafe_allow_html=True)

elif page == "Biometrics":
    header("Biometrics", "Sensor-oriented biometric information. Demo / simulated values.")
    st.warning("These biometric values are UI demonstrations and are not inferred by the current activity classifier.")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Respiratory Pattern","16 /min"); c2.metric("Heart Rate Variability","42 ms")
    c3.metric("Signal Quality","94%"); c4.metric("Stress Indicator","Low")
    x=np.arange(60)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=x,y=70+3*np.sin(x/3),name="Heart Rate",line=dict(color="#B91C1C")))
    fig.add_trace(go.Scatter(x=x,y=16+np.sin(x/4),name="Respiration",line=dict(color="#0E7490")))
    fig.update_layout(height=360,paper_bgcolor="#fff",plot_bgcolor="#fff")
    st.plotly_chart(fig,use_container_width=True)

elif page == "Attribution":
    header("Attribution", "Multi-occupant selection and activity attribution.")
    cols=st.columns(4)
    for i,col in enumerate(cols,1):
        p=f"Person {i}"
        with col:
            st.markdown(f'<div class="card"><div class="card-title">{p}</div><div class="muted">{"Active • Bedroom Zone A" if p==st.session_state.active_person else "Standby"}</div></div>',unsafe_allow_html=True)
            if st.button("Select",key=f"attr_{i}",use_container_width=True):
                st.session_state.active_person=p; st.rerun()
    a,b=st.columns(2)
    with a: st.metric("Selected Occupant",st.session_state.active_person)
    with b: st.metric("Attribution Confidence","98%","Prototype demo")
    st.info("Multi-person attribution is represented in the interface; the current trained classifier is an activity-recognition model.")

elif page == "Safety Log":
    header("Safety Log", "Review, acknowledge and manage safety events.")
    for i,a in enumerate(st.session_state.alerts):
        cls="alert-high" if a["risk"]=="High" else "alert-warn" if a["risk"]=="Moderate" else "card"
        st.markdown(f'<div class="{cls}"><b>{a["event"]}</b><br><span class="muted">{a["time"]} • {a["zone"]} • Risk: {a["risk"]} • Status: {a["status"]}</span></div>',unsafe_allow_html=True)
        x,y,z=st.columns([1,1,4])
        with x:
            if st.button("Acknowledge",key=f"ack_{i}"):
                st.session_state.alerts[i]["status"]="Acknowledged"; st.rerun()
        with y:
            if st.button("Open Twin",key=f"twin_{i}"):
                st.session_state.page="Digital Human Twin"; st.session_state.twin_layer="Risk"; st.rerun()

elif page == "Simple Summary":
    header("Simple Summary", "Everything important about the monitored person, explained in simple words.")
    current_activity = state["prediction"].replace("_", " ").title() if state else "Unavailable"
    current_conf = f'{state["confidence"]:.1f}%' if state else "—"
    current_status = state["status"] if state else "Unavailable"
    selected = st.session_state.active_person
    pain_count = len(st.session_state.pain_points)
    open_alerts = sum(1 for a in st.session_state.alerts if a["status"] == "Open")

    st.markdown(f"""<div class="card"><div class="card-title">At a Glance</div>
    <div class="big" style="font-size:22px">{selected} is currently {current_activity.lower()}.</div>
    <div class="muted" style="margin-top:8px">The AI is {current_conf} confident. Current safety interpretation: <b>{current_status}</b>.</div></div>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="card"><div class="card-title">Movement</div><div class="muted">WeCare currently recognizes the activity as <b>{current_activity}</b>. This comes from the CSI activity-recognition model.</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="card"><div class="card-title">Safety</div><div class="muted">There are <b>{open_alerts} open safety alerts</b>. NORMAL means no current prototype rule needs attention; MONITOR means the activity should be watched.</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="card"><div class="card-title">Body / Pain</div><div class="muted"><b>{pain_count}</b> reported pain or discomfort point(s) are currently saved on the Digital Human Twin.</div></div>""", unsafe_allow_html=True)

    st.subheader("What each area means")
    summary_rows = [
        ["Live Sensing", "Shows the Wi-Fi CSI signal and the activity predicted by the AI model."],
        ["AI Confidence", f"Shows how sure the model is about the current prediction. Current confidence: {current_conf}."],
        ["Digital Human Twin", "Shows the monitored person visually and lets the caregiver inspect body regions, sensing points, risk areas and reported discomfort."],
        ["Sleep & Vitals", "Shows prototype health and sleep information in an easy-to-read form. These values are simulated in the current MVP."],
        ["Mobility & Fall Risk", "Summarizes movement, inactivity and prototype safety-risk indicators."],
        ["Biometrics", "Shows demo biometric trends. They are not measurements produced by the current CSI activity classifier."],
        ["Attribution", "Shows which tracked occupant is selected and represents the planned multi-person monitoring workflow."],
        ["Safety Log", "Keeps safety events in one place so a caregiver can review, acknowledge and open the related Digital Human Twin."],
        ["Reports", "Turns the activity history into a simple table and downloadable CSV report."],
        ["System Status", "Shows whether the local CSI/model pipeline and interface are operating."],
    ]
    st.dataframe(pd.DataFrame(summary_rows, columns=["Criteria", "Simple Meaning"]), use_container_width=True, hide_index=True)
    st.info("Prototype note: Activity recognition is model-driven. Vitals, biometrics, multi-person attribution and some safety indicators are demonstration features unless connected to the required sensors/backend.")

elif page == "Reports":
    header("Reports", "Generate a concise monitoring report from the current prototype session.")
    period=st.segmented_control("Report period",["Daily","Weekly","Monthly"],default="Daily")
    hist=pd.DataFrame(st.session_state.activity_history)
    if hist.empty:
        st.info("Enable Live Monitoring to collect session activity before exporting.")
    else:
        st.dataframe(hist,use_container_width=True,hide_index=True)
        csv=hist.to_csv(index=False).encode("utf-8")
        st.download_button("Download Activity CSV",csv,"wecare_activity_report.csv","text/csv")
    st.caption(f"Selected report view: {period}")

elif page == "Settings":
    header("Settings", "Configure prototype sensing, alerts and interface preferences.")
    t1,t2,t3=st.tabs(["Sensing","Alerts","System"])
    with t1:
        rate=st.select_slider("Sample interval",options=[1,2,3,5,10],value=2,format_func=lambda x:f"{x} sec")
        st.toggle("Live Monitoring",value=st.session_state.live_mode,key="settings_live")
        st.selectbox("Sensor Zone",["Bedroom • Zone A","Living Room • Zone B","Kitchen • Zone C"])
    with t2:
        st.slider("Moderate risk threshold",0,100,40)
        st.slider("High risk threshold",0,100,75)
        st.toggle("Show Priority 1 alerts",True); st.toggle("Show Priority 2 alerts",True)
    with t3:
        st.text_input("Node ID","WC-PUNE-01")
        st.selectbox("Telemetry mode",["Prototype / Local","Connected backend"])
        st.info("Settings on this page are prototype UI controls unless connected to a backend.")

# -------------------- SYSTEM FOOTER --------------------
st.write("")
st.markdown("""<div class="system-strip"><b style="color:#0F172A">SYSTEM STATUS</b>
&nbsp; • &nbsp; CSI Engine: <b style="color:#047857">ONLINE</b>
&nbsp; • &nbsp; Live UI: <b style="color:#047857">CONNECTED</b>
&nbsp; • &nbsp; Model: <b style="color:#1D4ED8">Random Forest</b>
&nbsp; • &nbsp; Telemetry: <b style="color:#64748B">PROTOTYPE / LOCAL</b>
</div>""", unsafe_allow_html=True)

# Automatic live replay. Kept at the end so the full page renders first.
if st.session_state.live_mode and state is not None:
    st.session_state.live_sample_index += 1
    time.sleep(2)
    st.rerun()
