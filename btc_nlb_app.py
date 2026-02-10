"""
================================================================================
BITCOIN NLB (Never Look Back) ANALYSIS — STREAMLIT APP
================================================================================
Multi-Asset Power Law Channel with Risk Area System
Data: CryptoCompare | Models: NLB Floor Envelope, Balanced PL, Risk Area
================================================================================
"""
import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.optimize import minimize_scalar
import time

st.set_page_config(page_title="BTC NLB Analysis", page_icon="₿", layout="wide", initial_sidebar_state="expanded")

if 'disclaimer_accepted' not in st.session_state:
    st.session_state.disclaimer_accepted = False

# ============================================================================
# DISCLAIMER
# ============================================================================
def show_disclaimer():
    st.markdown('<p style="font-size:2.5rem;font-weight:bold;color:#F7931A;text-align:center;">₿ Bitcoin NLB Analysis</p>', unsafe_allow_html=True)
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        st.error("""
        **⚠️ DISCLAIMER — READ BEFORE PROCEEDING**
        
        This tool is for **informational and educational purposes only**. It does NOT constitute financial advice.
        Models are based on historical data and **do not guarantee future results**.
        The crypto market is highly volatile. **You may lose all invested capital.**
        Consult a qualified financial advisor before investing. The author assumes **no responsibility** for losses.
        """)
        accept = st.checkbox("✅ I have read, understood and accept the conditions above")
        if st.button("🚀 ENTER", type="primary", disabled=not accept, use_container_width=True):
            st.session_state.disclaimer_accepted = True
            st.rerun()

if not st.session_state.disclaimer_accepted:
    show_disclaimer()
    st.stop()

st.markdown("""
<style>
.main-header {font-size:2.5rem;font-weight:bold;color:#F7931A;text-align:center;padding:0.3rem;}
.sub-header {text-align:center;color:#888;margin-bottom:1rem;}
.sig-green {background:linear-gradient(135deg,#0f5132,#198754);color:white;padding:1rem;border-radius:10px;text-align:center;font-size:1.3rem;font-weight:bold;}
.sig-yellow {background:linear-gradient(135deg,#664d03,#ffc107);color:black;padding:1rem;border-radius:10px;text-align:center;font-size:1.3rem;font-weight:bold;}
.sig-orange {background:linear-gradient(135deg,#7c2d12,#ea580c);color:white;padding:1rem;border-radius:10px;text-align:center;font-size:1.3rem;font-weight:bold;}
.sig-red {background:linear-gradient(135deg,#7f1d1d,#dc2626);color:white;padding:1rem;border-radius:10px;text-align:center;font-size:1.3rem;font-weight:bold;}
.sig-gray {background:linear-gradient(135deg,#495057,#6c757d);color:white;padding:1rem;border-radius:10px;text-align:center;font-size:1.3rem;font-weight:bold;}
</style>
""", unsafe_allow_html=True)

GENESIS = pd.Timestamp('2009-01-03')

# ============================================================================
# DATA FUNCTIONS
# ============================================================================
@st.cache_data(ttl=3600)
def _fetch_cc(fsym, tsym, start_date='2010-07-18'):
    """Generic CryptoCompare histoday fetcher with pagination."""
    all_data = []
    cur_ts = int(datetime.now().timestamp())
    start_ts = int(pd.Timestamp(start_date).timestamp())
    while cur_ts > start_ts:
        r = requests.get('https://min-api.cryptocompare.com/data/v2/histoday',
                         params={'fsym': fsym, 'tsym': tsym, 'limit': 2000, 'toTs': cur_ts}, timeout=30)
        js = r.json()
        if js['Response'] != 'Success':
            break
        chunk = pd.DataFrame(js['Data']['Data'])
        all_data.append(chunk)
        oldest = chunk['time'].min()
        cur_ts = oldest - 86400
        if oldest <= start_ts:
            break
        time.sleep(0.2)
    if not all_data:
        return None
    df = pd.concat(all_data, ignore_index=True)
    df['Date'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('Date').drop_duplicates('Date', keep='first').reset_index(drop=True)
    df = df[['Date', 'close', 'high', 'low', 'open']]
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open']
    df = df[(df['Date'] >= start_date) & (df['Close'] > 0)].reset_index(drop=True)
    return df

# ============================================================================
# NLB CORE FUNCTIONS
# ============================================================================
def compute_nlb(series):
    """NLB = cummax(reverse_cummin(series)). Monotone non-decreasing floor."""
    s = series.to_numpy(dtype=float)
    fmin = np.minimum.accumulate(s[::-1])[::-1]
    return pd.Series(np.maximum.accumulate(fmin), index=series.index)

def fit_floor_envelope(days, nlb, b_min=4.0, b_max=7.5, b_steps=4000):
    """
    Floor A: power law y=a*x^b under all NLB update points,
    minimizing log-space area (gap). Grid search on b.
    """
    upd = np.zeros(len(nlb), dtype=bool)
    upd[0] = True
    upd[1:] = nlb[1:] > nlb[:-1]
    x, y = days[upd].astype(float), nlb[upd].astype(float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    logx, logy = np.log(x), np.log(y)
    best = None
    for b in np.linspace(b_min, b_max, b_steps):
        c = float(np.min(logy - b * logx))
        gap = float(np.sum(logy - (c + b * logx)))
        if best is None or gap < best['gap']:
            best = {'a': float(np.exp(c)), 'b': float(b), 'gap': gap}
    return best

def fit_balanced(days, nlb, a_fl, b_fl):
    """
    Balanced PL: shift Floor upward until area(NLB above) = area(NLB below)
    in log-space on NLB update points. Same exponent, only coefficient changes.
    """
    upd = np.zeros(len(nlb), dtype=bool)
    upd[0] = True
    upd[1:] = nlb[1:] > nlb[:-1]
    x, y = days[upd].astype(float), nlb[upd].astype(float)
    m = (x > 0) & (y > 0); x, y = x[m], y[m]
    lf, ly = np.log(a_fl * (x ** b_fl)), np.log(y)
    def obj(k):
        d = ly - (lf + k)
        return abs(float(np.sum(np.maximum(0, d))) - float(np.sum(np.maximum(0, -d))))
    res = minimize_scalar(obj, bounds=(-2, 3), method='bounded')
    return {'a': a_fl * np.exp(res.x), 'b': b_fl, 'k': float(res.x)}

def fit_max_crossing(days, nlb, a_fl, b_fl, k_steps=2000):
    """Max-Crossing PL: shift maximizing NLB crossings of the curve."""
    x, y = days.astype(float), nlb.astype(float)
    m = (x > 0) & (y > 0); x, y = x[m], y[m]
    lf, ly = np.log(a_fl * (x ** b_fl)), np.log(y)
    best_k, best_c = 0.0, 0
    for k in np.linspace(-0.5, 1.5, k_steps):
        s = np.sign(ly - (lf + k))
        for i in range(1, len(s)):
            if s[i] == 0: s[i] = s[i-1]
        c = int(np.sum(s[1:] != s[:-1]))
        if c > best_c:
            best_c, best_k = c, float(k)
    return {'a': a_fl * np.exp(best_k), 'b': b_fl, 'k': best_k, 'n_cross': best_c}

# ============================================================================
# RISK AREA SYSTEM — Exponential Decay (improved design)
# ============================================================================
#
# Design rationale (vs hard-reset approach in Colab):
#
# 1. HARD RESET PROBLEM: In the Colab we reset area to 0 when price touches
#    floor. But a 1-day touch doesn't truly "discharge" all risk.
#    With decay, a brief touch reduces area gradually — more realistic.
#
# 2. TIME NORMALIZATION: Raw cumulative area grows with time even if price
#    tracks floor exactly. The rolling-max normalization gives a 0-1 ratio
#    comparable across epochs.
#
# 3. λ CALIBRATION: We optimize λ by maximizing separation of forward returns
#    between high-risk and low-risk regimes. This is a single hyperparameter
#    with a clear, backtestable objective.
#
# area(t) = area(t-1) × λ + max(0, log(price_t / floor_t))
# risk(t) = area(t) / max(area[t-W:t])   ∈ [0, 1]
# ============================================================================

def compute_risk_area(price, floor_v, lam=0.997):
    """Decayed cumulative log-excess above floor."""
    n = len(price)
    excess = np.maximum(0.0, np.log(np.maximum(price / floor_v, 1e-12)))
    area = np.empty(n)
    area[0] = excess[0]
    for t in range(1, n):
        area[t] = area[t - 1] * lam + excess[t]
    return area

def compute_risk_ratio(area, window=365):
    """Normalize area to [0,1] using walk-forward rolling max."""
    n = len(area)
    risk = np.zeros(n)
    for t in range(n):
        rm = float(np.max(area[max(0, t - window):t + 1]))
        if rm > 0:
            risk[t] = area[t] / rm
    return np.clip(risk, 0, 1)

def backtest_risk_area(price, floor_v, lam, rw, horizons=(30, 90, 180, 365)):
    """
    Walk-forward backtest: for each day, compute risk ratio (past only),
    then measure forward return and max drawdown over horizons.
    Group by risk quintile.
    """
    n = len(price)
    area = compute_risk_area(price, floor_v, lam)
    risk = compute_risk_ratio(area, rw)
    buckets = [
        ('Very Low 0-20%', 0.0, 0.2),
        ('Low 20-40%', 0.2, 0.4),
        ('Mid 40-60%', 0.4, 0.6),
        ('High 60-80%', 0.6, 0.8),
        ('Very High 80-100%', 0.8, 1.01),
    ]
    rows = []
    for h in horizons:
        fwd = np.full(n, np.nan)
        fdd = np.full(n, np.nan)
        for t in range(n - h):
            w = price[t:t + h + 1]
            fwd[t] = w[-1] / w[0] - 1
            fdd[t] = float((w / np.maximum.accumulate(w) - 1).min())
        for bname, lo, hi in buckets:
            mask = (risk >= lo) & (risk < hi) & np.isfinite(fwd)
            cnt = int(mask.sum())
            if cnt < 15:
                continue
            r, d = fwd[mask], fdd[mask]
            rows.append({
                'Risk Bucket': bname,
                'Horizon': f"{h}d",
                'N': cnt,
                'Median Ret': f"{np.median(r)*100:+.1f}%",
                'Mean Ret': f"{np.mean(r)*100:+.1f}%",
                '% Positive': f"{(r > 0).mean()*100:.0f}%",
                'Median MaxDD': f"{np.median(d)*100:.1f}%",
                'P10 MaxDD': f"{np.percentile(d, 10)*100:.1f}%",
            })
    return pd.DataFrame(rows)

def calibrate_lambda(price, floor_v, rw=365):
    """Find λ maximizing 6M return separation between risk regimes."""
    fwd = np.full(len(price), np.nan)
    for t in range(len(price) - 180):
        fwd[t] = price[t + 180] / price[t] - 1
    best_lam, best_sep = 0.997, 0.0
    rows = []
    for lam in np.arange(0.990, 0.9996, 0.0005):
        a = compute_risk_area(price, floor_v, lam)
        r = compute_risk_ratio(a, rw)
        hi = (r > 0.7) & np.isfinite(fwd)
        lo = (r < 0.3) & np.isfinite(fwd)
        if hi.sum() > 20 and lo.sum() > 20:
            sep = float(np.median(fwd[lo]) - np.median(fwd[hi]))
        else:
            sep = 0.0
        rows.append({'lambda': float(lam), 'separation': sep})
        if sep > best_sep:
            best_sep, best_lam = sep, float(lam)
    return best_lam, pd.DataFrame(rows)

# ============================================================================
# CHANNEL & HISTORICAL PROBABILITIES
# ============================================================================
def channel_pos(nlb, fl, cl):
    """Position in Floor-Ceiling channel: 0=floor, 1=ceiling."""
    ln = np.log(np.maximum(nlb, 1e-12))
    lf = np.log(np.maximum(fl, 1e-12))
    lc = np.log(np.maximum(cl, 1e-12))
    w = lc - lf
    return np.clip(np.where(w > 0, (ln - lf) / w, 0.5), 0, 1)

def hist_prob_table(price, chpos, fl_v, bal_v, cur_ch,
                    tol=0.07, horizons=(30, 60, 90, 180, 365)):
    """
    From days with similar channel position, compute probability
    of price reaching each target level within each horizon.
    """
    n = len(price)
    max_h = max(horizons)
    sim = np.where((np.abs(chpos - cur_ch) < tol) & (np.arange(n) < n - max_h))[0]
    if len(sim) < 10:
        return None, 0
    targets = {'Floor': 'floor', 'Balanced': 'balanced',
               '-10%': -0.10, '-20%': -0.20, '-30%': -0.30, '-40%': -0.40}
    rows = []
    for h in horizons:
        vi = sim[sim < n - h]
        if len(vi) < 5:
            continue
        for tname, tdef in targets.items():
            hit = 0
            for idx in vi:
                mn = float(np.min(price[idx:idx + h + 1]))
                if tdef == 'floor':
                    tp = fl_v[idx]
                elif tdef == 'balanced':
                    tp = bal_v[idx]
                else:
                    tp = price[idx] * (1 + tdef)
                if mn <= tp:
                    hit += 1
            rows.append({'Horizon': f"{h}d", 'Level': tname,
                         'Prob': hit / len(vi), 'N': len(vi)})
    return (pd.DataFrame(rows) if rows else None), len(sim)

# ============================================================================
# MODEL BUILDER
# ============================================================================
def build_model(df_raw, name='BTC/USD', lam=0.997, rw=365):
    """Full NLB model: floor, balanced, ceiling, max-cross, risk area."""
    df = df_raw.copy()
    df['Days'] = (df['Date'] - GENESIS).dt.days
    base = df['Low'] if 'Low' in df.columns else df['Close']
    df['NLB'] = compute_nlb(base.dropna())
    df = df[df['Days'] > 0].copy().reset_index(drop=True)

    d = df['Days'].to_numpy(float)
    nl = df['NLB'].to_numpy(float)
    pr = df['Close'].to_numpy(float)

    fl = fit_floor_envelope(d, nl)
    bl = fit_balanced(d, nl, fl['a'], fl['b'])
    mc = fit_max_crossing(d, nl, fl['a'], fl['b'])
    a_ceil = bl['a'] * np.exp(bl['k'])

    df['Floor']    = fl['a'] * (d ** fl['b'])
    df['Balanced'] = bl['a'] * (d ** bl['b'])
    df['Ceiling']  = a_ceil  * (d ** bl['b'])
    df['MaxCross'] = mc['a'] * (d ** mc['b'])
    df['ChPos']    = channel_pos(nl, df['Floor'].values, df['Ceiling'].values)

    ra = compute_risk_area(pr, df['Floor'].values, lam)
    rr = compute_risk_ratio(ra, rw)
    df['RiskArea']  = ra
    df['RiskRatio'] = rr

    return {
        'df': df, 'name': name,
        'floor': fl, 'balanced': bl, 'ceiling': {'a': a_ceil, 'b': bl['b']},
        'maxcross': mc,
    }

def project_curves(model, years=5):
    """Project floor/balanced/ceiling forward."""
    df = model['df']
    last_d = df['Days'].iloc[-1]
    last_dt = df['Date'].iloc[-1]
    fd = pd.date_range(last_dt + pd.Timedelta(days=1), periods=years * 365, freq='D')
    fdays = np.array([(d - GENESIS).days for d in fd], dtype=float)
    return pd.DataFrame({
        'Date': fd, 'Days': fdays,
        'Floor':    model['floor']['a']    * (fdays ** model['floor']['b']),
        'Balanced': model['balanced']['a'] * (fdays ** model['balanced']['b']),
        'Ceiling':  model['ceiling']['a']  * (fdays ** model['ceiling']['b']),
    })

# ============================================================================
# SIGNAL
# ============================================================================
def get_signal(ch, rr, dist_fl):
    if ch < 0.15:
        return 'STRONG BUY', '🟢🟢', f'Near floor ({ch*100:.0f}% channel). Rare.', 'sig-green'
    elif ch < 0.30:
        return 'BUY', '🟢', f'Low channel ({ch*100:.0f}%). Good entry.', 'sig-green'
    elif ch < 0.50 and rr < 0.5:
        return 'ACCUMULATE', '🟡', f'Mid-low ({ch*100:.0f}%), moderate risk ({rr*100:.0f}%).', 'sig-yellow'
    elif ch > 0.85:
        return 'HIGH RISK', '🔴', f'Very high ({ch*100:.0f}%). Extended.', 'sig-red'
    elif ch > 0.70 and rr > 0.7:
        return 'CAUTION', '🟠', f'High ({ch*100:.0f}%) + high risk ({rr*100:.0f}%).', 'sig-orange'
    elif ch < 0.70:
        return 'HOLD', '⚪', f'Mid channel ({ch*100:.0f}%).', 'sig-gray'
    else:
        return 'WATCH', '🟡', f'Upper channel ({ch*100:.0f}%). Monitor.', 'sig-yellow'

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.markdown('<p class="main-header">₿ Bitcoin NLB Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Never Look Back Power Law Channel — Multi-Asset</p>', unsafe_allow_html=True)

    # ── SIDEBAR ──
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/1200px-Bitcoin.svg.png", width=60)
        st.title("⚙️ Settings")
        start_year = st.selectbox("Start Year", [2010, 2011, 2012, 2013], index=0)
        proj_years = st.slider("Projection (years)", 1, 5, 3)
        st.markdown("---")
        st.markdown("### 🌍 Multi-Asset")
        load_gold = st.checkbox("BTC / Gold (XAU)", value=False)
        load_eur  = st.checkbox("BTC / EUR", value=False)
        st.markdown("---")
        st.markdown("### 🔬 Risk Area")
        risk_lam = st.slider("Decay λ", 0.990, 0.999, 0.997, 0.001,
                              help="Higher = longer memory. 0.997 ≈ 231-day half-life.")
        risk_win = st.slider("Norm. Window (days)", 180, 730, 365, 30)
        st.markdown("---")
        st.markdown("### 📚 NLB Curves")
        st.caption("**Floor**: envelope (absolute support)\n\n"
                   "**Balanced**: equal log-area (fair value)\n\n"
                   "**Ceiling**: symmetric to floor\n\n"
                   "**Max-Cross**: max NLB crossings")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    # ── LOAD DATA ──
    with st.spinner("📥 Loading BTC data…"):
        btc_raw = _fetch_cc('BTC', 'USD', f'{start_year}-01-01')
        if btc_raw is None or len(btc_raw) < 100:
            st.error("Failed to load BTC data"); return

    # ── BUILD MODEL ──
    with st.spinner("🔧 Building NLB model…"):
        M = build_model(btc_raw, 'BTC/USD', risk_lam, risk_win)

    df  = M['df']
    d   = df['Days'].to_numpy(float)
    pr  = df['Close'].to_numpy(float)
    nl  = df['NLB'].to_numpy(float)
    fl  = df['Floor'].to_numpy(float)
    bl  = df['Balanced'].to_numpy(float)
    cl  = df['Ceiling'].to_numpy(float)
    chp = df['ChPos'].to_numpy(float)
    rr  = df['RiskRatio'].to_numpy(float)

    TODAY   = df['Date'].iloc[-1]
    TDAYS   = df['Days'].iloc[-1]
    CUR_P   = float(pr[-1])
    CUR_NLB = float(nl[-1])
    CUR_FL  = float(fl[-1])
    CUR_BL  = float(bl[-1])
    CUR_CL  = float(cl[-1])
    CUR_CH  = float(chp[-1])
    CUR_RR  = float(rr[-1])
    DST_FL  = (CUR_P / CUR_FL - 1) * 100
    DST_BL  = (CUR_P / CUR_BL - 1) * 100

    sig_name, sig_emo, sig_desc, sig_css = get_signal(CUR_CH, CUR_RR, DST_FL)

    # ── TOP METRICS ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Price",    f"${CUR_P:,.0f}",  f"{(CUR_P/pr[-2]-1)*100:+.2f}%")
    c2.metric("🔒 Floor",    f"${CUR_FL:,.0f}",  f"{DST_FL:+.1f}% away")
    c3.metric("⚖️ Balanced", f"${CUR_BL:,.0f}",  f"{DST_BL:+.1f}%")
    c4.metric("📊 Channel",  f"{CUR_CH*100:.1f}%", "0%=Floor | 100%=Ceil")
    c5.metric("⚠️ Risk",     f"{CUR_RR*100:.0f}%", "Low<30 | High>70")
    st.markdown("---")
    st.markdown(f'<div class="{sig_css}">{sig_emo} {sig_name}<br>'
                f'<span style="font-size:0.9rem;font-weight:normal;">{sig_desc}</span></div>',
                unsafe_allow_html=True)
    st.markdown("---")

    # ── TABS ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 NLB Channel", "⚠️ Risk Area", "📊 Probabilities",
        "🎯 Entry Strategy", "🌍 Multi-Asset"])

    # ================================================================
    # TAB 1 — NLB CHANNEL
    # ================================================================
    with tab1:
        st.subheader("📈 NLB Power Law Channel")
        proj = project_curves(M, proj_years)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='BTC Price',
                                 line=dict(color='#F7931A', width=1.5), opacity=0.7))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['NLB'], name='NLB',
                                 line=dict(color='white', width=2)))
        # channel fill
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Ceiling'], name='Ceiling',
                                 line=dict(color='rgba(220,53,69,0.3)', width=1)))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Floor'], name='Floor',
                                 line=dict(color='rgba(40,167,69,0.6)', width=2),
                                 fill='tonexty', fillcolor='rgba(247,147,26,0.05)'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Balanced'], name='Balanced',
                                 line=dict(color='#6c757d', width=1.5, dash='dash')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MaxCross'], name='Max-Cross',
                                 line=dict(color='#17a2b8', width=1, dash='dot'),
                                 visible='legendonly'))
        # projections
        for col, clr in [('Floor','rgba(40,167,69,0.4)'),('Balanced','rgba(108,117,125,0.4)'),
                          ('Ceiling','rgba(220,53,69,0.2)')]:
            fig.add_trace(go.Scatter(x=proj['Date'], y=proj[col], showlegend=False,
                                     line=dict(color=clr, width=1, dash='dot')))
        ymax = max(float(pr.max()), float(proj['Ceiling'].max())) * 1.5
        fig.update_layout(yaxis_type='log', yaxis_title='Price (USD)', height=550,
                          hovermode='x unified', template='plotly_dark',
                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                          margin=dict(l=0, r=0, t=30, b=0))
        fig.update_yaxes(range=[np.log10(max(float(fl.min()), 0.1)), np.log10(ymax)])
        st.plotly_chart(fig, use_container_width=True)

        # Channel position history
        st.subheader("📊 Channel Position Over Time")
        fig_ch = go.Figure()
        fig_ch.add_trace(go.Scatter(x=df['Date'], y=df['ChPos']*100, name='Channel %',
                                    line=dict(color='#F7931A', width=1.5),
                                    fill='tozeroy', fillcolor='rgba(247,147,26,0.1)'))
        fig_ch.add_hline(y=70, line_dash='dash', line_color='red', opacity=0.5,
                         annotation_text="Caution 70%")
        fig_ch.add_hline(y=30, line_dash='dash', line_color='green', opacity=0.5,
                         annotation_text="Opportunity 30%")
        fig_ch.update_layout(yaxis_title='Channel %', height=280, template='plotly_dark',
                             hovermode='x unified', margin=dict(l=0, r=0, t=30, b=0))
        fig_ch.update_yaxes(range=[0, 100])
        st.plotly_chart(fig_ch, use_container_width=True)

        # Levels & projections table
        st.subheader("📋 Current Levels & Projections")
        periods = {'Today': 0, '6M': 180, '1Y': 365, '2Y': 730, '3Y': 1095}
        tbl = {'Curve': ['Floor (Support)', 'Balanced (Fair Value)', 'Ceiling (Resistance)']}
        for pn, pd_ in periods.items():
            fd = TDAYS + pd_
            tbl[pn] = [
                f"${M['floor']['a'] * (fd ** M['floor']['b']):,.0f}",
                f"${M['balanced']['a'] * (fd ** M['balanced']['b']):,.0f}",
                f"${M['ceiling']['a'] * (fd ** M['ceiling']['b']):,.0f}",
            ]
        st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

        with st.expander("🔧 Model Parameters"):
            pc1, pc2 = st.columns(2)
            pc1.markdown(f"**Floor A**: a={M['floor']['a']:.4e}, b={M['floor']['b']:.4f}\n\n"
                         f"**Balanced**: a={M['balanced']['a']:.4e}, b={M['balanced']['b']:.4f}, "
                         f"k={M['balanced']['k']:.4f}")
            pc2.markdown(f"**Ceiling**: a={M['ceiling']['a']:.4e}, b={M['ceiling']['b']:.4f}\n\n"
                         f"**Max-Cross**: a={M['maxcross']['a']:.4e}, k={M['maxcross']['k']:.4f}, "
                         f"crossings={M['maxcross']['n_cross']}")

    # ================================================================
    # TAB 2 — RISK AREA
    # ================================================================
    with tab2:
        halflife = abs(int(np.log(0.5) / np.log(risk_lam)))
        st.subheader("⚠️ Risk Area Analysis")
        st.markdown(f"""
        Cumulative log-excess above floor with exponential decay 
        (λ = {risk_lam}, half-life ≈ **{halflife} days**).
        Higher values → market extended above floor for long → historically precedes corrections.
        
        **Current Risk: {CUR_RR*100:.0f}%** (vs rolling {risk_win}-day max)
        """)

        # Price + Risk chart
        fig_r = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                              subplot_titles=('Price & Floor', 'Risk Ratio (%)'))
        fig_r.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Price',
                                   line=dict(color='#F7931A', width=1.5)), row=1, col=1)
        fig_r.add_trace(go.Scatter(x=df['Date'], y=df['Floor'], name='Floor',
                                   line=dict(color='#28a745', width=1.5)), row=1, col=1)
        fig_r.add_trace(go.Scatter(x=df['Date'], y=df['RiskRatio']*100, name='Risk %',
                                   line=dict(color='#dc3545', width=1.5),
                                   fill='tozeroy', fillcolor='rgba(220,53,69,0.15)'), row=2, col=1)
        fig_r.add_hline(y=70, line_dash='dash', line_color='red', opacity=0.5, row=2, col=1)
        fig_r.add_hline(y=30, line_dash='dash', line_color='green', opacity=0.5, row=2, col=1)
        fig_r.update_layout(height=600, template='plotly_dark', hovermode='x unified',
                            margin=dict(l=0, r=0, t=40, b=0))
        fig_r.update_yaxes(type='log', row=1, col=1)
        fig_r.update_yaxes(range=[0, 100], title_text='Risk %', row=2, col=1)
        st.plotly_chart(fig_r, use_container_width=True)

        # Backtest
        st.subheader("🔬 Risk Area Backtest")
        st.markdown("Forward returns by risk bucket. Walk-forward: no lookahead bias.")
        with st.spinner("Running backtest…"):
            bt = backtest_risk_area(pr, fl, risk_lam, risk_win)
        if len(bt) > 0:
            for h in ['90d', '180d', '365d']:
                hdf = bt[bt['Horizon'] == h]
                if len(hdf) == 0:
                    continue
                st.markdown(f"#### {h} Forward Returns")
                st.dataframe(hdf.drop(columns='Horizon'), use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data for backtest.")

        # Calibration
        with st.expander("🔬 λ Calibration (slow ~30s)"):
            st.markdown("Find optimal λ by maximizing 6M return separation: low-risk vs high-risk.")
            if st.button("Run Calibration"):
                with st.spinner("Calibrating…"):
                    best_l, cal = calibrate_lambda(pr, fl, risk_win)
                st.success(f"**Best λ = {best_l:.4f}** (half-life ≈ {abs(int(np.log(0.5)/np.log(best_l)))} days)")
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(x=cal['lambda'], y=cal['separation'],
                                           mode='lines+markers', name='Separation'))
                fig_c.add_vline(x=best_l, line_dash='dash', line_color='green')
                fig_c.update_layout(xaxis_title='λ', yaxis_title='Separation', height=350,
                                    template='plotly_dark')
                st.plotly_chart(fig_c, use_container_width=True)

    # ================================================================
    # TAB 3 — HISTORICAL PROBABILITIES
    # ================================================================
    with tab3:
        st.subheader("📊 Historical Probabilities")
        st.markdown(f"From a channel position of **{CUR_CH*100:.1f}%** (±7%), what happened historically?")

        ptbl, n_sim = hist_prob_table(pr, chp, fl, bl, CUR_CH, tol=0.07)

        if ptbl is not None and len(ptbl) > 0:
            st.info(f"Found **{n_sim}** historical days with similar channel position.")
            piv = ptbl.pivot_table(index='Level', columns='Horizon', values='Prob', aggfunc='first')
            level_order = ['Floor', 'Balanced', '-10%', '-20%', '-30%', '-40%']
            piv = piv.reindex([l for l in level_order if l in piv.index])
            disp = piv.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
            st.markdown("#### Probability of Reaching Level Within Horizon")
            st.dataframe(disp, use_container_width=True)

            # Key takeaways
            st.markdown("---")
            st.markdown("#### 🎯 Key Takeaways")
            for lev in ['-20%', 'Balanced', 'Floor']:
                if lev in piv.index:
                    for hcol in piv.columns:
                        p = piv.loc[lev, hcol]
                        if pd.notna(p):
                            if p > 0.3:
                                st.warning(f"⚠️ **{lev}** reached in **{p*100:.0f}%** of cases within {hcol}")
                            elif p > 0.1:
                                st.info(f"ℹ️ **{lev}** reached in **{p*100:.0f}%** of cases within {hcol}")
                            break

            # Forward return stats from similar positions
            st.subheader("📈 Forward Returns from Similar Positions")
            max_h = 365
            sim_idx = np.where((np.abs(chp - CUR_CH) < 0.07) & (np.arange(len(pr)) < len(pr) - max_h))[0]
            if len(sim_idx) > 10:
                for h, hn in [(90, '90d'), (180, '6M'), (365, '1Y')]:
                    rets = np.array([pr[i+h]/pr[i]-1 for i in sim_idx if i+h < len(pr)])
                    if len(rets) > 5:
                        ca, cb, cc = st.columns(3)
                        ca.metric(f"Median ({hn})", f"{np.median(rets)*100:+.1f}%")
                        cb.metric(f"% Positive ({hn})", f"{(rets>0).mean()*100:.0f}%")
                        cc.metric(f"P25/P75 ({hn})",
                                  f"{np.percentile(rets,25)*100:+.0f}% / {np.percentile(rets,75)*100:+.0f}%")
        else:
            st.warning("Not enough similar historical days. Try adjusting start year.")

    # ================================================================
    # TAB 4 — ENTRY STRATEGY
    # ================================================================
    with tab4:
        st.subheader("🎯 Entry Strategy")
        sc1, sc2, sc3 = st.columns(3)
        budget = sc1.number_input("💰 Budget (€)", 100, 10_000_000, 10_000, 1000)
        hor_m  = sc2.selectbox("📅 Horizon", [12, 24, 36], index=1,
                               format_func=lambda x: f"{x} months")
        tgt_2y = M['balanced']['a'] * ((TDAYS + 730) ** M['balanced']['b'])
        tgt    = sc3.number_input("🎯 Target (2Y balanced)", 50000, 1_000_000, int(tgt_2y), 10000)

        st.markdown("---")

        # entry levels
        CUR_MC = float(M['maxcross']['a'] * (TDAYS ** M['maxcross']['b']))
        levels = {
            'Current Price': CUR_P,
            'Max-Crossing':  CUR_MC,
            'Balanced':      CUR_BL,
            '-20%':          CUR_P * 0.80,
            '-30%':          CUR_P * 0.70,
            'Floor':         CUR_FL,
        }

        # compute hist. prob for each level
        hor_d = int(hor_m * 30.44)
        sim_idx = np.where((np.abs(chp - CUR_CH) < 0.07) & (np.arange(len(pr)) < len(pr) - hor_d))[0]

        strat_rows = []
        for lname, lprice in levels.items():
            disc = (lprice / CUR_P - 1) * 100
            dd_need = lprice / CUR_P - 1
            if len(sim_idx) > 10:
                hits = sum(1 for i in sim_idx if np.min(pr[i:i+hor_d+1]) <= pr[i]*(1+dd_need))
                prob = hits / len(sim_idx)
            else:
                prob = float('nan')
            btc = budget / lprice if lprice > 0 else 0
            if np.isfinite(prob) and prob > 0:
                ev = prob * (btc * tgt) + (1 - prob) * budget
                ret = (ev / budget - 1) * 100
            else:
                ev = ret = float('nan')
            strat_rows.append({
                'Level': lname, 'Entry $': f"${lprice:,.0f}", 'Discount': f"{disc:+.1f}%",
                'Hist Prob': f"{prob*100:.1f}%" if np.isfinite(prob) else "—",
                'BTC': f"{btc:.6f}",
                'EV (€)': f"€{ev:,.0f}" if np.isfinite(ev) else "—",
                'Exp Ret': f"{ret:+.1f}%" if np.isfinite(ret) else "—",
            })
        st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)

        # DCA plan
        st.markdown("---")
        st.subheader("📅 Dynamic DCA Plan")
        st.markdown(f"Based on channel={CUR_CH*100:.0f}%, risk={CUR_RR*100:.0f}%:")

        if CUR_CH < 0.30:
            alloc = {'Immediate (lump)': 0.60, 'Reserve for dips': 0.40}
            note = "Low channel → heavier immediate allocation"
        elif CUR_CH < 0.50 and CUR_RR < 0.5:
            alloc = {'Immediate': 0.40, 'Weekly DCA': 0.30, 'Reserve': 0.30}
            note = "Mid channel, moderate risk → balanced"
        elif CUR_RR > 0.6:
            alloc = {'Immediate': 0.15, 'Weekly DCA': 0.35, 'Reserve': 0.50}
            note = "High risk → conservative, large reserve"
        else:
            alloc = {'Immediate': 0.30, 'Weekly DCA': 0.30, 'Reserve': 0.40}
            note = "Standard allocation"
        st.info(f"💡 **{note}**")

        dca_rows = []
        for comp, pct in alloc.items():
            amt = budget * pct
            if 'Immediate' in comp or 'lump' in comp:
                dca_rows.append({'Component': comp, '%': f"{pct*100:.0f}%",
                                 'Amount': f"€{amt:,.0f}",
                                 'Action': f"Buy now @ ${CUR_P:,.0f}",
                                 'BTC': f"{amt/CUR_P:.6f}"})
            elif 'Weekly' in comp:
                weeks = hor_m * 4
                dca_rows.append({'Component': comp, '%': f"{pct*100:.0f}%",
                                 'Amount': f"€{amt:,.0f}",
                                 'Action': f"€{amt/weeks:,.0f}/week × {weeks}w",
                                 'BTC': "varies"})
            else:
                dca_rows.append({'Component': comp, '%': f"{pct*100:.0f}%",
                                 'Amount': f"€{amt:,.0f}",
                                 'Action': "Deploy on -20%/-30%/Floor dips",
                                 'BTC': "conditional"})
        st.dataframe(pd.DataFrame(dca_rows), use_container_width=True, hide_index=True)

        # Return projections
        st.markdown("---")
        st.subheader("📈 Return Projections (from current price)")
        prows = []
        for y in [1, 2, 3, 4]:
            fd = TDAYS + y * 365
            ff = M['floor']['a'] * (fd ** M['floor']['b'])
            fb = M['balanced']['a'] * (fd ** M['balanced']['b'])
            fc = M['ceiling']['a'] * (fd ** M['ceiling']['b'])
            prows.append({
                'Horizon': f"{y}Y",
                'Floor': f"${ff:,.0f} ({(ff/CUR_P-1)*100:+.0f}%)",
                'Balanced': f"${fb:,.0f} ({(fb/CUR_P-1)*100:+.0f}%)",
                'Ceiling': f"${fc:,.0f} ({(fc/CUR_P-1)*100:+.0f}%)",
            })
        st.dataframe(pd.DataFrame(prows), use_container_width=True, hide_index=True)

        st.error("⚠️ Models can be wrong. Past ≠ future. **Only invest what you can afford to lose.**")

    # ================================================================
    # TAB 5 — MULTI-ASSET
    # ================================================================
    with tab5:
        st.subheader("🌍 Multi-Asset NLB Analysis")
        st.markdown("Compare BTC's NLB across denominators: USD, Gold (XAU), EUR.")

        models = {'BTC/USD': M}

        if load_gold:
            with st.spinner("Loading BTC/Gold…"):
                gdf = _fetch_cc('BTC', 'XAU', f'{start_year}-01-01')
                if gdf is not None and len(gdf) > 100:
                    models['BTC/XAU'] = build_model(gdf, 'BTC/XAU', risk_lam, risk_win)
                else:
                    st.warning("Could not load BTC/Gold data.")

        if load_eur:
            with st.spinner("Loading BTC/EUR…"):
                edf = _fetch_cc('BTC', 'EUR', f'{start_year}-01-01')
                if edf is not None and len(edf) > 100:
                    models['BTC/EUR'] = build_model(edf, 'BTC/EUR', risk_lam, risk_win)
                else:
                    st.warning("Could not load BTC/EUR data.")

        if len(models) == 1:
            st.info("Enable BTC/Gold or BTC/EUR in sidebar for multi-asset comparison.")

        # Charts for extra assets
        for aname, amod in models.items():
            if aname == 'BTC/USD':
                continue
            adf = amod['df']
            st.markdown(f"### {aname}")
            mc1, mc2, mc3, mc4 = st.columns(4)
            cp = float(adf['Close'].iloc[-1])
            unit = 'oz' if 'XAU' in aname else '€'
            mc1.metric("Price",    f"{cp:,.2f} {unit}")
            mc2.metric("Floor",    f"{adf['Floor'].iloc[-1]:,.2f} {unit}")
            mc3.metric("Balanced", f"{adf['Balanced'].iloc[-1]:,.2f} {unit}")
            mc4.metric("Channel",  f"{adf['ChPos'].iloc[-1]*100:.1f}%")

            fig_a = go.Figure()
            fig_a.add_trace(go.Scatter(x=adf['Date'], y=adf['Close'], name='Price',
                                       line=dict(color='#F7931A', width=1.5), opacity=0.7))
            fig_a.add_trace(go.Scatter(x=adf['Date'], y=adf['NLB'], name='NLB',
                                       line=dict(color='white', width=2)))
            fig_a.add_trace(go.Scatter(x=adf['Date'], y=adf['Floor'], name='Floor',
                                       line=dict(color='#28a745', width=1.5)))
            fig_a.add_trace(go.Scatter(x=adf['Date'], y=adf['Balanced'], name='Balanced',
                                       line=dict(color='#6c757d', width=1.5, dash='dash')))
            fig_a.add_trace(go.Scatter(x=adf['Date'], y=adf['Ceiling'], name='Ceiling',
                                       line=dict(color='rgba(220,53,69,0.4)', width=1)))
            fig_a.update_layout(yaxis_type='log', height=400, template='plotly_dark',
                                hovermode='x unified', margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_a, use_container_width=True)

        # Cross-asset comparison table
        if len(models) > 1:
            st.markdown("### 📊 Cross-Asset Comparison")
            crows = []
            for aname, amod in models.items():
                adf = amod['df']
                crows.append({
                    'Asset': aname,
                    'Channel': f"{adf['ChPos'].iloc[-1]*100:.1f}%",
                    'vs Floor': f"{(adf['Close'].iloc[-1]/adf['Floor'].iloc[-1]-1)*100:+.1f}%",
                    'vs Balanced': f"{(adf['Close'].iloc[-1]/adf['Balanced'].iloc[-1]-1)*100:+.1f}%",
                    'Risk': f"{adf['RiskRatio'].iloc[-1]*100:.0f}%",
                    'Floor b': f"{amod['floor']['b']:.4f}",
                })
            st.dataframe(pd.DataFrame(crows), use_container_width=True, hide_index=True)

            # Divergence analysis
            st.markdown("### 📉 Divergence Analysis")
            st.markdown("""
            If BTC/Gold channel is significantly lower than BTC/USD, it suggests BTC is 
            underperforming vs Gold — potential opportunity or warning depending on context.
            """)
            usd_ch = models['BTC/USD']['df']['ChPos'].iloc[-1]
            for aname in ['BTC/XAU', 'BTC/EUR']:
                if aname in models:
                    other_ch = models[aname]['df']['ChPos'].iloc[-1]
                    diff = (usd_ch - other_ch) * 100
                    if abs(diff) > 10:
                        st.warning(f"⚠️ **{aname}** channel diverges from USD by **{diff:+.1f}pp** — "
                                   f"USD={usd_ch*100:.1f}%, {aname}={other_ch*100:.1f}%")
                    else:
                        st.success(f"✅ **{aname}** aligned with USD (Δ={diff:+.1f}pp)")

    # ================================================================
    # FOOTER
    # ================================================================
    st.markdown("---")
    st.warning("""
    **⚠️ DISCLAIMER** — This tool is for informational/educational purposes only. NOT financial advice.
    Models based on historical data — no guarantee of future results. Crypto is volatile.
    **You may lose all capital.** Consult a financial advisor. Author assumes no responsibility.
    
    📊 Past ≠ future | 🔬 Research tool | 💡 DYOR
    """)
    st.caption(f"Data: CryptoCompare | Models: NLB Envelope, Balanced PL, Risk Area | "
               f"Updated: {TODAY.strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    main()
