"""
================================================================================
BITCOIN NLB (Never Look Back) ANALYSIS v2
================================================================================
BTC/USD only — Power Law Channel + Risk Area + Historical Probabilities
With debug data export and clear methodology explanations
================================================================================
"""
import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.optimize import minimize_scalar
import time, io

st.set_page_config(page_title="BTC NLB Analysis", page_icon="₿", layout="wide",
                   initial_sidebar_state="expanded")

if 'disclaimer_accepted' not in st.session_state:
    st.session_state.disclaimer_accepted = False

# ── DISCLAIMER ──
def show_disclaimer():
    st.markdown('<p style="font-size:2.5rem;font-weight:bold;color:#F7931A;text-align:center;">'
                '₿ Bitcoin NLB Analysis</p>', unsafe_allow_html=True)
    st.markdown("---")
    _, c, _ = st.columns([1, 3, 1])
    with c:
        st.error("""
        **⚠️ DISCLAIMER — READ BEFORE PROCEEDING**
        
        This tool is for **informational and educational purposes only**. NOT financial advice.
        Models based on historical data — **no guarantee of future results**.
        Crypto is highly volatile — **you may lose all invested capital**.
        Consult a qualified financial advisor. Author assumes **no responsibility** for losses.
        """)
        accept = st.checkbox("✅ I have read, understood and accept the conditions above")
        if st.button("🚀 ENTER", type="primary", disabled=not accept, use_container_width=True):
            st.session_state.disclaimer_accepted = True
            st.rerun()

if not st.session_state.disclaimer_accepted:
    show_disclaimer()
    st.stop()

# ── CSS ──
st.markdown("""<style>
.hdr{font-size:2.2rem;font-weight:bold;color:#F7931A;text-align:center;padding:0.3rem;}
.sub{text-align:center;color:#888;margin-bottom:1rem;font-size:0.95rem;}
.sig-green{background:linear-gradient(135deg,#0f5132,#198754);color:white;padding:1rem;border-radius:10px;text-align:center;font-size:1.2rem;font-weight:bold;}
.sig-yellow{background:linear-gradient(135deg,#664d03,#ffc107);color:black;padding:1rem;border-radius:10px;text-align:center;font-size:1.2rem;font-weight:bold;}
.sig-orange{background:linear-gradient(135deg,#7c2d12,#ea580c);color:white;padding:1rem;border-radius:10px;text-align:center;font-size:1.2rem;font-weight:bold;}
.sig-red{background:linear-gradient(135deg,#7f1d1d,#dc2626);color:white;padding:1rem;border-radius:10px;text-align:center;font-size:1.2rem;font-weight:bold;}
.sig-gray{background:linear-gradient(135deg,#495057,#6c757d);color:white;padding:1rem;border-radius:10px;text-align:center;font-size:1.2rem;font-weight:bold;}
.explain{background:#1a1a2e;border-left:4px solid #F7931A;padding:1rem;border-radius:5px;margin:0.5rem 0;font-size:0.9rem;}
</style>""", unsafe_allow_html=True)

GENESIS = pd.Timestamp('2009-01-03')

# ============================================================================
# DATA
# ============================================================================
@st.cache_data(ttl=3600)
def load_btc(start_date='2010-07-18'):
    """Download BTC/USD daily OHLC from CryptoCompare with pagination."""
    all_data = []
    cur_ts = int(datetime.now().timestamp())
    start_ts = int(pd.Timestamp(start_date).timestamp())
    while cur_ts > start_ts:
        r = requests.get('https://min-api.cryptocompare.com/data/v2/histoday',
                         params={'fsym': 'BTC', 'tsym': 'USD', 'limit': 2000, 'toTs': cur_ts},
                         timeout=30)
        js = r.json()
        if js.get('Response') != 'Success':
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
# NLB COMPUTATION
# ============================================================================
def compute_nlb(series):
    """
    NLB (Never Look Back) price:
    1. forward_min[t] = min of all prices from t onward
    2. nlb[t] = cumulative max of forward_min
    
    Result: monotone non-decreasing curve. The highest price that
    Bitcoin never went back below permanently.
    
    Returns a staircase-shaped series.
    """
    s = series.to_numpy(dtype=float)
    n = len(s)
    # Step 1: forward minimum (for each day, the lowest future price)
    fwd_min = np.empty(n)
    fwd_min[-1] = s[-1]
    for i in range(n - 2, -1, -1):
        fwd_min[i] = min(s[i], fwd_min[i + 1])
    # Step 2: cumulative max of forward_min
    nlb = np.maximum.accumulate(fwd_min)
    return pd.Series(nlb, index=series.index)

# ============================================================================
# CURVE FITTING
# ============================================================================
def fit_floor_envelope(days, nlb, b_min=4.0, b_max=7.5, b_steps=4000):
    """
    FLOOR A — Absolute support envelope.
    
    Fits a power law y = a × x^b that stays BELOW all NLB update points,
    minimizing the total gap in log-space (tightest possible lower envelope).
    
    Method:
    - Only uses NLB "update" points (days when NLB increases — the staircase rises)
    - Grid search on exponent b ∈ [4.0, 7.5]
    - For each b, finds maximum feasible intercept c = min(log(y) - b*log(x))
    - Picks b that minimizes total gap = Σ(log(y) - (c + b*log(x)))
    
    This is the most conservative curve: floor ≤ NLB always by construction.
    """
    # NLB update points only (where staircase rises)
    upd = np.zeros(len(nlb), dtype=bool)
    upd[0] = True
    upd[1:] = nlb[1:] > nlb[:-1]
    x = days[upd].astype(float)
    y = nlb[upd].astype(float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    logx, logy = np.log(x), np.log(y)
    
    best = None
    for b in np.linspace(b_min, b_max, b_steps):
        c = float(np.min(logy - b * logx))  # max intercept that stays below all points
        gap = float(np.sum(logy - (c + b * logx)))  # total gap in log-space
        if best is None or gap < best['gap']:
            best = {'a': float(np.exp(c)), 'b': float(b), 'gap': gap}
    
    # Count how many NLB points used
    best['n_update_points'] = int(mask.sum())
    return best


def fit_balanced(days, nlb, a_floor, b_floor):
    """
    BALANCED — Fair value line.
    
    Shifts the Floor upward by a factor e^k until the log-space area
    above the curve equals the area below (among NLB update points).
    
    Same exponent b as floor, only coefficient a changes:
        a_balanced = a_floor × e^k
    
    Represents the "median" of the NLB in log-space.
    """
    upd = np.zeros(len(nlb), dtype=bool)
    upd[0] = True
    upd[1:] = nlb[1:] > nlb[:-1]
    x = days[upd].astype(float)
    y = nlb[upd].astype(float)
    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    log_floor = np.log(a_floor * (x ** b_floor))
    log_y = np.log(y)
    
    def area_imbalance(k):
        diff = log_y - (log_floor + k)
        above = float(np.sum(np.maximum(0, diff)))
        below = float(np.sum(np.maximum(0, -diff)))
        return abs(above - below)
    
    res = minimize_scalar(area_imbalance, bounds=(-2, 3), method='bounded')
    k = float(res.x)
    return {'a': a_floor * np.exp(k), 'b': b_floor, 'k': k,
            'area_imbalance': float(res.fun)}


def fit_max_crossing(days, nlb, a_floor, b_floor, k_steps=2000):
    """
    MAX-CROSSING — Dynamic support/resistance.
    
    Finds the shift k that maximizes the number of times the NLB
    crosses the curve. Acts as a mean-reverting level.
    
    a_maxcross = a_floor × e^k
    """
    x = days.astype(float)
    y = nlb.astype(float)
    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    log_floor = np.log(a_floor * (x ** b_floor))
    log_y = np.log(y)
    
    best_k, best_c = 0.0, 0
    for k in np.linspace(-0.5, 1.5, k_steps):
        diff_sign = np.sign(log_y - (log_floor + k))
        # Handle zeros (on the curve)
        for i in range(1, len(diff_sign)):
            if diff_sign[i] == 0:
                diff_sign[i] = diff_sign[i - 1]
        crossings = int(np.sum(diff_sign[1:] != diff_sign[:-1]))
        if crossings > best_c:
            best_c, best_k = crossings, float(k)
    
    return {'a': a_floor * np.exp(best_k), 'b': b_floor, 'k': best_k,
            'n_crossings': best_c}

# ============================================================================
# RISK AREA SYSTEM — Exponential Decay
# ============================================================================
#
# KEY IDEA: When price stays above the NLB floor, "risk energy" accumulates.
# The longer and higher it stays, the more likely a correction.
#
# FORMULA:
#   excess(t) = max(0, log(price_t / floor_t))
#   area(t)   = area(t-1) × λ + excess(t)
#   risk(t)   = area(t) / rolling_max(area, window)
#
# WHY DECAY (not hard reset):
#   - Hard reset (area→0 when touching floor) is too binary
#   - A 1-day floor touch doesn't "discharge" all risk
#   - Decay (λ=0.997 ≈ 231-day half-life) is more realistic:
#     area reduces gradually, recent excess matters more
#
# WHY ROLLING MAX NORMALIZATION:
#   - Raw area grows with time (floor rises → excess grows)
#   - Dividing by rolling max gives 0-1 ratio comparable across epochs
#   - Walk-forward: only uses past data at each point
#
# CALIBRATION:
#   - λ is optimized to maximize separation of 6M forward returns
#     between high-risk (>70%) and low-risk (<30%) days
# ============================================================================

def compute_risk_area(price, floor_v, lam=0.997):
    """
    Decayed cumulative log-excess above floor.
    Returns raw area values (not normalized).
    """
    n = len(price)
    excess = np.maximum(0.0, np.log(np.maximum(price / floor_v, 1e-12)))
    area = np.empty(n)
    area[0] = excess[0]
    for t in range(1, n):
        area[t] = area[t - 1] * lam + excess[t]
    return area, excess  # return excess too for debug


def compute_risk_ratio(area, window=365):
    """Normalize to [0,1] using walk-forward rolling max."""
    n = len(area)
    risk = np.zeros(n)
    rolling_max = np.zeros(n)
    for t in range(n):
        rm = float(np.max(area[max(0, t - window):t + 1]))
        rolling_max[t] = rm
        if rm > 0:
            risk[t] = area[t] / rm
    return np.clip(risk, 0, 1), rolling_max


def backtest_risk_area(price, floor_v, lam, rw, horizons=(30, 90, 180, 365)):
    """
    Walk-forward backtest of Risk Area indicator.
    
    For each historical day:
      1. Compute risk ratio using ONLY past data (no lookahead)
      2. Group into risk buckets (quintiles)
      3. Measure forward return and max drawdown over each horizon
    
    This answers: "When risk was high, what typically happened next?"
    """
    n = len(price)
    area, _ = compute_risk_area(price, floor_v, lam)
    risk, _ = compute_risk_ratio(area, rw)
    
    buckets = [
        ('0-20% Very Low',  0.0,  0.2),
        ('20-40% Low',      0.2,  0.4),
        ('40-60% Mid',      0.4,  0.6),
        ('60-80% High',     0.6,  0.8),
        ('80-100% V.High',  0.8,  1.01),
    ]
    rows = []
    for h in horizons:
        # Pre-compute forward returns for all days (vectorized where possible)
        fwd_ret = np.full(n, np.nan)
        fwd_dd = np.full(n, np.nan)
        for t in range(n - h):
            window = price[t:t + h + 1]
            fwd_ret[t] = window[-1] / window[0] - 1
            fwd_dd[t] = float((window / np.maximum.accumulate(window) - 1).min())
        
        for bname, lo, hi in buckets:
            mask = (risk >= lo) & (risk < hi) & np.isfinite(fwd_ret)
            cnt = int(mask.sum())
            if cnt < 15:
                continue
            r = fwd_ret[mask]
            d = fwd_dd[mask]
            rows.append({
                'Risk Bucket': bname, 'Horizon': f"{h}d", 'N': cnt,
                'Median Ret': f"{np.median(r)*100:+.1f}%",
                'Mean Ret': f"{np.mean(r)*100:+.1f}%",
                '% Positive': f"{(r > 0).mean()*100:.0f}%",
                'Med MaxDD': f"{np.median(d)*100:.1f}%",
                'P10 MaxDD': f"{np.percentile(d, 10)*100:.1f}%",
            })
    return pd.DataFrame(rows)


def calibrate_lambda(price, floor_v, rw=365):
    """
    Find optimal λ: maximizes separation of 6-month forward returns
    between high-risk (>70%) and low-risk (<30%) days.
    """
    n = len(price)
    fwd180 = np.full(n, np.nan)
    for t in range(n - 180):
        fwd180[t] = price[t + 180] / price[t] - 1
    
    best_lam, best_sep = 0.997, 0.0
    rows = []
    for lam in np.arange(0.990, 0.9996, 0.0005):
        area, _ = compute_risk_area(price, floor_v, lam)
        risk, _ = compute_risk_ratio(area, rw)
        hi_mask = (risk > 0.7) & np.isfinite(fwd180)
        lo_mask = (risk < 0.3) & np.isfinite(fwd180)
        if hi_mask.sum() > 20 and lo_mask.sum() > 20:
            sep = float(np.median(fwd180[lo_mask]) - np.median(fwd180[hi_mask]))
        else:
            sep = 0.0
        rows.append({'lambda': float(lam), 'separation': sep,
                     'n_high': int(hi_mask.sum()), 'n_low': int(lo_mask.sum())})
        if sep > best_sep:
            best_sep, best_lam = sep, float(lam)
    return best_lam, pd.DataFrame(rows)

# ============================================================================
# CHANNEL POSITION & HISTORICAL PROBABILITIES
# ============================================================================
def channel_pos(nlb, floor_v, ceil_v):
    """
    Where is NLB within the Floor-Ceiling channel?
    0% = at floor (cheap), 100% = at ceiling (expensive).
    Computed in log-space for proper scaling.
    """
    ln = np.log(np.maximum(nlb, 1e-12))
    lf = np.log(np.maximum(floor_v, 1e-12))
    lc = np.log(np.maximum(ceil_v, 1e-12))
    width = lc - lf
    return np.clip(np.where(width > 0, (ln - lf) / width, 0.5), 0, 1)


def hist_prob_table(price, ch_pos, floor_v, bal_v, current_ch,
                    tol=0.07, horizons=(30, 60, 90, 180, 365)):
    """
    Historical probability analysis.
    
    Method:
    1. Find all historical days where channel position was within ±tol
       of the current position
    2. For each such day, track what happened in the next H days
    3. Count how often price reached each target level
    
    Targets: Floor, Balanced, -10%, -20%, -30%, -40% from entry
    
    This is fully empirical — no model assumptions, just counting.
    """
    n = len(price)
    max_h = max(horizons)
    # Find similar days (but exclude last max_h days — need future data)
    similar = np.where((np.abs(ch_pos - current_ch) < tol) &
                       (np.arange(n) < n - max_h))[0]
    if len(similar) < 10:
        return None, 0
    
    targets = {'Floor': 'floor', 'Balanced': 'balanced',
               '-10%': -0.10, '-20%': -0.20, '-30%': -0.30, '-40%': -0.40}
    rows = []
    for h in horizons:
        valid = similar[similar < n - h]
        if len(valid) < 5:
            continue
        for tname, tdef in targets.items():
            hits = 0
            for idx in valid:
                future_min = float(np.min(price[idx:idx + h + 1]))
                if tdef == 'floor':
                    target = floor_v[idx]
                elif tdef == 'balanced':
                    target = bal_v[idx]
                else:
                    target = price[idx] * (1 + tdef)
                if future_min <= target:
                    hits += 1
            rows.append({'Horizon': f"{h}d", 'Level': tname,
                         'Prob': hits / len(valid), 'N': len(valid)})
    return (pd.DataFrame(rows) if rows else None), len(similar)


# ============================================================================
# MODEL BUILDER
# ============================================================================
def build_model(df_raw, lam=0.997, rw=365):
    """
    Build complete NLB model from raw OHLC data.
    
    Steps:
    1. Compute NLB from daily Low prices (most conservative)
    2. Fit Floor envelope (power law under all NLB updates)
    3. Fit Balanced (equal area shift)
    4. Compute Ceiling (symmetric to floor vs balanced)
    5. Fit Max-Crossing curve
    6. Compute Risk Area with decay
    7. Compute Channel Position
    """
    df = df_raw.copy()
    df['Days'] = (df['Date'] - GENESIS).dt.days
    
    # NLB on Low prices (most conservative — captures intraday dips)
    df['NLB'] = compute_nlb(df['Low'])
    df = df[df['Days'] > 0].copy().reset_index(drop=True)
    
    days = df['Days'].to_numpy(float)
    nlb = df['NLB'].to_numpy(float)
    price = df['Close'].to_numpy(float)
    
    # Fit curves
    floor_p = fit_floor_envelope(days, nlb)
    bal_p = fit_balanced(days, nlb, floor_p['a'], floor_p['b'])
    mc_p = fit_max_crossing(days, nlb, floor_p['a'], floor_p['b'])
    ceil_a = bal_p['a'] * np.exp(bal_p['k'])  # symmetric: ceiling/balanced = balanced/floor
    
    # Apply curves to data
    df['Floor']    = floor_p['a'] * (days ** floor_p['b'])
    df['Balanced'] = bal_p['a']   * (days ** bal_p['b'])
    df['Ceiling']  = ceil_a       * (days ** bal_p['b'])
    df['MaxCross'] = mc_p['a']    * (days ** mc_p['b'])
    
    # Channel position
    df['ChPos'] = channel_pos(nlb, df['Floor'].values, df['Ceiling'].values)
    
    # Risk area
    risk_area, excess = compute_risk_area(price, df['Floor'].values, lam)
    risk_ratio, risk_rmax = compute_risk_ratio(risk_area, rw)
    df['Excess']     = excess       # daily log-excess (debug)
    df['RiskArea']   = risk_area    # decayed cumulative (debug)
    df['RiskRMax']   = risk_rmax    # rolling max (debug)
    df['RiskRatio']  = risk_ratio   # normalized 0-1
    
    return {
        'df': df,
        'floor': floor_p,
        'balanced': bal_p,
        'ceiling': {'a': ceil_a, 'b': bal_p['b']},
        'maxcross': mc_p,
    }


def project_curves(model, years=5):
    """Project floor/balanced/ceiling into the future."""
    last_date = model['df']['Date'].iloc[-1]
    last_days = model['df']['Days'].iloc[-1]
    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=years * 365, freq='D')
    fdays = np.array([(d - GENESIS).days for d in dates], dtype=float)
    return pd.DataFrame({
        'Date': dates, 'Days': fdays,
        'Floor':    model['floor']['a']    * (fdays ** model['floor']['b']),
        'Balanced': model['balanced']['a'] * (fdays ** model['balanced']['b']),
        'Ceiling':  model['ceiling']['a']  * (fdays ** model['ceiling']['b']),
    })


# ============================================================================
# SIGNAL
# ============================================================================
def get_signal(ch, rr):
    """Market signal based on channel position and risk ratio."""
    if ch < 0.15:
        return ('STRONG OPPORTUNITY', '🟢🟢',
                f'Very low in channel ({ch*100:.0f}%). Historically rare.', 'sig-green')
    elif ch < 0.30:
        return ('OPPORTUNITY', '🟢',
                f'Low in channel ({ch*100:.0f}%). Historically favorable entry zone.', 'sig-green')
    elif ch < 0.50 and rr < 0.5:
        return ('ACCUMULATE', '🟡',
                f'Mid-low channel ({ch*100:.0f}%), moderate risk ({rr*100:.0f}%).', 'sig-yellow')
    elif ch > 0.85:
        return ('HIGH RISK', '🔴',
                f'Very high in channel ({ch*100:.0f}%). Extended territory.', 'sig-red')
    elif ch > 0.70 and rr > 0.7:
        return ('CAUTION', '🟠',
                f'High channel ({ch*100:.0f}%) + elevated risk ({rr*100:.0f}%).', 'sig-orange')
    elif ch < 0.70:
        return ('NEUTRAL', '⚪',
                f'Mid channel ({ch*100:.0f}%). No strong directional signal.', 'sig-gray')
    else:
        return ('WATCH', '🟡',
                f'Upper channel ({ch*100:.0f}%). Monitor risk indicators.', 'sig-yellow')

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.markdown('<p class="hdr">₿ Bitcoin NLB Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub">Never Look Back Price — Power Law Channel & Risk Area</p>',
                unsafe_allow_html=True)

    # ── SIDEBAR ──────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("---")
        proj_years = st.slider("📐 Projection (years)", 1, 5, 3)
        st.markdown("---")
        st.markdown("### 🔬 Risk Area Parameters")
        risk_lam = st.slider("Decay λ", 0.990, 0.999, 0.997, 0.001,
                              help="Exponential decay factor. Higher = longer memory. "
                                   "0.997 ≈ 231-day half-life, 0.995 ≈ 139 days.")
        risk_win = st.slider("Normalization window (days)", 180, 730, 365, 30,
                              help="Rolling window for max area normalization.")
        st.markdown("---")
        st.markdown("### 📚 What Are the NLB Curves?")
        st.markdown("""
        **NLB** = Never Look Back price. The highest price BTC never 
        permanently went below. A staircase that only goes up.
        
        **Floor** = tightest power law *under* all NLB steps.
        Absolute historical support.
        
        **Balanced** = Floor shifted up until area above = area below 
        (in log-space). Acts as "fair value".
        
        **Ceiling** = symmetric to Floor relative to Balanced. 
        Upper boundary of the channel.
        
        **Max-Cross** = shift that maximizes NLB crossings. 
        Dynamic support/resistance level.
        """)
        st.markdown("---")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    # ── LOAD & BUILD ─────────────────────────────────────────────────
    with st.spinner("📥 Loading BTC data from CryptoCompare…"):
        btc_raw = load_btc('2010-07-18')
        if btc_raw is None or len(btc_raw) < 100:
            st.error("❌ Failed to load BTC data. Check internet connection.")
            return

    with st.spinner("🔧 Building NLB model (fitting curves, computing risk)…"):
        M = build_model(btc_raw, lam=risk_lam, rw=risk_win)

    df   = M['df']
    days = df['Days'].to_numpy(float)
    pr   = df['Close'].to_numpy(float)
    nlb  = df['NLB'].to_numpy(float)
    fl_v = df['Floor'].to_numpy(float)
    bl_v = df['Balanced'].to_numpy(float)
    cl_v = df['Ceiling'].to_numpy(float)
    chp  = df['ChPos'].to_numpy(float)
    rr   = df['RiskRatio'].to_numpy(float)

    # Current values
    TODAY  = df['Date'].iloc[-1]
    TDAYS  = float(days[-1])
    P      = float(pr[-1])
    NLB_V  = float(nlb[-1])
    FL     = float(fl_v[-1])
    BL     = float(bl_v[-1])
    CL     = float(cl_v[-1])
    CH     = float(chp[-1])
    RR     = float(rr[-1])
    D_FL   = (P / FL - 1) * 100   # % distance price vs floor
    D_BL   = (P / BL - 1) * 100   # % distance price vs balanced

    sig_name, sig_emo, sig_desc, sig_css = get_signal(CH, RR)

    # ── TOP METRICS ──────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Price", f"${P:,.0f}",
              f"{(P / pr[-2] - 1)*100:+.2f}%" if len(pr) > 1 else "")
    c2.metric("🔒 NLB Floor", f"${FL:,.0f}", f"Price is {D_FL:+.1f}% above")
    c3.metric("⚖️ Balanced", f"${BL:,.0f}", f"Price is {D_BL:+.1f}%")
    c4.metric("📊 Channel Position", f"{CH*100:.1f}%",
              "0% = Floor, 100% = Ceiling")
    c5.metric("⚠️ Risk Ratio", f"{RR*100:.0f}%",
              "Low < 30% • High > 70%")

    st.markdown("---")
    st.markdown(f'<div class="{sig_css}">{sig_emo} {sig_name}<br>'
                f'<span style="font-size:0.85rem;font-weight:normal;">{sig_desc}</span></div>',
                unsafe_allow_html=True)
    st.markdown("")

    # ── TABS ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 NLB Channel", "⚠️ Risk Area", "📊 Probabilities",
        "🎯 Entry Strategy", "📦 Debug Data"])

    # ================================================================
    # TAB 1 — NLB CHANNEL
    # ================================================================
    with tab1:
        st.markdown('<div class="explain">'
                    '<b>What you see:</b> BTC price (orange), NLB staircase (white), '
                    'and the power law channel: Floor (green, absolute support), '
                    'Balanced (gray dashed, fair value), Ceiling (red, upper boundary). '
                    'Dotted lines = projections. The NLB is a staircase because it only rises '
                    'when BTC establishes a new "never look back" level.'
                    '</div>', unsafe_allow_html=True)

        proj = project_curves(M, proj_years)

        fig = go.Figure()
        # Price as thin line
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Close'], name='BTC Price',
            line=dict(color='#F7931A', width=1.2), opacity=0.6))
        # NLB as thick staircase
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['NLB'], name='NLB (staircase)',
            line=dict(color='white', width=2.5, shape='hv')))  # hv = horizontal-then-vertical = staircase
        # Channel
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Ceiling'], name='Ceiling',
            line=dict(color='rgba(220,53,69,0.35)', width=1)))
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Floor'], name='Floor',
            line=dict(color='#28a745', width=2),
            fill='tonexty', fillcolor='rgba(247,147,26,0.04)'))
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Balanced'], name='Balanced (Fair Value)',
            line=dict(color='#adb5bd', width=1.5, dash='dash')))
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MaxCross'], name='Max-Crossing',
            line=dict(color='#17a2b8', width=1, dash='dot'), visible='legendonly'))
        # Projections
        for col, clr in [('Floor', 'rgba(40,167,69,0.35)'),
                          ('Balanced', 'rgba(173,181,189,0.35)'),
                          ('Ceiling', 'rgba(220,53,69,0.2)')]:
            fig.add_trace(go.Scatter(
                x=proj['Date'], y=proj[col], showlegend=False,
                line=dict(color=clr, width=1, dash='dot')))

        y_hi = max(float(pr.max()), float(proj['Ceiling'].max())) * 2
        y_lo = max(float(fl_v.min()), 0.01)
        fig.update_layout(
            yaxis_type='log', yaxis_title='USD (log scale)', height=580,
            hovermode='x unified', template='plotly_dark',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font_size=11),
            margin=dict(l=0, r=0, t=30, b=0))
        fig.update_yaxes(range=[np.log10(y_lo), np.log10(y_hi)])
        st.plotly_chart(fig, use_container_width=True)

        # Channel position over time
        st.markdown("#### 📊 Channel Position History")
        st.markdown('<div class="explain">'
                    'Where is NLB inside the Floor-Ceiling channel over time? '
                    '0% = at floor (historically cheap), 100% = at ceiling (expensive). '
                    'Green zone (< 30%) = opportunity. Red zone (> 70%) = caution.'
                    '</div>', unsafe_allow_html=True)

        fig_ch = go.Figure()
        fig_ch.add_trace(go.Scatter(
            x=df['Date'], y=df['ChPos'] * 100, name='Channel %',
            line=dict(color='#F7931A', width=1.5),
            fill='tozeroy', fillcolor='rgba(247,147,26,0.08)'))
        fig_ch.add_hrect(y0=0, y1=30, fillcolor='rgba(40,167,69,0.08)', line_width=0)
        fig_ch.add_hrect(y0=70, y1=100, fillcolor='rgba(220,53,69,0.08)', line_width=0)
        fig_ch.add_hline(y=70, line_dash='dash', line_color='red', opacity=0.4)
        fig_ch.add_hline(y=30, line_dash='dash', line_color='green', opacity=0.4)
        fig_ch.update_layout(
            yaxis_title='Channel Position (%)', height=280, template='plotly_dark',
            hovermode='x unified', margin=dict(l=0, r=0, t=10, b=0))
        fig_ch.update_yaxes(range=[0, 100])
        st.plotly_chart(fig_ch, use_container_width=True)

        # Levels table
        st.markdown("#### 📋 Current Levels & Projections")
        periods = {'Today': 0, '6 Months': 180, '1 Year': 365, '2 Years': 730, '3 Years': 1095}
        trows = []
        for curve, params in [('Floor', M['floor']), ('Balanced', M['balanced']),
                               ('Ceiling', M['ceiling'])]:
            row = {'Curve': curve}
            for pn, pd_ in periods.items():
                fd = TDAYS + pd_
                val = params['a'] * (fd ** params['b'])
                if pd_ == 0:
                    row[pn] = f"${val:,.0f}"
                else:
                    ret = (val / P - 1) * 100
                    row[pn] = f"${val:,.0f} ({ret:+.0f}%)"
            trows.append(row)
        st.dataframe(pd.DataFrame(trows), use_container_width=True, hide_index=True)

        # Parameters
        with st.expander("🔧 Model Parameters"):
            st.markdown(f"""
            | Curve | a | b | k (shift) | Note |
            |-------|---|---|-----------|------|
            | Floor | {M['floor']['a']:.6e} | {M['floor']['b']:.4f} | — | {M['floor']['n_update_points']} NLB update points |
            | Balanced | {M['balanced']['a']:.6e} | {M['balanced']['b']:.4f} | {M['balanced']['k']:.4f} | Area imbalance: {M['balanced']['area_imbalance']:.4f} |
            | Ceiling | {M['ceiling']['a']:.6e} | {M['ceiling']['b']:.4f} | {2*M['balanced']['k']:.4f} | Symmetric |
            | Max-Cross | {M['maxcross']['a']:.6e} | {M['maxcross']['b']:.4f} | {M['maxcross']['k']:.4f} | {M['maxcross']['n_crossings']} crossings |
            """)

    # ================================================================
    # TAB 2 — RISK AREA
    # ================================================================
    with tab2:
        halflife = abs(int(np.log(0.5) / np.log(risk_lam)))
        st.markdown('<div class="explain">'
                    '<b>Risk Area</b> measures how long and how far price has been above the NLB floor. '
                    'Think of it like a spring: the more it\'s stretched (price above floor for longer), '
                    'the more likely a snap-back (correction).<br><br>'
                    f'<b>Formula:</b> area(t) = area(t-1) × {risk_lam} + max(0, log(price/floor))<br>'
                    f'<b>Decay half-life:</b> {halflife} days (older excess fades exponentially)<br>'
                    f'<b>Risk Ratio:</b> area / rolling {risk_win}-day max → normalized 0-1<br><br>'
                    '<b>Interpretation:</b> Risk > 70% = historically precedes corrections. '
                    'Risk < 30% = historically favorable.'
                    '</div>', unsafe_allow_html=True)
        st.markdown("")

        # 3-panel chart: price, raw risk area, risk ratio
        fig_r = make_subplots(
            rows=3, cols=1, shared_xaxes=True, row_heights=[0.45, 0.25, 0.30],
            subplot_titles=('Price vs Floor', 'Decayed Risk Area (raw)', 'Risk Ratio (0-100%)'),
            vertical_spacing=0.06)

        fig_r.add_trace(go.Scatter(
            x=df['Date'], y=df['Close'], name='Price',
            line=dict(color='#F7931A', width=1.5)), row=1, col=1)
        fig_r.add_trace(go.Scatter(
            x=df['Date'], y=df['Floor'], name='Floor',
            line=dict(color='#28a745', width=1.5)), row=1, col=1)

        fig_r.add_trace(go.Scatter(
            x=df['Date'], y=df['RiskArea'], name='Risk Area',
            line=dict(color='#6f42c1', width=1.2),
            fill='tozeroy', fillcolor='rgba(111,66,193,0.1)'), row=2, col=1)
        fig_r.add_trace(go.Scatter(
            x=df['Date'], y=df['RiskRMax'], name='Rolling Max',
            line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot')), row=2, col=1)

        fig_r.add_trace(go.Scatter(
            x=df['Date'], y=df['RiskRatio'] * 100, name='Risk %',
            line=dict(color='#dc3545', width=1.5),
            fill='tozeroy', fillcolor='rgba(220,53,69,0.12)'), row=3, col=1)
        fig_r.add_hrect(y0=70, y1=100, fillcolor='rgba(220,53,69,0.1)',
                        line_width=0, row=3, col=1)
        fig_r.add_hrect(y0=0, y1=30, fillcolor='rgba(40,167,69,0.1)',
                        line_width=0, row=3, col=1)
        fig_r.add_hline(y=70, line_dash='dash', line_color='red', opacity=0.5, row=3, col=1)
        fig_r.add_hline(y=30, line_dash='dash', line_color='green', opacity=0.5, row=3, col=1)

        fig_r.update_layout(height=700, template='plotly_dark', hovermode='x unified',
                            margin=dict(l=0, r=0, t=40, b=0), showlegend=True)
        fig_r.update_yaxes(type='log', title_text='USD', row=1, col=1)
        fig_r.update_yaxes(title_text='Area', row=2, col=1)
        fig_r.update_yaxes(range=[0, 100], title_text='Risk %', row=3, col=1)
        st.plotly_chart(fig_r, use_container_width=True)

        # Backtest
        st.markdown("#### 🔬 Risk Area Backtest")
        st.markdown('<div class="explain">'
                    'For each historical day, we compute the risk ratio using <b>only past data</b> '
                    '(no future information — walk-forward). Then we group by risk quintile and '
                    'measure what the forward return and max drawdown were. '
                    'If the indicator works, high-risk days should show worse forward returns.'
                    '</div>', unsafe_allow_html=True)

        with st.spinner("Running walk-forward backtest…"):
            bt = backtest_risk_area(pr, fl_v, risk_lam, risk_win)

        if len(bt) > 0:
            h_sel = st.selectbox("Horizon", ['90d', '180d', '365d'], index=1,
                                  key='bt_horizon')
            h_data = bt[bt['Horizon'] == h_sel]
            if len(h_data) > 0:
                st.dataframe(h_data.drop(columns='Horizon'),
                             use_container_width=True, hide_index=True)
            else:
                st.info("Not enough data for this horizon.")
        else:
            st.warning("Not enough data for backtest.")

        # Lambda calibration
        with st.expander("🔬 λ Calibration (click to run — takes ~30s)"):
            st.markdown("Searches λ ∈ [0.990, 0.999] to find the value that best separates "
                        "6-month forward returns between high-risk and low-risk regimes.")
            if st.button("Run Calibration", key='cal_btn'):
                with st.spinner("Testing 20 λ values…"):
                    best_l, cal_df = calibrate_lambda(pr, fl_v, risk_win)
                hl = abs(int(np.log(0.5) / np.log(best_l)))
                st.success(f"**Best λ = {best_l:.4f}** (half-life ≈ {hl} days)")
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(
                    x=cal_df['lambda'], y=cal_df['separation'],
                    mode='lines+markers', name='Separation',
                    line=dict(color='#17a2b8')))
                fig_c.add_vline(x=best_l, line_dash='dash', line_color='#28a745')
                fig_c.update_layout(
                    xaxis_title='λ (decay factor)',
                    yaxis_title='Separation (low-risk median − high-risk median)',
                    height=350, template='plotly_dark')
                st.plotly_chart(fig_c, use_container_width=True)

    # ================================================================
    # TAB 3 — HISTORICAL PROBABILITIES
    # ================================================================
    with tab3:
        st.markdown('<div class="explain">'
                    '<b>Method:</b> We find all historical days when the channel position '
                    f'was similar to today ({CH*100:.1f}% ± 7%). For each such day, we track '
                    'what the minimum price was over the next 30/60/90/180/365 days. '
                    'Then we count how often each target level was reached.<br><br>'
                    'This is <b>purely empirical</b> — no model assumptions, just historical counting. '
                    'It answers: "From similar positions, how often did price drop to X?"'
                    '</div>', unsafe_allow_html=True)
        st.markdown("")

        prob_df, n_sim = hist_prob_table(pr, chp, fl_v, bl_v, CH, tol=0.07)

        if prob_df is not None and len(prob_df) > 0:
            st.info(f"Found **{n_sim}** historical days with channel position "
                    f"between {(CH-0.07)*100:.0f}% and {(CH+0.07)*100:.0f}%.")

            piv = prob_df.pivot_table(index='Level', columns='Horizon',
                                       values='Prob', aggfunc='first')
            order = ['Floor', 'Balanced', '-10%', '-20%', '-30%', '-40%']
            piv = piv.reindex([l for l in order if l in piv.index])
            disp = piv.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")

            st.markdown("#### Probability of Price Reaching Level Within Horizon")
            st.dataframe(disp, use_container_width=True)

            # Key takeaways
            st.markdown("---")
            st.markdown("#### 🎯 Key Observations")
            for lev in ['-20%', 'Balanced', 'Floor']:
                if lev in piv.index:
                    for hcol in sorted(piv.columns, key=lambda x: int(x.replace('d',''))):
                        p = piv.loc[lev, hcol]
                        if pd.notna(p):
                            if p >= 0.3:
                                st.warning(f"⚠️ **{lev}** reached in **{p*100:.0f}%** of cases within {hcol}")
                            elif p >= 0.1:
                                st.info(f"ℹ️ **{lev}** reached in **{p*100:.0f}%** of cases within {hcol}")
                            break

            # Forward returns from similar days
            st.markdown("---")
            st.markdown("#### 📈 Forward Returns from Similar Positions")
            max_h = 365
            sim_idx = np.where((np.abs(chp - CH) < 0.07) &
                               (np.arange(len(pr)) < len(pr) - max_h))[0]
            if len(sim_idx) > 10:
                ret_cols = st.columns(3)
                for i, (h, hn) in enumerate([(90, '90 days'), (180, '6 months'), (365, '1 year')]):
                    rets = np.array([pr[j + h] / pr[j] - 1 for j in sim_idx if j + h < len(pr)])
                    if len(rets) > 5:
                        with ret_cols[i]:
                            st.metric(f"Median ({hn})", f"{np.median(rets)*100:+.1f}%")
                            st.metric(f"% Positive", f"{(rets > 0).mean()*100:.0f}%")
                            st.caption(f"P25={np.percentile(rets,25)*100:+.0f}% | "
                                       f"P75={np.percentile(rets,75)*100:+.0f}% | "
                                       f"N={len(rets)}")
        else:
            st.warning("Not enough historical days with similar channel position. "
                       "This can happen when channel position is at extreme values.")

    # ================================================================
    # TAB 4 — ENTRY STRATEGY
    # ================================================================
    with tab4:
        st.markdown('<div class="explain">'
                    '<b>Entry Strategy</b> combines the channel position, risk area, and '
                    'historical probabilities to suggest entry levels and a DCA plan.<br><br>'
                    'For each price level, we compute:<br>'
                    '• <b>Historical probability</b> of price reaching that level (from tab 3)<br>'
                    '• <b>BTC amount</b> you would buy at that level<br>'
                    '• <b>Expected Value</b> = prob × (BTC × target) + (1-prob) × budget<br>'
                    '</div>', unsafe_allow_html=True)
        st.markdown("")

        sc1, sc2 = st.columns(2)
        budget = sc1.number_input("💰 Budget (€)", 100, 10_000_000, 10_000, 1000)
        hor_m = sc2.selectbox("📅 Horizon", [12, 24, 36], index=1,
                               format_func=lambda x: f"{x} months")

        # Target: balanced at horizon
        tgt_days = TDAYS + hor_m * 30.44
        tgt_balanced = M['balanced']['a'] * (tgt_days ** M['balanced']['b'])
        st.caption(f"Target price (Balanced at {hor_m}M): **${tgt_balanced:,.0f}**")

        st.markdown("---")

        # Entry levels
        MC_V = float(M['maxcross']['a'] * (TDAYS ** M['maxcross']['b']))
        levels = [
            ('Current Price', P),
            ('Max-Crossing', MC_V),
            ('Balanced', BL),
            ('-20% from now', P * 0.80),
            ('-30% from now', P * 0.70),
            ('Floor', FL),
        ]

        # Historical probabilities of reaching each level
        hor_d = int(hor_m * 30.44)
        sim_idx = np.where((np.abs(chp - CH) < 0.07) &
                           (np.arange(len(pr)) < len(pr) - hor_d))[0]

        rows = []
        for lname, lprice in levels:
            disc = (lprice / P - 1) * 100
            dd_need = lprice / P - 1

            if len(sim_idx) > 10:
                hits = sum(1 for i in sim_idx
                           if float(np.min(pr[i:i + hor_d + 1])) <= pr[i] * (1 + dd_need))
                prob = hits / len(sim_idx)
            else:
                prob = float('nan')

            btc_qty = budget / lprice if lprice > 0 else 0
            if np.isfinite(prob) and prob > 0:
                val_if_hit = btc_qty * tgt_balanced
                ev = prob * val_if_hit + (1 - prob) * budget
                exp_ret = (ev / budget - 1) * 100
            else:
                ev = exp_ret = float('nan')

            rows.append({
                'Level': lname,
                'Price': f"${lprice:,.0f}",
                'vs Now': f"{disc:+.1f}%",
                'Hist. Prob': f"{prob*100:.1f}%" if np.isfinite(prob) else "—",
                'BTC': f"{btc_qty:.6f}",
                'EV (€)': f"€{ev:,.0f}" if np.isfinite(ev) else "—",
                'Exp Ret': f"{exp_ret:+.1f}%" if np.isfinite(exp_ret) else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # DCA plan
        st.markdown("---")
        st.markdown("#### 📅 Dynamic DCA Plan")
        st.markdown(f"Allocation based on channel={CH*100:.0f}%, risk={RR*100:.0f}%:")

        if CH < 0.30:
            alloc = [('Immediate (lump sum)', 0.60), ('Reserve for dips', 0.40)]
            note = "Low in channel → favor immediate entry"
        elif CH < 0.50 and RR < 0.5:
            alloc = [('Immediate', 0.40), ('Weekly DCA', 0.30), ('Reserve', 0.30)]
            note = "Mid channel, moderate risk → balanced approach"
        elif RR > 0.6:
            alloc = [('Immediate', 0.15), ('Weekly DCA', 0.35), ('Reserve', 0.50)]
            note = "Elevated risk → conservative, keep large reserve for dips"
        else:
            alloc = [('Immediate', 0.30), ('Weekly DCA', 0.30), ('Reserve', 0.40)]
            note = "Standard balanced allocation"

        st.info(f"💡 **{note}**")

        dca_rows = []
        for comp, pct in alloc:
            amt = budget * pct
            if 'Immediate' in comp or 'lump' in comp:
                action = f"Buy now @ ${P:,.0f}"
                btc = f"{amt / P:.6f}"
            elif 'Weekly' in comp:
                weeks = hor_m * 4
                action = f"€{amt / weeks:,.0f}/week × {weeks} weeks"
                btc = "varies (DCA)"
            else:
                action = "Deploy at -20%, -30%, or Floor dips"
                btc = "conditional"
            dca_rows.append({'Component': comp, 'Allocation': f"{pct*100:.0f}%",
                             'Amount (€)': f"€{amt:,.0f}", 'Action': action, 'BTC': btc})
        st.dataframe(pd.DataFrame(dca_rows), use_container_width=True, hide_index=True)

        # Return projections
        st.markdown("---")
        st.markdown("#### 📈 Curve Projections (from current price)")
        prows = []
        for y in [1, 2, 3, 4]:
            fd = TDAYS + y * 365
            vals = {}
            for cname, cp in [('Floor', M['floor']), ('Balanced', M['balanced']),
                                ('Ceiling', M['ceiling'])]:
                v = cp['a'] * (fd ** cp['b'])
                vals[cname] = f"${v:,.0f} ({(v / P - 1)*100:+.0f}%)"
            prows.append({'Horizon': f"{y}Y", **vals})
        st.dataframe(pd.DataFrame(prows), use_container_width=True, hide_index=True)

        st.error("⚠️ **DISCLAIMER** — Models can be wrong. Past performance ≠ future results. "
                 "Only invest what you can afford to lose. This is NOT financial advice.")

    # ================================================================
    # TAB 5 — DEBUG DATA EXPORT
    # ================================================================
    with tab5:
        st.markdown('<div class="explain">'
                    '<b>Debug & Verification Data</b><br>'
                    'Download a CSV with all computed values for every day. '
                    'Use this to verify calculations, debug issues, or feed into other tools.<br><br>'
                    '<b>Columns explained:</b><br>'
                    '• <b>Date, Days</b>: date and days since Genesis (2009-01-03)<br>'
                    '• <b>Close, Low</b>: daily prices from CryptoCompare<br>'
                    '• <b>NLB</b>: Never Look Back price (staircase)<br>'
                    '• <b>NLB_Update</b>: 1 if NLB increased this day, 0 otherwise<br>'
                    '• <b>Floor, Balanced, Ceiling, MaxCross</b>: power law curves<br>'
                    '• <b>ChPos</b>: channel position 0-1 (NLB within Floor-Ceiling)<br>'
                    '• <b>Excess</b>: daily log(price/floor), 0 if below floor<br>'
                    '• <b>RiskArea</b>: decayed cumulative excess<br>'
                    '• <b>RiskRMax</b>: rolling max of RiskArea (normalization denominator)<br>'
                    '• <b>RiskRatio</b>: RiskArea / RiskRMax, 0-1<br>'
                    '</div>', unsafe_allow_html=True)
        st.markdown("")

        # Build export dataframe
        export = df[['Date', 'Days', 'Close', 'Low', 'NLB', 'Floor', 'Balanced',
                      'Ceiling', 'MaxCross', 'ChPos', 'Excess', 'RiskArea',
                      'RiskRMax', 'RiskRatio']].copy()
        # Add NLB update flag
        nlb_arr = export['NLB'].to_numpy()
        upd_flag = np.zeros(len(nlb_arr), dtype=int)
        upd_flag[0] = 1
        upd_flag[1:] = (nlb_arr[1:] > nlb_arr[:-1]).astype(int)
        export.insert(5, 'NLB_Update', upd_flag)

        # Round for readability
        for col in ['Close', 'Low', 'NLB', 'Floor', 'Balanced', 'Ceiling', 'MaxCross']:
            export[col] = export[col].round(2)
        for col in ['ChPos', 'Excess', 'RiskArea', 'RiskRMax', 'RiskRatio']:
            export[col] = export[col].round(6)

        # Summary stats
        st.markdown("#### 📊 Summary")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Days", f"{len(export):,}")
        sc2.metric("NLB Updates", f"{upd_flag.sum():,}")
        sc3.metric("Floor b", f"{M['floor']['b']:.4f}")
        sc4.metric("Balanced k", f"{M['balanced']['k']:.4f}")

        st.markdown("#### 🔧 Model Parameters (copy-paste)")
        params_text = (
            f"Floor:     a = {M['floor']['a']:.10e},  b = {M['floor']['b']:.6f}\n"
            f"Balanced:  a = {M['balanced']['a']:.10e},  b = {M['balanced']['b']:.6f},  "
            f"k = {M['balanced']['k']:.6f}\n"
            f"Ceiling:   a = {M['ceiling']['a']:.10e},  b = {M['ceiling']['b']:.6f}\n"
            f"MaxCross:  a = {M['maxcross']['a']:.10e},  b = {M['maxcross']['b']:.6f},  "
            f"k = {M['maxcross']['k']:.6f},  crossings = {M['maxcross']['n_crossings']}\n"
            f"Risk:      lambda = {risk_lam},  window = {risk_win}\n"
            f"Data:      {export['Date'].iloc[0].strftime('%Y-%m-%d')} to "
            f"{export['Date'].iloc[-1].strftime('%Y-%m-%d')} ({len(export)} days)\n"
        )
        st.code(params_text, language='text')

        # Preview
        st.markdown("#### 👀 Data Preview (last 30 days)")
        st.dataframe(export.tail(30), use_container_width=True, hide_index=True)

        # Download
        csv_buf = io.StringIO()
        export.to_csv(csv_buf, index=False)
        st.download_button(
            label="📥 Download Full CSV",
            data=csv_buf.getvalue(),
            file_name=f"btc_nlb_debug_{TODAY.strftime('%Y%m%d')}.csv",
            mime='text/csv',
            use_container_width=True)

        # Also offer parameters as JSON
        import json
        params_json = json.dumps({
            'floor': M['floor'], 'balanced': M['balanced'],
            'ceiling': M['ceiling'], 'maxcross': M['maxcross'],
            'risk_lambda': risk_lam, 'risk_window': risk_win,
            'data_start': export['Date'].iloc[0].strftime('%Y-%m-%d'),
            'data_end': export['Date'].iloc[-1].strftime('%Y-%m-%d'),
            'n_days': len(export),
            'current': {
                'price': P, 'nlb': NLB_V, 'floor': FL, 'balanced': BL,
                'ceiling': CL, 'channel_pct': round(CH * 100, 2),
                'risk_pct': round(RR * 100, 2),
            }
        }, indent=2)
        st.download_button(
            label="📥 Download Parameters JSON",
            data=params_json,
            file_name=f"btc_nlb_params_{TODAY.strftime('%Y%m%d')}.json",
            mime='application/json',
            use_container_width=True)

    # ── FOOTER ───────────────────────────────────────────────────────
    st.markdown("---")
    st.warning("**⚠️ DISCLAIMER** — Informational/educational only. NOT financial advice. "
               "Models based on history — no future guarantees. Crypto is volatile. "
               "You may lose all capital. Consult a financial advisor. Author: no responsibility.")
    st.caption(f"Data: CryptoCompare | Model: NLB Floor Envelope + Balanced PL + Risk Area | "
               f"Last update: {TODAY.strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()
