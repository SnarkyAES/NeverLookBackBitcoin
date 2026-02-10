"""
================================================================================
BITCOIN NLB (Never Look Back) ANALYSIS v3
================================================================================
BTC/USD — Power Law Channel + Risk Area + Historical Probabilities
With debug data export. Fixes: floor constraint, channel width, risk decay, CSS
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
import time, io, json

st.set_page_config(page_title="BTC NLB Analysis", page_icon="₿", layout="wide",
                   initial_sidebar_state="expanded")

if 'disclaimer_accepted' not in st.session_state:
    st.session_state.disclaimer_accepted = False

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

# CSS — FIX: explicit light text everywhere for dark theme
st.markdown("""<style>
.hdr{font-size:2.2rem;font-weight:bold;color:#F7931A;text-align:center;padding:0.3rem;}
.sub{text-align:center;color:#aaa;margin-bottom:1rem;font-size:0.95rem;}
.sig-green{background:linear-gradient(135deg,#0f5132,#198754);color:white;padding:1rem;border-radius:10px;text-align:center;font-size:1.2rem;font-weight:bold;}
.sig-yellow{background:linear-gradient(135deg,#664d03,#ffc107);color:#1a1a1a;padding:1rem;border-radius:10px;text-align:center;font-size:1.2rem;font-weight:bold;}
.sig-orange{background:linear-gradient(135deg,#7c2d12,#ea580c);color:white;padding:1rem;border-radius:10px;text-align:center;font-size:1.2rem;font-weight:bold;}
.sig-red{background:linear-gradient(135deg,#7f1d1d,#dc2626);color:white;padding:1rem;border-radius:10px;text-align:center;font-size:1.2rem;font-weight:bold;}
.sig-gray{background:linear-gradient(135deg,#495057,#6c757d);color:white;padding:1rem;border-radius:10px;text-align:center;font-size:1.2rem;font-weight:bold;}
.explain{background:rgba(247,147,26,0.08);border-left:4px solid #F7931A;padding:1rem;border-radius:5px;margin:0.5rem 0;font-size:0.9rem;color:#e0e0e0;}
.explain b{color:#F7931A;}
</style>""", unsafe_allow_html=True)

GENESIS = pd.Timestamp('2009-01-03')

# ============================================================================
# DATA
# ============================================================================
@st.cache_data(ttl=3600)
def load_btc(start_date='2010-07-18'):
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
    NLB = cummax(forward_min).
    forward_min[t] = min(series[t], series[t+1], ..., series[N])
    nlb[t] = max(forward_min[0], forward_min[1], ..., forward_min[t])
    
    Returns monotone non-decreasing staircase.
    """
    s = series.to_numpy(dtype=float)
    n = len(s)
    fwd_min = np.empty(n)
    fwd_min[-1] = s[-1]
    for i in range(n - 2, -1, -1):
        fwd_min[i] = min(s[i], fwd_min[i + 1])
    nlb = np.maximum.accumulate(fwd_min)
    return pd.Series(nlb, index=series.index)

# ============================================================================
# CURVE FITTING — v3 fixes
# ============================================================================
def fit_floor_envelope(days, nlb, b_min=4.5, b_max=6.5, b_steps=4000):
    """
    FLOOR A — Tightest power law envelope under ALL NLB points.
    
    v3 FIX: Uses ALL NLB points (not just updates) to ensure floor <= NLB everywhere.
    The constraint c = min(log(NLB) - b*log(days)) guarantees floor stays below.
    
    Grid search on b, for each b:
      c = min(log(nlb_i) - b * log(days_i))  → ensures floor ≤ NLB_i for all i
      gap = sum(log(nlb_i) - c - b*log(days_i))  → total log-space gap to minimize
    """
    x = days.astype(float)
    y = nlb.astype(float)
    # Use ALL points to guarantee floor <= NLB everywhere
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    logx, logy = np.log(x), np.log(y)
    
    best = None
    for b in np.linspace(b_min, b_max, b_steps):
        c = float(np.min(logy - b * logx))  # max intercept staying below ALL points
        gap = float(np.sum(logy - (c + b * logx)))
        if best is None or gap < best['gap']:
            best = {'a': float(np.exp(c)), 'b': float(b), 'gap': gap}
    
    # Verify constraint
    floor_vals = best['a'] * (x ** best['b'])
    violations = int(np.sum(floor_vals > y * 1.001))  # 0.1% tolerance for float
    best['n_points'] = int(mask.sum())
    best['violations'] = violations
    return best


def fit_balanced(days, nlb, a_floor, b_floor):
    """
    BALANCED — Fair value: shift Floor up until area above = area below in log-space.
    Uses ALL NLB points. Same exponent b, only coefficient changes.
    """
    x = days.astype(float)
    y = nlb.astype(float)
    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    log_floor = np.log(a_floor * (x ** b_floor))
    log_y = np.log(y)
    
    def area_imbalance(k):
        diff = log_y - (log_floor + k)
        return abs(float(np.sum(np.maximum(0, diff))) - float(np.sum(np.maximum(0, -diff))))
    
    res = minimize_scalar(area_imbalance, bounds=(0, 3), method='bounded')
    k = float(res.x)
    return {'a': a_floor * np.exp(k), 'b': b_floor, 'k': k,
            'area_imbalance': float(res.fun)}


def fit_max_crossing(days, nlb, a_floor, b_floor, k_steps=2000):
    """MAX-CROSSING: shift maximizing NLB crossings."""
    x = days.astype(float)
    y = nlb.astype(float)
    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    lf = np.log(a_floor * (x ** b_floor))
    ly = np.log(y)
    best_k, best_c = 0.0, 0
    for k in np.linspace(-0.5, 1.5, k_steps):
        s = np.sign(ly - (lf + k))
        for i in range(1, len(s)):
            if s[i] == 0: s[i] = s[i-1]
        c = int(np.sum(s[1:] != s[:-1]))
        if c > best_c:
            best_c, best_k = c, float(k)
    return {'a': a_floor * np.exp(best_k), 'b': b_floor, 'k': best_k,
            'n_crossings': best_c}

# ============================================================================
# RISK AREA — v3: use PRICE vs BALANCED (not floor)
# ============================================================================
#
# v3 FIX: The risk area now measures excess above BALANCED (fair value),
# not above FLOOR. This is more meaningful because:
#   - Price is often well above floor even in "normal" times
#   - What matters for risk is: how far above fair value are we?
#   - Being 2x above balanced is riskier than being 2x above floor
#
# Also: use shorter default λ (0.995 ≈ 139-day halflife) so the
# indicator responds faster to price drops.
# ============================================================================

def compute_risk_area(price, ref_curve, lam=0.995):
    """
    Decayed cumulative log-excess above reference curve.
    
    excess(t) = max(0, log(price_t / ref_t))
    area(t) = area(t-1) * λ + excess(t)
    
    ref_curve should be BALANCED (fair value), not floor.
    """
    n = len(price)
    ratio = np.maximum(price / ref_curve, 1e-12)
    excess = np.maximum(0.0, np.log(ratio))
    area = np.empty(n)
    area[0] = excess[0]
    for t in range(1, n):
        area[t] = area[t - 1] * lam + excess[t]
    return area, excess


def compute_risk_ratio(area, window=365):
    """Normalize to [0,1] via walk-forward rolling max."""
    n = len(area)
    risk = np.zeros(n)
    rmax = np.zeros(n)
    for t in range(n):
        rm = float(np.max(area[max(0, t - window):t + 1]))
        rmax[t] = rm
        if rm > 0:
            risk[t] = area[t] / rm
    return np.clip(risk, 0, 1), rmax


def backtest_risk_area(price, ref_curve, lam, rw, horizons=(30, 90, 180, 365)):
    """Walk-forward backtest of risk indicator."""
    n = len(price)
    area, _ = compute_risk_area(price, ref_curve, lam)
    risk, _ = compute_risk_ratio(area, rw)
    buckets = [
        ('0-20% Very Low', 0.0, 0.2), ('20-40% Low', 0.2, 0.4),
        ('40-60% Mid', 0.4, 0.6), ('60-80% High', 0.6, 0.8),
        ('80-100% V.High', 0.8, 1.01),
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
            if cnt < 15: continue
            r, d = fwd[mask], fdd[mask]
            rows.append({
                'Risk Bucket': bname, 'Horizon': f"{h}d", 'N': cnt,
                'Median Ret': f"{np.median(r)*100:+.1f}%",
                'Mean Ret': f"{np.mean(r)*100:+.1f}%",
                '% Positive': f"{(r > 0).mean()*100:.0f}%",
                'Med MaxDD': f"{np.median(d)*100:.1f}%",
                'P10 MaxDD': f"{np.percentile(d, 10)*100:.1f}%",
            })
    return pd.DataFrame(rows)


def calibrate_lambda(price, ref_curve, rw=365):
    """Find λ maximizing 6M return separation."""
    n = len(price)
    fwd = np.full(n, np.nan)
    for t in range(n - 180):
        fwd[t] = price[t + 180] / price[t] - 1
    best_lam, best_sep = 0.995, 0.0
    rows = []
    for lam in np.arange(0.990, 0.9996, 0.0005):
        a, _ = compute_risk_area(price, ref_curve, lam)
        r, _ = compute_risk_ratio(a, rw)
        hi = (r > 0.7) & np.isfinite(fwd)
        lo = (r < 0.3) & np.isfinite(fwd)
        sep = float(np.median(fwd[lo]) - np.median(fwd[hi])) if hi.sum() > 20 and lo.sum() > 20 else 0.0
        rows.append({'lambda': float(lam), 'separation': sep})
        if sep > best_sep:
            best_sep, best_lam = sep, float(lam)
    return best_lam, pd.DataFrame(rows)

# ============================================================================
# CHANNEL & PROBABILITIES
# ============================================================================
def channel_pos(nlb, floor_v, ceil_v):
    """Channel position 0-1 in log-space."""
    ln = np.log(np.maximum(nlb, 1e-12))
    lf = np.log(np.maximum(floor_v, 1e-12))
    lc = np.log(np.maximum(ceil_v, 1e-12))
    w = lc - lf
    return np.clip(np.where(w > 0, (ln - lf) / w, 0.5), 0, 1)


def hist_prob_table(price, ch_pos, floor_v, bal_v, current_ch,
                    tol=0.07, horizons=(30, 60, 90, 180, 365)):
    """Historical probability from similar channel positions."""
    n = len(price)
    max_h = max(horizons)
    sim = np.where((np.abs(ch_pos - current_ch) < tol) & (np.arange(n) < n - max_h))[0]
    if len(sim) < 10:
        return None, 0
    targets = {'Floor': 'floor', 'Balanced': 'balanced',
               '-10%': -0.10, '-20%': -0.20, '-30%': -0.30, '-40%': -0.40}
    rows = []
    for h in horizons:
        vi = sim[sim < n - h]
        if len(vi) < 5: continue
        for tname, tdef in targets.items():
            hits = sum(1 for idx in vi
                       if float(np.min(price[idx:idx+h+1])) <= (
                           floor_v[idx] if tdef == 'floor' else
                           bal_v[idx] if tdef == 'balanced' else
                           price[idx] * (1 + tdef)))
            rows.append({'Horizon': f"{h}d", 'Level': tname,
                         'Prob': hits / len(vi), 'N': len(vi)})
    return (pd.DataFrame(rows) if rows else None), len(sim)

# ============================================================================
# MODEL BUILDER — v3
# ============================================================================
def build_model(df_raw, lam=0.995, rw=365):
    df = df_raw.copy()
    df['Days'] = (df['Date'] - GENESIS).dt.days
    df['NLB'] = compute_nlb(df['Low'])
    df = df[df['Days'] > 0].copy().reset_index(drop=True)

    days = df['Days'].to_numpy(float)
    nlb = df['NLB'].to_numpy(float)
    price = df['Close'].to_numpy(float)

    floor_p = fit_floor_envelope(days, nlb)
    bal_p = fit_balanced(days, nlb, floor_p['a'], floor_p['b'])
    mc_p = fit_max_crossing(days, nlb, floor_p['a'], floor_p['b'])
    ceil_a = bal_p['a'] * np.exp(bal_p['k'])

    df['Floor']    = floor_p['a'] * (days ** floor_p['b'])
    df['Balanced'] = bal_p['a']   * (days ** bal_p['b'])
    df['Ceiling']  = ceil_a       * (days ** bal_p['b'])
    df['MaxCross'] = mc_p['a']    * (days ** mc_p['b'])
    df['ChPos']    = channel_pos(nlb, df['Floor'].values, df['Ceiling'].values)

    # v3: Risk vs BALANCED (not floor)
    ra, excess = compute_risk_area(price, df['Balanced'].values, lam)
    rr, rmax = compute_risk_ratio(ra, rw)
    df['Excess']    = excess
    df['RiskArea']  = ra
    df['RiskRMax']  = rmax
    df['RiskRatio'] = rr

    # Also compute price distance to balanced (useful metric)
    df['DistBal'] = np.log(price / df['Balanced'].values)  # negative = below balanced

    return {
        'df': df, 'floor': floor_p, 'balanced': bal_p,
        'ceiling': {'a': ceil_a, 'b': bal_p['b']}, 'maxcross': mc_p,
    }


def project_curves(model, years=5):
    last_date = model['df']['Date'].iloc[-1]
    last_days = float(model['df']['Days'].iloc[-1])
    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=years*365, freq='D')
    fdays = np.array([(d - GENESIS).days for d in dates], dtype=float)
    return pd.DataFrame({
        'Date': dates, 'Days': fdays,
        'Floor':    model['floor']['a']    * (fdays ** model['floor']['b']),
        'Balanced': model['balanced']['a'] * (fdays ** model['balanced']['b']),
        'Ceiling':  model['ceiling']['a']  * (fdays ** model['ceiling']['b']),
    })


def get_signal(ch, rr):
    if ch < 0.15:
        return ('STRONG OPPORTUNITY', '🟢🟢',
                f'Very low in channel ({ch*100:.0f}%). Historically rare.', 'sig-green')
    elif ch < 0.30:
        return ('OPPORTUNITY', '🟢',
                f'Low channel ({ch*100:.0f}%). Historically favorable.', 'sig-green')
    elif ch < 0.50 and rr < 0.5:
        return ('ACCUMULATE', '🟡',
                f'Mid-low ({ch*100:.0f}%), moderate risk ({rr*100:.0f}%).', 'sig-yellow')
    elif ch > 0.85:
        return ('HIGH RISK', '🔴',
                f'Very high ({ch*100:.0f}%). Extended.', 'sig-red')
    elif ch > 0.70 and rr > 0.7:
        return ('CAUTION', '🟠',
                f'High ({ch*100:.0f}%) + elevated risk ({rr*100:.0f}%).', 'sig-orange')
    elif ch < 0.70:
        return ('NEUTRAL', '⚪',
                f'Mid channel ({ch*100:.0f}%).', 'sig-gray')
    else:
        return ('WATCH', '🟡', f'Upper ({ch*100:.0f}%).', 'sig-yellow')

# ============================================================================
# MAIN
# ============================================================================
def main():
    st.markdown('<p class="hdr">₿ Bitcoin NLB Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub">Never Look Back Price — Power Law Channel & Risk Area</p>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("---")
        proj_years = st.slider("📐 Projection (years)", 1, 5, 3)
        st.markdown("---")
        st.markdown("### 🔬 Risk Area")
        risk_lam = st.slider("Decay λ", 0.990, 0.999, 0.995, 0.001,
                              help="0.995 ≈ 139-day half-life. 0.997 ≈ 231 days.")
        risk_win = st.slider("Norm. window", 180, 730, 365, 30)
        st.markdown("---")
        st.markdown("### 📚 NLB Curves")
        st.markdown("""
        **NLB** = highest price BTC never went below permanently (staircase).
        
        **Floor** = power law under ALL NLB points (absolute support).
        
        **Balanced** = Floor shifted up so area above = area below (fair value).
        
        **Ceiling** = symmetric to Floor vs Balanced (upper bound).
        
        **Max-Cross** = shift maximizing NLB crossings (support/resistance).
        """)
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

    # Load & build
    with st.spinner("📥 Loading BTC data…"):
        raw = load_btc('2010-07-18')
        if raw is None or len(raw) < 100:
            st.error("Failed to load data"); return

    with st.spinner("🔧 Building NLB model…"):
        M = build_model(raw, lam=risk_lam, rw=risk_win)

    df   = M['df']
    days = df['Days'].to_numpy(float)
    pr   = df['Close'].to_numpy(float)
    nlb  = df['NLB'].to_numpy(float)
    fl_v = df['Floor'].to_numpy(float)
    bl_v = df['Balanced'].to_numpy(float)
    cl_v = df['Ceiling'].to_numpy(float)
    chp  = df['ChPos'].to_numpy(float)
    rr   = df['RiskRatio'].to_numpy(float)

    TODAY = df['Date'].iloc[-1]
    TDAYS = float(days[-1])
    P     = float(pr[-1])
    FL    = float(fl_v[-1])
    BL    = float(bl_v[-1])
    CL    = float(cl_v[-1])
    CH    = float(chp[-1])
    RR    = float(rr[-1])
    D_FL  = (P / FL - 1) * 100
    D_BL  = (P / BL - 1) * 100

    sig_name, sig_emo, sig_desc, sig_css = get_signal(CH, RR)

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Price", f"${P:,.0f}",
              f"{(P/pr[-2]-1)*100:+.2f}%" if len(pr) > 1 else "")
    c2.metric("🔒 Floor", f"${FL:,.0f}", f"{D_FL:+.1f}% above")
    c3.metric("⚖️ Balanced", f"${BL:,.0f}", f"{D_BL:+.1f}%")
    c4.metric("📊 Channel", f"{CH*100:.1f}%", "0%=Floor 100%=Ceil")
    c5.metric("⚠️ Risk", f"{RR*100:.0f}%",
              "vs Balanced • Low<30 High>70")

    st.markdown("---")
    st.markdown(f'<div class="{sig_css}">{sig_emo} {sig_name}<br>'
                f'<span style="font-size:0.85rem;font-weight:normal;">{sig_desc}</span></div>',
                unsafe_allow_html=True)
    st.markdown("")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 NLB Channel", "⚠️ Risk Area", "📊 Probabilities",
        "🎯 Entry Strategy", "📦 Debug Data"])

    # ── TAB 1: NLB CHANNEL ─────────────────────────────────────────
    with tab1:
        st.markdown('<div class="explain">'
                    '<b>Chart:</b> BTC price (orange), NLB staircase (white, shape=hv), '
                    'and the power law channel. The NLB rises in steps — each step is a new '
                    '"never look back" level. Floor (green) stays below ALL NLB points by construction. '
                    'Balanced (gray dashed) is the log-median. Ceiling (red) is the symmetric upper bound.'
                    '</div>', unsafe_allow_html=True)

        proj = project_curves(M, proj_years)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='BTC Price',
                                 line=dict(color='#F7931A', width=1.2), opacity=0.6))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['NLB'], name='NLB',
                                 line=dict(color='white', width=2.5, shape='hv')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Ceiling'], name='Ceiling',
                                 line=dict(color='rgba(220,53,69,0.35)', width=1)))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Floor'], name='Floor',
                                 line=dict(color='#28a745', width=2),
                                 fill='tonexty', fillcolor='rgba(247,147,26,0.04)'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Balanced'], name='Balanced',
                                 line=dict(color='#adb5bd', width=1.5, dash='dash')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MaxCross'], name='Max-Cross',
                                 line=dict(color='#17a2b8', width=1, dash='dot'),
                                 visible='legendonly'))
        for col, clr in [('Floor','rgba(40,167,69,0.35)'),('Balanced','rgba(173,181,189,0.35)'),
                          ('Ceiling','rgba(220,53,69,0.2)')]:
            fig.add_trace(go.Scatter(x=proj['Date'], y=proj[col], showlegend=False,
                                     line=dict(color=clr, width=1, dash='dot')))
        y_hi = max(float(pr.max()), float(proj['Ceiling'].max())) * 2
        y_lo = max(float(fl_v.min()), 0.001)
        fig.update_layout(yaxis_type='log', yaxis_title='USD (log)', height=580,
                          hovermode='x unified', template='plotly_dark',
                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font_size=11),
                          margin=dict(l=0, r=0, t=30, b=0))
        fig.update_yaxes(range=[np.log10(y_lo), np.log10(y_hi)])
        st.plotly_chart(fig, use_container_width=True)

        # Channel position
        st.markdown("#### Channel Position History")
        st.markdown('<div class="explain">'
                    '0% = NLB at floor (cheap). 100% = NLB at ceiling (expensive). '
                    'Green zone < 30%. Red zone > 70%.'
                    '</div>', unsafe_allow_html=True)
        fig_ch = go.Figure()
        fig_ch.add_trace(go.Scatter(x=df['Date'], y=df['ChPos']*100, name='Channel %',
                                    line=dict(color='#F7931A', width=1.5),
                                    fill='tozeroy', fillcolor='rgba(247,147,26,0.08)'))
        fig_ch.add_hrect(y0=0, y1=30, fillcolor='rgba(40,167,69,0.06)', line_width=0)
        fig_ch.add_hrect(y0=70, y1=100, fillcolor='rgba(220,53,69,0.06)', line_width=0)
        fig_ch.add_hline(y=70, line_dash='dash', line_color='red', opacity=0.4)
        fig_ch.add_hline(y=30, line_dash='dash', line_color='green', opacity=0.4)
        fig_ch.update_layout(yaxis_title='Channel %', height=280, template='plotly_dark',
                             hovermode='x unified', margin=dict(l=0, r=0, t=10, b=0))
        fig_ch.update_yaxes(range=[0, 100])
        st.plotly_chart(fig_ch, use_container_width=True)

        # Levels table
        st.markdown("#### Current Levels & Projections")
        periods = {'Today': 0, '6M': 180, '1Y': 365, '2Y': 730, '3Y': 1095}
        trows = []
        for cname, cp in [('Floor', M['floor']), ('Balanced', M['balanced']),
                            ('Ceiling', M['ceiling'])]:
            row = {'Curve': cname}
            for pn, pd_ in periods.items():
                v = cp['a'] * ((TDAYS + pd_) ** cp['b'])
                row[pn] = f"${v:,.0f}" + (f" ({(v/P-1)*100:+.0f}%)" if pd_ > 0 else "")
            trows.append(row)
        st.dataframe(pd.DataFrame(trows), use_container_width=True, hide_index=True)

        with st.expander("🔧 Model Parameters"):
            st.code(
                f"Floor:     a = {M['floor']['a']:.10e},  b = {M['floor']['b']:.6f}  "
                f"({M['floor']['n_points']} pts, {M['floor']['violations']} violations)\n"
                f"Balanced:  a = {M['balanced']['a']:.10e},  b = {M['balanced']['b']:.6f},  "
                f"k = {M['balanced']['k']:.6f}\n"
                f"Ceiling:   a = {M['ceiling']['a']:.10e},  b = {M['ceiling']['b']:.6f}\n"
                f"MaxCross:  a = {M['maxcross']['a']:.10e},  k = {M['maxcross']['k']:.6f},  "
                f"crossings = {M['maxcross']['n_crossings']}\n"
                f"Ceil/Floor ratio: {CL/FL:.2f}x  |  Bal/Floor: {BL/FL:.2f}x",
                language='text')

    # ── TAB 2: RISK AREA ──────────────────────────────────────────
    with tab2:
        hl = abs(int(np.log(0.5) / np.log(risk_lam)))
        st.markdown('<div class="explain">'
                    '<b>v3 Change:</b> Risk Area now measures excess above <b>BALANCED</b> (fair value), '
                    'not above Floor. This makes more sense: being at $90k when balanced is $149k '
                    'means you are BELOW fair value → low risk. Being at $200k when balanced is $149k '
                    'means high excess → high risk.<br><br>'
                    f'<b>Formula:</b> area(t) = area(t-1) × {risk_lam} + max(0, log(price/balanced))<br>'
                    f'<b>Half-life:</b> {hl} days<br>'
                    f'<b>Current Risk:</b> {RR*100:.0f}%<br>'
                    f'<b>Price vs Balanced:</b> {D_BL:+.1f}% '
                    f'{"(BELOW fair value → no excess accumulating)" if D_BL < 0 else "(ABOVE → excess accumulating)"}'
                    '</div>', unsafe_allow_html=True)

        fig_r = make_subplots(
            rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.25, 0.35],
            subplot_titles=('Price vs Balanced (fair value)', 'Risk Area (decayed)',
                            'Risk Ratio (normalized 0-100%)'),
            vertical_spacing=0.06)
        fig_r.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Price',
                                   line=dict(color='#F7931A', width=1.5)), row=1, col=1)
        fig_r.add_trace(go.Scatter(x=df['Date'], y=df['Balanced'], name='Balanced',
                                   line=dict(color='#adb5bd', width=1.5, dash='dash')), row=1, col=1)
        fig_r.add_trace(go.Scatter(x=df['Date'], y=df['Floor'], name='Floor',
                                   line=dict(color='#28a745', width=1, dash='dot')), row=1, col=1)
        fig_r.add_trace(go.Scatter(x=df['Date'], y=df['RiskArea'], name='Risk Area',
                                   line=dict(color='#6f42c1', width=1.2),
                                   fill='tozeroy', fillcolor='rgba(111,66,193,0.08)'), row=2, col=1)
        fig_r.add_trace(go.Scatter(x=df['Date'], y=df['RiskRMax'], name='Rolling Max',
                                   line=dict(color='rgba(255,255,255,0.25)', width=1, dash='dot')),
                        row=2, col=1)
        fig_r.add_trace(go.Scatter(x=df['Date'], y=df['RiskRatio']*100, name='Risk %',
                                   line=dict(color='#dc3545', width=1.5),
                                   fill='tozeroy', fillcolor='rgba(220,53,69,0.1)'), row=3, col=1)
        fig_r.add_hrect(y0=70, y1=100, fillcolor='rgba(220,53,69,0.08)',
                        line_width=0, row=3, col=1)
        fig_r.add_hrect(y0=0, y1=30, fillcolor='rgba(40,167,69,0.08)',
                        line_width=0, row=3, col=1)
        fig_r.add_hline(y=70, line_dash='dash', line_color='red', opacity=0.4, row=3, col=1)
        fig_r.add_hline(y=30, line_dash='dash', line_color='green', opacity=0.4, row=3, col=1)
        fig_r.update_layout(height=700, template='plotly_dark', hovermode='x unified',
                            margin=dict(l=0, r=0, t=40, b=0))
        fig_r.update_yaxes(type='log', title_text='USD', row=1, col=1)
        fig_r.update_yaxes(title_text='Area', row=2, col=1)
        fig_r.update_yaxes(range=[0, 100], title_text='%', row=3, col=1)
        st.plotly_chart(fig_r, use_container_width=True)

        # Backtest
        st.markdown("#### 🔬 Risk Area Backtest")
        st.markdown('<div class="explain">'
                    'Walk-forward: for each historical day, risk is computed with past-only data, '
                    'then we measure what happened next. If the indicator works, high-risk buckets '
                    'should show worse forward returns than low-risk buckets.'
                    '</div>', unsafe_allow_html=True)
        with st.spinner("Running backtest…"):
            bt = backtest_risk_area(pr, bl_v, risk_lam, risk_win)
        if len(bt) > 0:
            h_sel = st.selectbox("Horizon", ['90d', '180d', '365d'], index=1, key='bt_h')
            h_data = bt[bt['Horizon'] == h_sel]
            if len(h_data) > 0:
                st.dataframe(h_data.drop(columns='Horizon'), use_container_width=True, hide_index=True)

        with st.expander("🔬 Calibrate λ (~30s)"):
            if st.button("Run Calibration", key='cal'):
                with st.spinner("Testing…"):
                    bl_, cal = calibrate_lambda(pr, bl_v, risk_win)
                st.success(f"Best λ = {bl_:.4f} (half-life ≈ {abs(int(np.log(0.5)/np.log(bl_)))} days)")
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(x=cal['lambda'], y=cal['separation'],
                                           mode='lines+markers', line=dict(color='#17a2b8')))
                fig_c.add_vline(x=bl_, line_dash='dash', line_color='#28a745')
                fig_c.update_layout(xaxis_title='λ', yaxis_title='Separation', height=300,
                                    template='plotly_dark')
                st.plotly_chart(fig_c, use_container_width=True)

    # ── TAB 3: PROBABILITIES ───────────────────────────────────────
    with tab3:
        st.markdown('<div class="explain">'
                    '<b>Method:</b> Find all historical days when channel position ≈ today '
                    f'({CH*100:.1f}% ± 7%). Track minimum price over next N days. '
                    'Count how often each target was reached. Purely empirical — no model.'
                    '</div>', unsafe_allow_html=True)

        prob_df, n_sim = hist_prob_table(pr, chp, fl_v, bl_v, CH, tol=0.07)
        if prob_df is not None and len(prob_df) > 0:
            st.info(f"Found **{n_sim}** similar historical days (channel {(CH-0.07)*100:.0f}%–{(CH+0.07)*100:.0f}%)")
            piv = prob_df.pivot_table(index='Level', columns='Horizon', values='Prob', aggfunc='first')
            order = ['Floor', 'Balanced', '-10%', '-20%', '-30%', '-40%']
            piv = piv.reindex([l for l in order if l in piv.index])
            st.markdown("#### Probability of Reaching Level Within Horizon")
            st.dataframe(piv.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"),
                         use_container_width=True)

            st.markdown("---")
            st.markdown("#### Forward Returns from Similar Positions")
            sim_idx = np.where((np.abs(chp - CH) < 0.07) & (np.arange(len(pr)) < len(pr) - 365))[0]
            if len(sim_idx) > 10:
                cols = st.columns(3)
                for i, (h, hn) in enumerate([(90, '90d'), (180, '6M'), (365, '1Y')]):
                    rets = np.array([pr[j+h]/pr[j]-1 for j in sim_idx if j+h < len(pr)])
                    if len(rets) > 5:
                        with cols[i]:
                            st.metric(f"Median ({hn})", f"{np.median(rets)*100:+.1f}%")
                            st.metric(f"% Positive", f"{(rets>0).mean()*100:.0f}%")
                            st.caption(f"P25={np.percentile(rets,25)*100:+.0f}% "
                                       f"P75={np.percentile(rets,75)*100:+.0f}% N={len(rets)}")
        else:
            st.warning("Not enough similar days. Channel position may be at extreme values.")

    # ── TAB 4: ENTRY STRATEGY ─────────────────────────────────────
    with tab4:
        st.markdown('<div class="explain">'
                    'Combines channel position, risk, and historical probabilities '
                    'to evaluate entry levels and suggest a DCA plan.'
                    '</div>', unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        budget = sc1.number_input("💰 Budget (€)", 100, 10_000_000, 10_000, 1000)
        hor_m = sc2.selectbox("Horizon", [12, 24, 36], index=1,
                               format_func=lambda x: f"{x} months")
        tgt_d = TDAYS + hor_m * 30.44
        tgt = M['balanced']['a'] * (tgt_d ** M['balanced']['b'])
        st.caption(f"Target (Balanced at {hor_m}M): **${tgt:,.0f}**")
        st.markdown("---")

        MC_V = float(M['maxcross']['a'] * (TDAYS ** M['maxcross']['b']))
        levels = [('Current', P), ('Max-Cross', MC_V), ('Balanced', BL),
                  ('-20%', P*0.8), ('-30%', P*0.7), ('Floor', FL)]
        hor_d = int(hor_m * 30.44)
        sim_idx = np.where((np.abs(chp - CH) < 0.07) & (np.arange(len(pr)) < len(pr) - hor_d))[0]

        rows = []
        for lname, lp in levels:
            disc = (lp / P - 1) * 100
            dd = lp / P - 1
            prob = (sum(1 for i in sim_idx if np.min(pr[i:i+hor_d+1]) <= pr[i]*(1+dd))
                    / len(sim_idx)) if len(sim_idx) > 10 else float('nan')
            btc = budget / lp if lp > 0 else 0
            ev = prob * btc * tgt + (1-prob) * budget if np.isfinite(prob) and prob > 0 else float('nan')
            rows.append({
                'Level': lname, 'Price': f"${lp:,.0f}", 'vs Now': f"{disc:+.1f}%",
                'Hist Prob': f"{prob*100:.1f}%" if np.isfinite(prob) else "—",
                'BTC': f"{btc:.6f}",
                'EV': f"€{ev:,.0f}" if np.isfinite(ev) else "—",
                'Exp Ret': f"{(ev/budget-1)*100:+.1f}%" if np.isfinite(ev) else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 📅 Dynamic DCA")
        if CH < 0.30:
            alloc = [('Lump sum', 0.60), ('Reserve', 0.40)]
            note = "Low channel → favor immediate"
        elif CH < 0.50 and RR < 0.5:
            alloc = [('Immediate', 0.40), ('Weekly DCA', 0.30), ('Reserve', 0.30)]
            note = "Mid, moderate risk → balanced"
        elif RR > 0.6:
            alloc = [('Immediate', 0.15), ('Weekly DCA', 0.35), ('Reserve', 0.50)]
            note = "High risk → conservative"
        else:
            alloc = [('Immediate', 0.30), ('Weekly DCA', 0.30), ('Reserve', 0.40)]
            note = "Standard"
        st.info(f"💡 **{note}** (channel={CH*100:.0f}%, risk={RR*100:.0f}%)")
        dca = []
        for comp, pct in alloc:
            amt = budget * pct
            if 'Immediate' in comp or 'Lump' in comp:
                dca.append({'': comp, '%': f"{pct*100:.0f}", '€': f"{amt:,.0f}",
                            'Action': f"Buy @ ${P:,.0f}", 'BTC': f"{amt/P:.6f}"})
            elif 'Weekly' in comp:
                w = hor_m * 4
                dca.append({'': comp, '%': f"{pct*100:.0f}", '€': f"{amt:,.0f}",
                            'Action': f"€{amt/w:,.0f}/wk × {w}", 'BTC': "varies"})
            else:
                dca.append({'': comp, '%': f"{pct*100:.0f}", '€': f"{amt:,.0f}",
                            'Action': "Deploy on dips (-20/-30/Floor)", 'BTC': "cond."})
        st.dataframe(pd.DataFrame(dca), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Projections")
        prows = []
        for y in [1, 2, 3, 4]:
            fd = TDAYS + y*365
            prows.append({
                'Y': f"{y}Y",
                'Floor': f"${M['floor']['a']*(fd**M['floor']['b']):,.0f}",
                'Balanced': f"${M['balanced']['a']*(fd**M['balanced']['b']):,.0f}",
                'Ceiling': f"${M['ceiling']['a']*(fd**M['ceiling']['b']):,.0f}",
            })
        st.dataframe(pd.DataFrame(prows), use_container_width=True, hide_index=True)
        st.error("⚠️ Models can be wrong. Past ≠ future. Only invest what you can afford to lose.")

    # ── TAB 5: DEBUG DATA ─────────────────────────────────────────
    with tab5:
        st.markdown('<div class="explain">'
                    '<b>Debug & Verification.</b> Download all computed data to check calculations.<br>'
                    '<b>Columns:</b> Date, Days (since genesis), Close, Low, NLB, NLB_Update (1=step up), '
                    'Floor, Balanced, Ceiling, MaxCross, ChPos (0-1), '
                    'DistBal (log distance to balanced, negative=below), '
                    'Excess (daily max(0,log(price/balanced))), '
                    'RiskArea (decayed sum), RiskRMax (rolling max), RiskRatio (0-1).'
                    '</div>', unsafe_allow_html=True)

        export = df[['Date','Days','Close','Low','NLB','Floor','Balanced','Ceiling',
                      'MaxCross','ChPos','DistBal','Excess','RiskArea','RiskRMax','RiskRatio']].copy()
        nlb_arr = export['NLB'].to_numpy()
        upd = np.zeros(len(nlb_arr), dtype=int)
        upd[0] = 1
        upd[1:] = (nlb_arr[1:] > nlb_arr[:-1]).astype(int)
        export.insert(5, 'NLB_Update', upd)

        # Check floor constraint
        floor_violations = int((export['Floor'] > export['NLB'] * 1.001).sum())

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Days", f"{len(export):,}")
        sc2.metric("NLB Steps", f"{upd.sum():,}")
        sc3.metric("Floor b", f"{M['floor']['b']:.4f}")
        sc4.metric("Floor Violations", f"{floor_violations}",
                   "✅ OK" if floor_violations == 0 else "❌ BUG")

        st.markdown("#### Parameters")
        st.code(
            f"Floor:     a = {M['floor']['a']:.10e},  b = {M['floor']['b']:.6f}\n"
            f"Balanced:  a = {M['balanced']['a']:.10e},  b = {M['balanced']['b']:.6f},  "
            f"k = {M['balanced']['k']:.6f}  (Bal/Floor = {BL/FL:.2f}x)\n"
            f"Ceiling:   a = {M['ceiling']['a']:.10e},  b = {M['ceiling']['b']:.6f}  "
            f"(Ceil/Floor = {CL/FL:.2f}x)\n"
            f"MaxCross:  a = {M['maxcross']['a']:.10e},  k = {M['maxcross']['k']:.6f},  "
            f"crossings = {M['maxcross']['n_crossings']}\n"
            f"Risk:      λ = {risk_lam},  window = {risk_win},  "
            f"reference = BALANCED\n"
            f"Data:      {export['Date'].iloc[0].strftime('%Y-%m-%d')} → "
            f"{export['Date'].iloc[-1].strftime('%Y-%m-%d')}",
            language='text')

        st.markdown("#### Last 30 days")
        st.dataframe(export.tail(30).round(4), use_container_width=True, hide_index=True)

        buf = io.StringIO()
        export.to_csv(buf, index=False)
        st.download_button("📥 Download CSV", buf.getvalue(),
                           f"btc_nlb_debug_{TODAY.strftime('%Y%m%d')}.csv",
                           'text/csv', use_container_width=True)

        params_json = json.dumps({
            'floor': M['floor'], 'balanced': M['balanced'],
            'ceiling': M['ceiling'], 'maxcross': M['maxcross'],
            'risk': {'lambda': risk_lam, 'window': risk_win, 'reference': 'balanced'},
            'data': {'start': export['Date'].iloc[0].strftime('%Y-%m-%d'),
                     'end': export['Date'].iloc[-1].strftime('%Y-%m-%d'),
                     'n': len(export)},
            'current': {'price': P, 'floor': FL, 'balanced': BL, 'ceiling': CL,
                        'channel_pct': round(CH*100, 2), 'risk_pct': round(RR*100, 2),
                        'dist_balanced_pct': round(D_BL, 2)},
        }, indent=2)
        st.download_button("📥 Download JSON", params_json,
                           f"btc_nlb_params_{TODAY.strftime('%Y%m%d')}.json",
                           'application/json', use_container_width=True)

    # Footer
    st.markdown("---")
    st.warning("⚠️ **DISCLAIMER** — Informational/educational only. NOT financial advice. "
               "No guarantees. You may lose all capital. Consult a financial advisor.")
    st.caption(f"v3 | Data: CryptoCompare | NLB Floor+Balanced+RiskArea | {TODAY.strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()
