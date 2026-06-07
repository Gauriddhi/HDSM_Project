import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ks_2samp

import yfinance as yf
import requests

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(page_title="HDSM Drift Monitoring System", layout="wide")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("HDSM Control Panel")
theme        = st.sidebar.selectbox("Theme Mode", ["Light Mode", "Dark Mode"])
save_results = st.sidebar.checkbox("Save Analysis Results")
show_info    = st.sidebar.checkbox("Project Information")
show_formula = st.sidebar.checkbox("Show HDSM Formula")

# =========================================================
# DARK MODE
# =========================================================
if "prev_theme" not in st.session_state:
    st.session_state.prev_theme = theme
if st.session_state.prev_theme != theme:
    st.session_state.prev_theme = theme
    st.rerun()
    
if theme == "Dark Mode":
    css = """
    <style>
    [data-testid="stAppViewContainer"], .main {background-color: #0E1117!important;}
    [data-testid="stHeader"] {background-color: #0E1117!important;}
    html, body, [class*="css"] {color: white!important;}
    .stButton>button {background-color: #262730!important; color: white!important; border: 1px solid #444!important;}
    </style>"""
else:
    css = """
    <style>
    [data-testid="stAppViewContainer"], .main {background-color: #FFFFFF!important;}
    [data-testid="stHeader"] {background-color: #FFFFFF!important;}
    html, body, [class*="css"] {color: black!important;}
    .stButton>button {background-color: #F0F2F6!important; color: black!important; border: 1px solid #ddd!important;}
    </style>"""
    
st.markdown(css, unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.title("HDSM Drift Monitoring System")
st.write("Real-Time Dataset Drift Detection and Machine Learning Reliability Monitoring")

# =========================================================
# PROJECT INFORMATION
# =========================================================

if show_info:
    with st.expander("What is HDSM and How Does It Work?"):
        st.write("""
            HDSM (Hybrid Drift Stability Model) helps organizations monitor
            whether incoming real-time data is becoming different from historical baseline data.

            HOW IT WORKS
            1. Historical data becomes the BASELINE.
            2. Incoming/new data becomes STREAMING DATA.
            3. The stream is divided into small sliding windows.
            4. Each window is statistically compared with the baseline.
            5. Weights and threshold are auto-calibrated from the baseline.
            6. Final drift score: Stable / Moderate Drift / High Drift

            LIVE DATA SOURCES
            Tab 1: Yahoo Finance — real stock, crypto, index data
            Tab 2: Open-Meteo  — real IoT weather sensor data
            Tab 3: Healthcare  — synthetic patient population simulation
            Tab 4: CSV / Excel — your own dataset

            NOTE: Kafka / AWS Kinesis integration is left as future work.
        """)

# =========================================================
# FORMULA
# =========================================================

if show_formula:
    with st.expander("HDSM Formula Reference", expanded=True):
        st.latex(r"D_t = \alpha \left|\mu_t - \mu_0\right| + \beta \,\mathrm{PSI}_t + \gamma \,\mathrm{KS}_t + \lambda \,S_t")
        st.write(r"• |μt − μ0| = Confidence Drift")
        st.write(r"• PSIt      = Population Stability Index")
        st.write(r"• KSt       = Kolmogorov–Smirnov Statistic")
        st.write(r"• St        = Stability Penalty |μt − μt−1|")
        st.write(r"• α β γ λ   = Auto-calibrated weights")
        st.write(r"• δ         = Auto-calibrated threshold (μ_base + 2σ_base)")

st.info("Choose a data source tab below. HDSM auto-calibrates all weights and threshold from your baseline data.")

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def calculate_psi(expected, actual, bins=10):
    expected = np.array(expected)
    actual   = np.array(actual)
    breakpoints   = np.linspace(0, 100, bins + 1)
    expected_bins = np.unique(np.percentile(expected, breakpoints))
    if len(expected_bins) < 2:
        return 0.0
    expected_counts = np.histogram(expected, bins=expected_bins)[0]
    actual_counts   = np.histogram(actual,   bins=expected_bins)[0]
    expected_ratios = expected_counts / len(expected)
    actual_ratios   = actual_counts   / len(actual)
    psi = np.sum(
        (expected_ratios - actual_ratios)
        * np.log((expected_ratios + 1e-6) / (actual_ratios + 1e-6))
    )
    return float(min(psi, 1.0))


def classify_drift(score, threshold):
    if score > threshold * 1.5:
        return "High Drift"
    elif score > threshold:
        return "Moderate Drift"
    else:
        return "Low Drift"


def preprocess_features(df):
    X = df.copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns
    dropped = []
    for col in cat_cols:
        if X[col].nunique() > 50:
            dropped.append(col)
            X = X.drop(columns=[col])
    if dropped:
        st.warning("Removed high-cardinality columns: " + ", ".join(dropped))
    remaining_cat = X.select_dtypes(include=["object", "category", "bool"]).columns
    if len(remaining_cat) > 0:
        X[remaining_cat] = X[remaining_cat].fillna("missing")
    X = pd.get_dummies(X, drop_first=True)
    if X.shape[1] > 100:
        variances = X.select_dtypes(include=[np.number]).var()
        top_cols  = variances.nlargest(100).index
        dropped_n = X.shape[1] - 100
        X = X[top_cols]
        st.warning(f"Kept top 100 columns by variance (dropped {dropped_n} low-signal columns).")
    return X


def get_auto_split_sizes(n_rows, stream_ratio=0.15):
    stream_rows   = max(20, int(n_rows * stream_ratio))
    baseline_rows = n_rows - stream_rows
    return baseline_rows, stream_rows


def calibrate_weights(baseline_signal, baseline_df, window_size,
                      monitor_mode, feature_to_monitor, mu_0,
                      model=None, baseline_X=None, is_classifier=True):
    conf_drifts, psi_scores, ks_scores, stabilities = [], [], [], []
    prev_mu = mu_0
    n    = len(baseline_df)
    half = n // 2
    ref  = baseline_signal[:half]
    for i in range(0, half, window_size):
        wdf = baseline_df.iloc[half + i: half + i + window_size]
        if len(wdf) < window_size:
            break
        if monitor_mode == "Model confidence" and baseline_X is not None:
            Xw = baseline_X.iloc[half + i: half + i + window_size]
            actual = model.predict_proba(Xw).max(axis=1) if is_classifier else model.predict(Xw)
            mu_t = float(np.mean(actual))
        else:
            actual = wdf[feature_to_monitor].values
            mu_t   = float(np.mean(actual))
        conf_drifts.append(abs(mu_t - mu_0))
        psi_scores.append(min(calculate_psi(ref, actual), 1.0))
        ks_val, _ = ks_2samp(ref, actual)
        ks_scores.append(ks_val)
        stabilities.append(abs(mu_t - prev_mu))
        prev_mu = mu_t
    if len(conf_drifts) < 2:
        return 0.25, 0.25, 0.25, 0.25
    sc = np.std(conf_drifts) + 1e-9
    sp = np.std(psi_scores)  + 1e-9
    sk = np.std(ks_scores)   + 1e-9
    ss = np.std(stabilities) + 1e-9
    total = sc + sp + sk + ss
    return round(sc/total,4), round(sp/total,4), round(sk/total,4), round(ss/total,4)


def calibrate_threshold(baseline_signal, baseline_df, window_size,
                        alpha, beta, gamma, lam,
                        monitor_mode, feature_to_monitor, mu_0,
                        model=None, baseline_X=None, is_classifier=True):
    baseline_Dt = []
    prev_mu = mu_0
    n    = len(baseline_df)
    half = n // 2
    ref  = baseline_signal[:half]
    for i in range(0, half, window_size):
        wdf = baseline_df.iloc[half + i: half + i + window_size]
        if len(wdf) < window_size:
            break
        if monitor_mode == "Model confidence" and baseline_X is not None:
            Xw = baseline_X.iloc[half + i: half + i + window_size]
            actual = model.predict_proba(Xw).max(axis=1) if is_classifier else model.predict(Xw)
            mu_t = float(np.mean(actual))
        else:
            actual = wdf[feature_to_monitor].values
            mu_t   = float(np.mean(actual))
        cd    = abs(mu_t - mu_0)
        psi   = min(calculate_psi(ref, actual), 1.0)
        ks, _ = ks_2samp(ref, actual)
        stab  = abs(mu_t - prev_mu)
        Dt    = alpha * cd + beta * psi + gamma * ks + lam * stab
        baseline_Dt.append(Dt)
        prev_mu = mu_t
    if len(baseline_Dt) < 2:
        return 0.2
    auto = float(np.mean(baseline_Dt) + 2 * np.std(baseline_Dt))
    return round(max(auto, 0.05), 4)


# =========================================================
# SHARED DRIFT RUNNER
# baseline_override: pass exact row count for baseline
# when you want a clean split (used by healthcare tab)
# =========================================================

def run_hdsm(df, target_col, monitor_mode, feature_to_monitor,
             target_is_classification, ov_alpha, ov_beta,
             ov_gamma, ov_lam, ov_thresh,
             baseline_override=None):

    if baseline_override is not None:
        baseline_size = baseline_override
        stream_rows   = len(df) - baseline_size
    else:
        baseline_size, stream_rows = get_auto_split_sizes(len(df))

    window_size = max(5, min(50, stream_rows // 3))

    st.subheader("Auto Configuration")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Baseline Rows",  baseline_size)
    cc2.metric("Streaming Rows", stream_rows)
    cc3.metric("Window Size",    window_size)

    if len(df) < 50:
        st.error("Dataset needs at least 50 rows.")
        return
    if stream_rows < window_size:
        st.error("Not enough streaming rows. Use a larger dataset or more drifted patients.")
        return

    st.success("HDSM Drift Monitoring Started")

    baseline = df.iloc[:baseline_size]
    stream   = df.iloc[baseline_size:]

    model         = None
    baseline_X    = None
    stream_X      = None
    is_classifier = target_is_classification

    if monitor_mode == "Model confidence":
        X          = preprocess_features(df.drop(columns=[target_col]))
        baseline_X = X.iloc[:baseline_size]
        stream_X   = X.iloc[baseline_size:]
        y_base     = df[target_col].iloc[:baseline_size]
        if is_classifier:
            y_base_enc = pd.factorize(y_base)[0]
            model      = LogisticRegression(max_iter=1000)
            model.fit(baseline_X, y_base_enc)
            expected   = model.predict_proba(baseline_X).max(axis=1)
        else:
            model    = RandomForestRegressor(random_state=42)
            model.fit(baseline_X, y_base)
            expected = model.predict(baseline_X)
        mu_0 = float(np.mean(expected))
    else:
        expected = baseline[feature_to_monitor].values
        mu_0     = float(np.mean(expected))

    with st.spinner("Auto-calibrating weights from baseline..."):
        auto_a, auto_b, auto_g, auto_l = calibrate_weights(
            baseline_signal=expected, baseline_df=baseline,
            window_size=window_size, monitor_mode=monitor_mode,
            feature_to_monitor=feature_to_monitor, mu_0=mu_0,
            model=model, baseline_X=baseline_X, is_classifier=is_classifier)

    if (ov_alpha + ov_beta + ov_gamma + ov_lam) > 0:
        alpha, beta, gamma, lam = ov_alpha, ov_beta, ov_gamma, ov_lam
        w_src = "Manual Override"
    else:
        alpha, beta, gamma, lam = auto_a, auto_b, auto_g, auto_l
        w_src = "Auto-Calibrated from Baseline"

    with st.spinner("Auto-calibrating threshold from baseline..."):
        auto_thresh = calibrate_threshold(
            baseline_signal=expected, baseline_df=baseline,
            window_size=window_size, alpha=alpha, beta=beta,
            gamma=gamma, lam=lam, monitor_mode=monitor_mode,
            feature_to_monitor=feature_to_monitor, mu_0=mu_0,
            model=model, baseline_X=baseline_X, is_classifier=is_classifier)

    threshold = ov_thresh if ov_thresh > 0 else auto_thresh
    t_src     = "Manual Override" if ov_thresh > 0 else "Auto-Calibrated from Baseline"

    st.subheader("Calibration Summary")
    st.caption(f"Weights: {w_src}   |   Threshold: {t_src}")
    wc1, wc2, wc3, wc4, wc5 = st.columns(5)
    wc1.metric("α  Confidence", alpha)
    wc2.metric("β  PSI",        beta)
    wc3.metric("γ  KS",         gamma)
    wc4.metric("λ  Stability",  lam)
    wc5.metric("δ  Threshold",  threshold)

    drift_scores, severity_list = [], []
    conf_drift_list, psi_list, ks_list, stability_list, drift_deriv_list = [], [], [], [], []
    prev_mu = mu_0
    prev_Dt = None

    for i in range(0, len(stream), window_size):
        window = stream.iloc[i:i + window_size]
        if len(window) < window_size:
            continue
        if monitor_mode == "Model confidence":
            Xw     = stream_X.iloc[i:i + window_size]
            actual = model.predict_proba(Xw).max(axis=1) if is_classifier else model.predict(Xw)
            mu_t   = float(np.mean(actual))
        else:
            actual = window[feature_to_monitor].values
            mu_t   = float(np.mean(actual))

        cd    = abs(mu_t - mu_0)
        psi   = min(calculate_psi(expected, actual), 1.0)
        ks, _ = ks_2samp(expected, actual)
        stab  = abs(mu_t - prev_mu)
        D_t   = alpha * cd + beta * psi + gamma * ks + lam * stab
        D_prime  = D_t - prev_Dt if prev_Dt is not None else 0.0
        severity = classify_drift(D_t, threshold)

        drift_scores.append(D_t)
        severity_list.append(severity)
        conf_drift_list.append(cd)
        psi_list.append(psi)
        ks_list.append(ks)
        stability_list.append(stab)
        drift_deriv_list.append(D_prime)
        prev_mu = mu_t
        prev_Dt = D_t

    if len(drift_scores) == 0:
        st.warning("No full windows found. Use a larger dataset.")
        return

    final_score    = drift_scores[-1]
    final_severity = severity_list[-1]
    drift_status   = "Drift Detected" if final_score > threshold else "No Drift"

    st.subheader("Final Results")
    r1, r2, r3 = st.columns(3)
    r1.metric("Final Drift Score (Dt)", round(final_score, 4))
    r2.metric("Drift Status",           drift_status)
    r3.metric("Final Severity",         final_severity)

    st.subheader("Recommendations")
    if final_severity == "High Drift":
        st.error("High drift detected. Model retraining is recommended immediately.")
    elif final_severity == "Moderate Drift":
        st.warning("Moderate drift detected. Monitor closely and consider retraining soon.")
    else:
        st.success("Dataset is stable relative to its own baseline. No action required.")

    windows_idx = list(range(0, len(drift_scores) * window_size, window_size))

    result_df = pd.DataFrame({
        "Window":            windows_idx,
        "Confidence Drift":  [round(v,4) for v in conf_drift_list],
        "PSI":               [round(v,4) for v in psi_list],
        "KS":                [round(v,4) for v in ks_list],
        "Stability":         [round(v,4) for v in stability_list],
        "Drift Score (Dt)":  [round(v,4) for v in drift_scores],
        "Drift Speed (D't)": [round(v,4) for v in drift_deriv_list],
        "Severity":          severity_list,
    })

    st.subheader("Drift Monitoring Graph")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    y_top = max(max(drift_scores) * 1.4, threshold * 2.5, 0.5)
    ax1.axhspan(0,               threshold,       alpha=0.12, color="green",  label="Stable zone")
    ax1.axhspan(threshold,       threshold * 1.5, alpha=0.12, color="orange", label="Moderate zone")
    ax1.axhspan(threshold * 1.5, y_top,           alpha=0.12, color="red",    label="High zone")
    ax1.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, label=f"Auto threshold δ = {threshold}")
    ax1.plot(windows_idx, drift_scores, marker="o", linewidth=2, color="steelblue", label="HDSM Drift Score (Dt)")
    ax1.set_ylim(0, y_top)
    ax1.set_xlabel("Streaming Window")
    ax1.set_ylabel("Drift Score (Dt)")
    ax1.set_title("HDSM Dataset Drift Over Time")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    st.subheader("Component Breakdown")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(windows_idx, conf_drift_list, marker="o", label=f"Confidence Drift (α={alpha})")
    ax2.plot(windows_idx, psi_list,        marker="s", label=f"PSI (β={beta})")
    ax2.plot(windows_idx, ks_list,         marker="^", label=f"KS Statistic (γ={gamma})")
    ax2.plot(windows_idx, stability_list,  marker="D", label=f"Stability Penalty (λ={lam})")
    ax2.axhline(y=threshold, color="red", linestyle=":", linewidth=1, label=f"Threshold δ={threshold}")
    ax2.set_xlabel("Streaming Window")
    ax2.set_ylabel("Component Score")
    ax2.set_title("Individual Drift Components Over Time")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    st.subheader("Auto-Calibrated Weight Distribution")
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.pie([alpha, beta, gamma, lam],
            labels=[f"α Confidence\n{alpha}", f"β PSI\n{beta}",
                    f"γ KS\n{gamma}", f"λ Stability\n{lam}"],
            colors=["steelblue", "darkorange", "green", "purple"],
            autopct="%1.1f%%", startangle=140)
    ax3.set_title("Weight share per drift component")
    st.pyplot(fig3)

    st.subheader("Detailed Results")
    st.dataframe(result_df)

    if save_results:
        result_df.to_csv("hdsm_results.csv", index=False)
        with open("hdsm_results.csv", "rb") as f:
            st.download_button(label="Download Results CSV", data=f,
                               file_name="hdsm_results.csv", mime="text/csv")


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Finance — Yahoo Finance",
    "🌦️  IoT — Weather Sensors",
    "🏥  Healthcare — Patient Simulation",
    "📂  Upload CSV / Excel",
])


# =========================================================
# TAB 1 — YAHOO FINANCE
# =========================================================

with tab1:
    st.markdown("### Live Stock / Crypto Data — Yahoo Finance")
    st.caption("Real market data fetched directly from the cloud. No file upload needed.")

    presets = {
        "Apple (AAPL)":          "AAPL",
        "Tesla (TSLA)":          "TSLA",
        "Google (GOOGL)":        "GOOGL",
        "Microsoft (MSFT)":      "MSFT",
        "Amazon (AMZN)":         "AMZN",
        "Reliance India":        "RELIANCE.NS",
        "TCS India":             "TCS.NS",
        "Bitcoin (BTC-USD)":     "BTC-USD",
        "Gold (GC=F)":           "GC=F",
        "S&P 500 (^GSPC)":       "^GSPC",
        "Custom — type below":   "CUSTOM",
    }

    sel_preset = st.selectbox("Choose a stock / asset", list(presets.keys()))

    if presets[sel_preset] == "CUSTOM":
        ticker = st.text_input("Enter ticker symbol",
                               placeholder="e.g. INFY.NS, NFLX, ETH-USD").upper().strip()
    else:
        ticker = presets[sel_preset]
        st.caption(f"Selected ticker: **{ticker}**")

    fc1, fc2 = st.columns(2)
    with fc1:
        period_map = {"6 Months":"6mo","1 Year":"1y","2 Years":"2y","5 Years":"5y"}
        sel_period = st.selectbox("Historical Period", list(period_map.keys()), index=1)
        period     = period_map[sel_period]
    with fc2:
        interval_map = {"Daily":"1d","Weekly":"1wk"}
        sel_interval = st.selectbox("Interval", list(interval_map.keys()), index=0)
        interval     = interval_map[sel_interval]

    if st.button("Fetch Live Data & Run HDSM", type="primary", key="btn_finance"):
        if not ticker:
            st.error("Please enter a ticker symbol.")
        else:
            with st.spinner(f"Fetching {ticker} from Yahoo Finance..."):
                try:
                    raw = yf.Ticker(ticker).history(period=period, interval=interval)
                    if raw.empty:
                        st.error(f"No data for '{ticker}'. Check symbol. Indian stocks need .NS suffix.")
                    else:
                        raw = raw.reset_index()
                        raw.columns = [str(c).replace(" ","_") for c in raw.columns]
                        for col in ["Date","Datetime"]:
                            if col in raw.columns:
                                raw[col] = pd.to_datetime(raw[col]).dt.tz_localize(None)
                        keep    = [c for c in ["Open","High","Low","Close","Volume"] if c in raw.columns]
                        df_live = raw[keep].dropna().reset_index(drop=True)

                        if len(df_live) < 50:
                            st.error(f"Only {len(df_live)} rows. Try a longer period.")
                        else:
                            st.success(f"Fetched **{len(df_live)} rows** for **{ticker}**")
                            st.dataframe(df_live.head())
                            lc1, lc2 = st.columns(2)
                            lc1.metric("Rows", df_live.shape[0])
                            lc2.metric("Columns", df_live.shape[1])
                            st.info("Monitoring **Close price** for market drift.")

                            with st.expander("Manual override (optional)"):
                                st.caption("Leave at 0 for auto-calibration.")
                                lo1, lo2 = st.columns(2)
                                with lo1:
                                    lov_a = st.number_input("Alpha",  0.0,1.0,0.0,0.01,key="la")
                                    lov_b = st.number_input("Beta",   0.0,1.0,0.0,0.01,key="lb")
                                with lo2:
                                    lov_g = st.number_input("Gamma",  0.0,1.0,0.0,0.01,key="lg")
                                    lov_l = st.number_input("Lambda", 0.0,1.0,0.0,0.01,key="ll")
                                lov_t = st.number_input("Threshold δ",0.0,10.0,0.0,0.01,key="lt")

                            run_hdsm(df=df_live, target_col="Close",
                                     monitor_mode="Feature value",
                                     feature_to_monitor="Close",
                                     target_is_classification=False,
                                     ov_alpha=lov_a, ov_beta=lov_b,
                                     ov_gamma=lov_g, ov_lam=lov_l, ov_thresh=lov_t)

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.caption("Indian stocks: add .NS  |  Crypto: add -USD")


# =========================================================
# TAB 2 — OPEN-METEO WEATHER / IOT
# =========================================================

with tab2:
    st.markdown("### Live IoT Sensor Data — Weather (Open-Meteo)")
    st.caption("Real daily weather readings for any city. Free, no API key needed.")

    city_presets = {
        "Mumbai, India":    (19.0760, 72.8777),
        "Delhi, India":     (28.6139, 77.2090),
        "Bangalore, India": (12.9716, 77.5946),
        "London, UK":       (51.5074, -0.1278),
        "New York, USA":    (40.7128, -74.0060),
        "Tokyo, Japan":     (35.6762, 139.6503),
        "Custom location":  None,
    }

    sel_city = st.selectbox("Choose a city", list(city_presets.keys()))

    if city_presets[sel_city] is None:
        wc1, wc2 = st.columns(2)
        lat = wc1.number_input("Latitude",  value=19.07, format="%.4f")
        lon = wc2.number_input("Longitude", value=72.87, format="%.4f")
    else:
        lat, lon = city_presets[sel_city]
        st.caption(f"Coordinates: {lat}, {lon}")

    weather_days = st.slider("Days of historical data", 30, 365, 180)

    sensor_map = {
        "Temperature (°C)": "Temperature",
        "Humidity (%)":      "Humidity",
        "Wind Speed (km/h)": "Wind_Speed",
        "Rainfall (mm)":     "Rainfall",
    }
    sel_sensor = st.selectbox("Sensor to monitor", list(sensor_map.keys()))
    feat_col   = sensor_map[sel_sensor]

    if st.button("Fetch Weather Data & Run HDSM", type="primary", key="btn_weather"):
        with st.spinner(f"Fetching {weather_days} days of weather for {sel_city}..."):
            try:
                from datetime import datetime, timedelta
                end_date   = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=weather_days)).strftime("%Y-%m-%d")
                url = (f"https://archive-api.open-meteo.com/v1/archive"
                       f"?latitude={lat}&longitude={lon}"
                       f"&start_date={start_date}&end_date={end_date}"
                       f"&daily=temperature_2m_max,precipitation_sum,"
                       f"wind_speed_10m_max,relative_humidity_2m_mean"
                       f"&timezone=auto")

                resp = requests.get(url, timeout=15)
                data = resp.json()

                if "daily" not in data:
                    st.error("Could not fetch weather data. Try a different location.")
                else:
                    daily = data["daily"]
                    df_weather = pd.DataFrame({
                        "Date":        daily.get("time", []),
                        "Temperature": daily.get("temperature_2m_max", []),
                        "Humidity":    daily.get("relative_humidity_2m_mean", []),
                        "Wind_Speed":  daily.get("wind_speed_10m_max", []),
                        "Rainfall":    daily.get("precipitation_sum", []),
                    })
                    df_weather = df_weather.dropna().reset_index(drop=True)

                    st.success(f"Fetched **{len(df_weather)} daily readings** for **{sel_city}**")
                    st.dataframe(df_weather.head())
                    wm1, wm2 = st.columns(2)
                    wm1.metric("Rows", df_weather.shape[0])
                    wm2.metric("Columns", df_weather.shape[1])
                    st.info(f"Monitoring **{sel_sensor}** for IoT sensor drift.")

                    with st.expander("Manual override (optional)"):
                        st.caption("Leave at 0 for auto-calibration.")
                        wo1, wo2 = st.columns(2)
                        with wo1:
                            wov_a = st.number_input("Alpha",  0.0,1.0,0.0,0.01,key="wa")
                            wov_b = st.number_input("Beta",   0.0,1.0,0.0,0.01,key="wb")
                        with wo2:
                            wov_g = st.number_input("Gamma",  0.0,1.0,0.0,0.01,key="wg")
                            wov_l = st.number_input("Lambda", 0.0,1.0,0.0,0.01,key="wl")
                        wov_t = st.number_input("Threshold δ",0.0,10.0,0.0,0.01,key="wt")

                    run_hdsm(df=df_weather, target_col=feat_col,
                             monitor_mode="Feature value",
                             feature_to_monitor=feat_col,
                             target_is_classification=False,
                             ov_alpha=wov_a, ov_beta=wov_b,
                             ov_gamma=wov_g, ov_lam=wov_l, ov_thresh=wov_t)

            except Exception as e:
                st.error(f"Error fetching weather data: {e}")


# =========================================================
# TAB 3 — HEALTHCARE SYNTHETIC SIMULATION
# =========================================================

with tab3:
    st.markdown("### Healthcare Patient Data — Drift Simulation")
    st.caption(
        "Generates realistic synthetic patient data. "
        "Real patient APIs require hospital credentials (HIPAA/DPDP privacy laws). "
        "Synthetic simulation is standard in healthcare ML research."
    )
    st.info(
        "**How it works:** Generates STABLE patients as baseline and "
        "DRIFTED patients as stream. Baseline split is exact — "
        "no drifted data leaks into baseline. "
        "HDSM detects the population shift correctly."
    )

    hc_scenarios = {
        "Diabetes Progression — High Drift": {
            "desc": "Glucose and insulin rise significantly. Hospital patient mix shifting toward diabetic cases. Expect High Drift.",
            "stable":  {"glucose":(100,15),"blood_pressure":(80,10),"bmi":(26,4),"age":(40,12),"insulin":(80,20)},
            "drifted": {"glucose":(155,25),"blood_pressure":(98,15),"bmi":(34,6),"age":(58,10),"insulin":(190,45)},
            "target":  "glucose",
        },
        "Post-COVID Patient Shift — Moderate Drift": {
            "desc": "Mild increase in age, BMI, and blood pressure post-COVID. Expect Moderate Drift.",
            "stable":  {"glucose":(95,12),"blood_pressure":(75,8),"bmi":(24,3),"age":(35,10),"insulin":(75,18)},
            "drifted": {"glucose":(103,14),"blood_pressure":(84,10),"bmi":(28,4),"age":(45,11),"insulin":(95,22)},
            "target":  "blood_pressure",
        },
        "Stable Hospital — Seasonal Variation — Low Drift": {
            "desc": "Very small natural seasonal variation in patient readings. Expect Low Drift — shows HDSM correctly identifies stability.",
            "stable":  {"glucose":(99,14),"blood_pressure":(79,9),"bmi":(25,4),"age":(40,12),"insulin":(79,20)},
            "drifted": {"glucose":(101,14),"blood_pressure":(81,9),"bmi":(26,4),"age":(41,12),"insulin":(82,20)},
            "target":  "glucose",
        },
        "Cardiac Risk Population — High Drift": {
            "desc": "Strongly rising age and blood pressure — aging patient population arriving. Expect High Drift.",
            "stable":  {"glucose":(98,14),"blood_pressure":(78,9),"bmi":(25,4),"age":(38,11),"insulin":(78,22)},
            "drifted": {"glucose":(125,22),"blood_pressure":(108,18),"bmi":(31,5),"age":(65,11),"insulin":(155,40)},
            "target":  "blood_pressure",
        },
    }

    sel_scenario = st.selectbox("Choose a healthcare scenario", list(hc_scenarios.keys()))
    scenario     = hc_scenarios[sel_scenario]
    st.caption(scenario["desc"])

    hc1, hc2 = st.columns(2)
    with hc1:
        n_stable  = st.slider("Stable patients (baseline)",  100, 800, 350, 50)
    with hc2:
        n_drifted = st.slider("Drifted patients (stream)",   100, 800, 250, 50)

    if st.button("Generate Data & Run HDSM", type="primary", key="btn_health"):

        np.random.seed(42)
        s = scenario["stable"]
        d = scenario["drifted"]

        stable_df = pd.DataFrame({
            "glucose":        np.random.normal(s["glucose"][0],        s["glucose"][1],        n_stable),
            "blood_pressure": np.random.normal(s["blood_pressure"][0], s["blood_pressure"][1], n_stable),
            "bmi":            np.random.normal(s["bmi"][0],            s["bmi"][1],            n_stable),
            "age":            np.random.normal(s["age"][0],            s["age"][1],            n_stable).clip(18,90),
            "insulin":        np.random.normal(s["insulin"][0],        s["insulin"][1],        n_stable).clip(0,500),
            "outcome":        np.random.choice([0,1], n_stable, p=[0.65,0.35]),
        })

        drifted_df = pd.DataFrame({
            "glucose":        np.random.normal(d["glucose"][0],        d["glucose"][1],        n_drifted),
            "blood_pressure": np.random.normal(d["blood_pressure"][0], d["blood_pressure"][1], n_drifted),
            "bmi":            np.random.normal(d["bmi"][0],            d["bmi"][1],            n_drifted),
            "age":            np.random.normal(d["age"][0],            d["age"][1],            n_drifted).clip(18,90),
            "insulin":        np.random.normal(d["insulin"][0],        d["insulin"][1],        n_drifted).clip(0,500),
            "outcome":        np.random.choice([0,1], n_drifted, p=[0.30,0.70]),
        })

        df_health = pd.concat([stable_df, drifted_df], ignore_index=True).round(2)

        st.success(f"Generated **{len(df_health)} patient records** ({n_stable} stable + {n_drifted} drifted)")

        st.markdown("**Population Comparison — Stable vs Drifted:**")
        compare = pd.DataFrame({
            "Feature":      ["Glucose","Blood Pressure","BMI","Age","Insulin"],
            "Stable Mean":  [round(stable_df[c].mean(),1) for c in ["glucose","blood_pressure","bmi","age","insulin"]],
            "Drifted Mean": [round(drifted_df[c].mean(),1) for c in ["glucose","blood_pressure","bmi","age","insulin"]],
        })
        compare["Change"] = compare.apply(
            lambda r: f"+{round(r['Drifted Mean']-r['Stable Mean'],1)}"
            if r["Drifted Mean"] >= r["Stable Mean"]
            else str(round(r["Drifted Mean"]-r["Stable Mean"],1)), axis=1)
        st.dataframe(compare, use_container_width=True)
        st.dataframe(df_health.head())

        target_col_h = scenario["target"]
        st.info(f"Monitoring **{target_col_h}** for patient population drift.")

        with st.expander("Manual override (optional)"):
            st.caption("Leave at 0 for auto-calibration.")
            ho1, ho2 = st.columns(2)
            with ho1:
                hov_a = st.number_input("Alpha",  0.0,1.0,0.0,0.01,key="ha")
                hov_b = st.number_input("Beta",   0.0,1.0,0.0,0.01,key="hb")
            with ho2:
                hov_g = st.number_input("Gamma",  0.0,1.0,0.0,0.01,key="hg")
                hov_l = st.number_input("Lambda", 0.0,1.0,0.0,0.01,key="hl")
            hov_t = st.number_input("Threshold δ",0.0,10.0,0.0,0.01,key="ht")

        # KEY FIX: baseline_override=n_stable ensures only stable patients
        # go into the baseline — no drifted data leaks in
        run_hdsm(
            df=df_health,
            target_col=target_col_h,
            monitor_mode="Feature value",
            feature_to_monitor=target_col_h,
            target_is_classification=False,
            ov_alpha=hov_a, ov_beta=hov_b,
            ov_gamma=hov_g, ov_lam=hov_l, ov_thresh=hov_t,
            baseline_override=n_stable,
        )


# =========================================================
# TAB 4 — CSV / EXCEL UPLOAD
# =========================================================

with tab4:
    st.markdown("### Upload Your Own Dataset")

    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV or Excel)",
        type=["csv","xlsx","xls"]
    )

    if uploaded_file is not None:
        fname = uploaded_file.name
        if fname.endswith((".xlsx",".xls")):
            df = pd.read_excel(uploaded_file)
            st.caption("Excel file loaded successfully.")
        else:
            try:
                df = pd.read_csv(uploaded_file)
                if df.shape[1] == 1:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=";", decimal=",")
                    st.caption("Semicolon-separated file detected and loaded correctly.")
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=";", decimal=",")

        df = df.dropna(how="all")

        if len(df) > 15_000:
            st.warning(f"Large dataset: {len(df):,} rows. Using first 15,000 rows.")
            df = df.head(15_000)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        c1, c2 = st.columns(2)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])

        cols         = df.columns.tolist()
        target_col   = st.selectbox("Select Target Column", cols)
        monitor_mode = st.selectbox("Monitoring Mode", ["Model confidence","Feature value"])

        feature_to_monitor = None
        if monitor_mode == "Feature value":
            numeric_features = [c for c in cols
                                if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_features:
                st.error("No numeric features. Switch to Model confidence mode.")
            else:
                feature_to_monitor = st.selectbox("Select Feature to Monitor", numeric_features)

        y_full = df[target_col]
        if pd.api.types.is_numeric_dtype(y_full) and y_full.nunique() > 10:
            target_is_classification = False
            st.info("Regression target detected.")
        else:
            target_is_classification = True
            st.info("Classification target detected.")

        with st.expander("Manual override (optional)"):
            st.caption("Leave at 0 for auto-calibration.")
            oc1, oc2 = st.columns(2)
            with oc1:
                ov_alpha = st.number_input("Alpha",  0.0,1.0,0.0,0.01,key="ca")
                ov_beta  = st.number_input("Beta",   0.0,1.0,0.0,0.01,key="cb")
            with oc2:
                ov_gamma = st.number_input("Gamma",  0.0,1.0,0.0,0.01,key="cg")
                ov_lam   = st.number_input("Lambda", 0.0,1.0,0.0,0.01,key="cl")
            ov_thresh = st.number_input("Threshold δ",0.0,10.0,0.0,0.01,key="ct")

        if st.button("Run Drift Detection", key="btn_csv"):
            if monitor_mode == "Feature value" and feature_to_monitor is None:
                st.error("Select a feature to monitor.")
            else:
                run_hdsm(
                    df=df, target_col=target_col,
                    monitor_mode=monitor_mode,
                    feature_to_monitor=feature_to_monitor,
                    target_is_classification=target_is_classification,
                    ov_alpha=ov_alpha, ov_beta=ov_beta,
                    ov_gamma=ov_gamma, ov_lam=ov_lam, ov_thresh=ov_thresh,
                )
