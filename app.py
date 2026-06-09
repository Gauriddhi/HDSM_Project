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

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: inherit;
}
[data-testid="stHeader"] {
    background-color: transparent;
}
[data-testid="stToolbar"] {
    right: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("HDSM Control Panel")
save_results = st.sidebar.checkbox("Save Analysis Results")
show_info    = st.sidebar.checkbox("Project Information")
show_formula = st.sidebar.checkbox("Show HDSM Formula")

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

            **HOW IT WORKS**
            1. Historical data becomes the BASELINE.
            2. Incoming/new data becomes STREAMING DATA.
            3. The stream is divided into small sliding windows.
            4. Each window is statistically compared with the baseline.
            5. Weights and threshold are auto-calibrated from the baseline.
            6. Final drift score: Stable / Moderate Drift / High Drift

            **LIVE DATA SOURCES**
            * Tab 1: Yahoo Finance — real stock, crypto, index data
            * Tab 2: Open-Meteo  — real IoT weather sensor data
            * Tab 3: Healthcare  — synthetic patient population simulation
            * Tab 4: CSV / Excel — your own dataset
        """)

# =========================================================
# FORMULA
# =========================================================

if show_formula:
    with st.expander("HDSM Formula Reference", expanded=True):
        st.latex(r"D_t = \alpha \left|\mu_t - \mu_0\right| + \beta \,\mathrm{PSI}_t + \gamma \,\mathrm{KS}_t + \lambda \,S_t")
        st.write(r"• $| \mu_t - \mu_0 |$ = Confidence Drift")
        st.write(r"• $\mathrm{PSI}_t$      = Population Stability Index")
        st.write(r"• $\mathrm{KS}_t$       = Kolmogorov–Smirnov Statistic")
        st.write(r"• $S_t$         = Stability Penalty $| \mu_t - \mu_{t-1} |$")
        st.write(r"• $\alpha, \beta, \gamma, \lambda$   = Auto-calibrated weights")
        st.write(r"• $\delta$          = Auto-calibrated threshold ($\mu_{\text{base}} + 2\sigma_{\text{base}}$)")

st.info("Choose a data source tab below. HDSM auto-calibrates all weights and thresholds from your baseline data.")

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
# MAIN DRIFT RUNNER ENGINE
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
        st.error("Not enough streaming rows.")
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
    wc1.metric("α Confidence", alpha)
    wc2.metric("β PSI",        beta)
    wc3.metric("γ KS",         gamma)
    wc4.metric("λ Stability",  lam)
    wc5.metric("δ Threshold",  threshold)

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
    r3.metric("Final Severity",          final_severity)

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
# TABS SETUP
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
    
    presets = {
        "Apple (AAPL)":          "AAPL",
        "Tesla (TSLA)":          "TSLA",
        "Google (GOOGL)":        "GOOGL",
        "Bitcoin (BTC-USD)":     "BTC-USD",
        "Custom — type below":   "CUSTOM",
    }
    sel_preset = st.selectbox("Choose a stock / asset", list(presets.keys()), key="fin_preset")
    ticker = st.text_input("Enter ticker symbol", value="AAPL") if presets[sel_preset] == "CUSTOM" else presets[sel_preset]

    fc1, fc2 = st.columns(2)
    with fc1:
        sel_period = st.selectbox("Historical Period", ["6 Months","1 Year","2 Years"], index=1, key="fin_period")
        period = {"6 Months":"6mo","1 Year":"1y","2 Years":"2y"}[sel_period]
    with fc2:
        sel_interval = st.selectbox("Interval", ["Daily","Weekly"], key="fin_interval")
        interval = {"Daily":"1d","Weekly":"1wk"}[sel_interval]

    if st.button("Fetch Live Data & Run HDSM", type="primary", key="btn_finance"):
        with st.spinner(f"Fetching {ticker}..."):
            raw = yf.Ticker(ticker).history(period=period, interval=interval)
            if not raw.empty:
                df_live = raw[['Close', 'Open', 'High', 'Low', 'Volume']].reset_index(drop=True)
                
                with st.expander("Manual override (optional)"):
                    f_ov_a = st.number_input("Alpha", 0.0,1.0,0.0, key="fin_ov_a")
                    f_ov_b = st.number_input("Beta",  0.0,1.0,0.0, key="fin_ov_b")
                    f_ov_g = st.number_input("Gamma", 0.0,1.0,0.0, key="fin_ov_g")
                    f_ov_l = st.number_input("Lambda",0.0,1.0,0.0, key="fin_ov_l")
                    f_ov_t = st.number_input("Threshold δ",0.0,10.0,0.0, key="fin_ov_t")

                run_hdsm(df=df_live, target_col="Close", monitor_mode="Feature value",
                         feature_to_monitor="Close", target_is_classification=False,
                         ov_alpha=f_ov_a, ov_beta=f_ov_b, ov_gamma=f_ov_g, ov_lam=f_ov_l, ov_thresh=f_ov_t)

# =========================================================
# TAB 2 — OPEN-METEO WEATHER
# =========================================================
with tab2:
    st.markdown("### Live IoT Sensor Data — Weather")
    sel_city = st.selectbox("Choose a city", ["Mumbai, India", "London, UK", "New York, USA"], key="weather_city")
    lat, lon = {"Mumbai, India": (19.0760, 72.8777), "London, UK": (51.5074, -0.1278), "New York, USA": (40.7128, -74.0060)}[sel_city]
    
    weather_days = st.slider("Days of historical data", 30, 365, 180, key="weather_days")
    feat_col = st.selectbox("Sensor to monitor", ["Temperature", "Humidity", "Wind_Speed"], key="weather_sensor")

    if st.button("Fetch Weather Data & Run HDSM", type="primary", key="btn_weather"):
        with st.spinner("Fetching weather records..."):
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2025-01-01&end_date=2025-06-01&daily=temperature_2m_max,relative_humidity_2m_mean,wind_speed_10m_max&timezone=auto"
            resp = requests.get(url, timeout=15).json()
            if "daily" in resp:
                df_weather = pd.DataFrame({
                    "Temperature": resp["daily"].get("temperature_2m_max", []),
                    "Humidity": resp["daily"].get("relative_humidity_2m_mean", []),
                    "Wind_Speed": resp["daily"].get("wind_speed_10m_max", [])
                }).dropna().reset_index(drop=True)
                
                with st.expander("Manual override (optional)"):
                    w_ov_a = st.number_input("Alpha", 0.0,1.0,0.0, key="w_ov_a")
                    w_ov_b = st.number_input("Beta",  0.0,1.0,0.0, key="w_ov_b")
                    w_ov_g = st.number_input("Gamma", 0.0,1.0,0.0, key="w_ov_g")
                    w_ov_l = st.number_input("Lambda",0.0,1.0,0.0, key="w_ov_l")
                    w_ov_t = st.number_input("Threshold δ",0.0,10.0,0.0, key="w_ov_t")

                run_hdsm(df=df_weather, target_col=feat_col, monitor_mode="Feature value",
                         feature_to_monitor=feat_col, target_is_classification=False,
                         ov_alpha=w_ov_a, ov_beta=w_ov_b, ov_gamma=w_ov_g, ov_lam=w_ov_l, ov_thresh=w_ov_t)

# =========================================================
# TAB 3 — HEALTHCARE SYNTHETIC SIMULATION
# =========================================================
with tab3:
    st.markdown("### Healthcare Patient Data — Drift Simulation")
    hc_scenarios = {
        "Diabetes Progression — High Drift": {"stable": (100, 15), "drifted": (155, 25), "target": "glucose"},
        "Stable Hospital — Low Drift": {"stable": (99, 14), "drifted": (101, 14), "target": "glucose"}
    }
    sel_scenario = st.selectbox("Choose a healthcare scenario", list(hc_scenarios.keys()), key="hc_scen")
    scen = hc_scenarios[sel_scenario]
    
    n_stable  = st.slider("Stable patients (baseline)", 100, 800, 350, key="hc_stable")
    n_drifted = st.slider("Drifted patients (stream)",  100, 800, 250, key="hc_drift")

    if st.button("Generate Data & Run HDSM", type="primary", key="btn_health"):
        np.random.seed(42)
        stable_df = pd.DataFrame({"glucose": np.random.normal(scen["stable"][0], scen["stable"][1], n_stable), "outcome": np.random.choice([0,1], n_stable)})
        drifted_df = pd.DataFrame({"glucose": np.random.normal(scen["drifted"][0], scen["drifted"][1], n_drifted), "outcome": np.random.choice([0,1], n_drifted)})
        df_health = pd.concat([stable_df, drifted_df], ignore_index=True)

        with st.expander("Manual override (optional)"):
            h_ov_a = st.number_input("Alpha", 0.0,1.0,0.0, key="h_ov_a")
            h_ov_b = st.number_input("Beta",  0.0,1.0,0.0, key="h_ov_b")
            h_ov_g = st.number_input("Gamma", 0.0,1.0,0.0, key="h_ov_g")
            h_ov_l = st.number_input("Lambda",0.0,1.0,0.0, key="h_ov_l")
            h_ov_t = st.number_input("Threshold δ",0.0,10.0,0.0, key="h_ov_t")

        run_hdsm(df=df_health, target_col="outcome", monitor_mode="Feature value",
                 feature_to_monitor=scen["target"], target_is_classification=True,
                 ov_alpha=h_ov_a, ov_beta=h_ov_b, ov_gamma=h_ov_g, ov_lam=h_ov_l, ov_thresh=h_ov_t,
                 baseline_override=n_stable)

# =========================================================
# TAB 4 — CSV / EXCEL UPLOAD
# =========================================================
with tab4:
    st.markdown("### Upload Your Own Dataset")
    uploaded_file = st.file_uploader("Upload Dataset (CSV or Excel)", type=["csv","xlsx","xls"], key="uploader")

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith((".xlsx",".xls")) else pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        cols = df.columns.tolist()
        target_col = st.selectbox("Select Target Column", cols, key="csv_target")
        monitor_mode = st.selectbox("Monitoring Mode", ["Model confidence","Feature value"], key="csv_mode")

        feature_to_monitor = None
        if monitor_mode == "Feature value":
            num_cols = [c for c in cols if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
            feature_to_monitor = st.selectbox("Select Feature to Monitor", num_cols, key="csv_feat")

        if st.button("Run Drift Detection", key="btn_csv"):
            run_hdsm(df=df, target_col=target_col, monitor_mode=monitor_mode,
                     feature_to_monitor=feature_to_monitor, target_is_classification=True,
                     ov_alpha=0.0, ov_beta=0.0, ov_gamma=0.0, ov_lam=0.0, ov_thresh=0.0)
