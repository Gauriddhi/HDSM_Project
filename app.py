import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ks_2samp


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HDSM Drift Monitoring System",
    layout="wide"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("HDSM Control Panel")

theme = st.sidebar.selectbox("Theme Mode", ["Light Mode", "Dark Mode"])
save_results = st.sidebar.checkbox("Save Analysis Results")
show_info    = st.sidebar.checkbox("Project Information")
show_formula = st.sidebar.checkbox("Show HDSM Formula")


# =========================================================
# DARK MODE
# =========================================================

if theme == "Dark Mode":
    st.markdown(
        """
        <style>
        .stApp { background-color: #0E1117; color: white; }
        .stButton>button { background-color: #262730; color: white; }
        .stDataFrame { background-color: #262730; }
        </style>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# TITLE
# =========================================================

st.title("HDSM Drift Monitoring System")
st.write(
    "Real-Time Dataset Drift Detection and "
    "Machine Learning Reliability Monitoring"
)


# =========================================================
# PROJECT INFORMATION
# =========================================================

if show_info:
    with st.expander("What is HDSM and How Does It Work?"):
        st.write(
            """
            HDSM (Hybrid Drift Stability Model) helps organizations monitor
            whether incoming real-time data is becoming different from
            historical baseline data.

            HOW IT WORKS
            1. Historical data becomes the BASELINE.
            2. Incoming/new data becomes STREAMING DATA.
            3. The stream is divided into small sliding windows.
            4. Each window is statistically compared with the baseline.
            5. Weights (α β γ λ) and threshold (δ) are auto-calibrated
               from the baseline — no manual tuning needed.
            6. Final drift score determines:
               Stable / Moderate Drift / High Drift

            NOTE ON CLOUD STREAMS
            CSV upload simulates streaming behaviour. Deployment to real
            cloud streams (Kafka, AWS Kinesis) is left as future work.
            """
        )


# =========================================================
# FORMULA — only shown if user toggles it in sidebar
# =========================================================

if show_formula:
    with st.expander("HDSM Formula Reference", expanded=True):
        st.latex(
            r"D_t = \alpha \left|\mu_t - \mu_0\right|"
            r"+ \beta \,\mathrm{PSI}_t"
            r"+ \gamma \,\mathrm{KS}_t"
            r"+ \lambda \,S_t"
        )
        st.write("Where:")
        st.write(r"• |μt − μ0|  =  Confidence Drift")
        st.write(r"• PSIt       =  Population Stability Index")
        st.write(r"• KSt        =  Kolmogorov–Smirnov Statistic")
        st.write(r"• St         =  Stability Penalty  |μt − μt−1|")
        st.write(r"• α β γ λ    =  Auto-calibrated weights")
        st.write(r"• δ          =  Auto-calibrated threshold  (μ_base + 2σ_base)")


# =========================================================
# INTRO INFO BOX
# =========================================================

st.info(
    "Upload any tabular CSV or Excel dataset. "
    "HDSM will automatically calibrate all weights and threshold "
    "from your baseline data and detect drift in the stream."
)


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
    return float(psi)


def classify_drift(score, auto_threshold):
    # Primary: relative to dataset-calibrated threshold
    # Secondary: absolute floors ensure large real drift
    # is never hidden behind a very small auto threshold
    if score > auto_threshold * 1.5 or score > 0.25:
        return "High Drift"
    elif score > auto_threshold or score > 0.10:
        return "Moderate Drift"
    else:
        return "Low Drift"


def preprocess_features(df):
    X = df.copy()

    # numeric: fill missing with median
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    # categorical: drop high-cardinality columns (>50 unique) to prevent
    # memory explosion from get_dummies on free-text / ID columns
    cat_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    dropped = []
    for col in cat_cols:
        if X[col].nunique() > 50:
            dropped.append(col)
            X = X.drop(columns=[col])

    if dropped:
        st.warning(
            "These columns were removed automatically — too many unique "
            "text values to encode safely:\n"
            + ", ".join(dropped)
        )

    remaining_cat = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns
    if len(remaining_cat) > 0:
        X[remaining_cat] = X[remaining_cat].fillna("missing")

    X = pd.get_dummies(X, drop_first=True)

    # Hard cap at 100 columns — keeps top columns by variance.
    # Logistic Regression cannot fit 22,000+ columns on a standard laptop.
    # Highest-variance columns carry the most signal for drift detection.
    if X.shape[1] > 100:
        numeric_X  = X.select_dtypes(include=[np.number])
        variances  = numeric_X.var()
        top_cols   = variances.nlargest(100).index
        dropped_n  = X.shape[1] - 100
        X          = X[top_cols]
        st.warning(
            f"Dataset had {X.shape[1] + dropped_n} columns after encoding. "
            f"Keeping top 100 by variance (dropped {dropped_n} low-signal columns). "
            "This prevents memory overflow on standard laptops."
        )

    return X


def get_auto_split_sizes(n_rows, stream_ratio=0.15):
    stream_rows   = max(20, int(n_rows * stream_ratio))
    baseline_rows = n_rows - stream_rows
    return baseline_rows, stream_rows


def calibrate_weights(
    baseline_signal, baseline_df, window_size,
    monitor_mode, feature_to_monitor, mu_0,
    model=None, baseline_X=None,
    target_is_classification=None, is_classifier=True
):
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
            if is_classifier:
                actual = model.predict_proba(Xw).max(axis=1)
            else:
                actual = model.predict(Xw)
            mu_t = float(np.mean(actual))
        else:
            actual = wdf[feature_to_monitor].values
            mu_t   = float(np.mean(actual))

        conf_drifts.append(abs(mu_t - mu_0))
        psi_scores.append(calculate_psi(ref, actual))
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

    return (
        round(sc / total, 4),
        round(sp / total, 4),
        round(sk / total, 4),
        round(ss / total, 4)
    )


def calibrate_threshold(
    baseline_signal, baseline_df, window_size,
    alpha, beta, gamma, lam,
    monitor_mode, feature_to_monitor, mu_0,
    model=None, baseline_X=None, is_classifier=True
):
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
            if is_classifier:
                actual = model.predict_proba(Xw).max(axis=1)
            else:
                actual = model.predict(Xw)
            mu_t = float(np.mean(actual))
        else:
            actual = wdf[feature_to_monitor].values
            mu_t   = float(np.mean(actual))

        cd    = abs(mu_t - mu_0)
        psi   = calculate_psi(ref, actual)
        ks, _ = ks_2samp(ref, actual)
        stab  = abs(mu_t - prev_mu)
        Dt    = alpha * cd + beta * psi + gamma * ks + lam * stab
        baseline_Dt.append(Dt)
        prev_mu = mu_t

    if len(baseline_Dt) < 2:
        return 0.2

    auto = float(np.mean(baseline_Dt) + 2 * np.std(baseline_Dt))

    # Floor: threshold must never collapse below 0.05.
    # On small static datasets (e.g. diabetes) where baseline windows
    # are nearly identical, std approaches zero and the threshold
    # becomes meaninglessly small, causing false drift alarms or
    # making everything appear "stable" relative to a near-zero bar.
    # 0.05 is a safe statistical minimum for any real-world signal.
    auto = max(auto, 0.05)

    return round(auto, 4)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Dataset (CSV or Excel)",
    type=["csv", "xlsx", "xls"]
)


# =========================================================
# MAIN SYSTEM
# =========================================================

if uploaded_file is not None:

    fname = uploaded_file.name

    if fname.endswith((".xlsx", ".xls")):
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
        st.warning(
            f"Large dataset detected: {len(df):,} rows. "
            "Automatically using first 15,000 rows to prevent memory overflow. "
            "This is standard practice for drift detection simulation."
        )
        df = df.head(15_000)

    # =====================================================
    # DATASET PREVIEW
    # =====================================================

    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    c1, c2 = st.columns(2)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])

    cols       = df.columns.tolist()
    target_col = st.selectbox("Select Target Column", cols)
    monitor_mode = st.selectbox(
        "Monitoring Mode",
        ["Model confidence", "Feature value"]
    )

    feature_to_monitor = None
    if monitor_mode == "Feature value":
        numeric_features = [
            c for c in cols
            if c != target_col
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_features:
            st.error("No numeric features available. Switch to Model confidence mode.")
        else:
            feature_to_monitor = st.selectbox(
                "Select Feature to Monitor", numeric_features
            )

    # =====================================================
    # TARGET TYPE
    # =====================================================

    y_full = df[target_col]
    if pd.api.types.is_numeric_dtype(y_full) and y_full.nunique() > 10:
        target_is_classification = False
        st.info("Regression target detected.")
    else:
        target_is_classification = True
        st.info("Classification target detected.")

    # =====================================================
    # AUTO CONFIG
    # =====================================================

    baseline_size, stream_rows = get_auto_split_sizes(len(df))
    window_size = max(5, min(50, stream_rows // 3))

    st.subheader("Auto Configuration")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Baseline Rows",  baseline_size)
    cc2.metric("Streaming Rows", stream_rows)
    cc3.metric("Window Size",    window_size)

    # =====================================================
    # MANUAL OVERRIDE (hidden by default)
    # =====================================================

    with st.expander("Manual override — weights and threshold (optional)"):
        st.caption(
            "Leave all at 0 to use auto-calibration (recommended). "
            "Override only if you have domain-specific knowledge."
        )
        oc1, oc2 = st.columns(2)
        with oc1:
            ov_alpha = st.number_input("Alpha",  0.0, 1.0, 0.0, 0.01)
            ov_beta  = st.number_input("Beta",   0.0, 1.0, 0.0, 0.01)
        with oc2:
            ov_gamma = st.number_input("Gamma",  0.0, 1.0, 0.0, 0.01)
            ov_lam   = st.number_input("Lambda", 0.0, 1.0, 0.0, 0.01)
        ov_thresh = st.number_input("Threshold δ", 0.0, 10.0, 0.0, 0.01)

    # =====================================================
    # RUN
    # =====================================================

    if st.button("Run Drift Detection"):

        if len(df) < 50:
            st.error("Dataset needs at least 50 rows.")
        elif monitor_mode == "Model confidence" and not target_col:
            st.error("Select a target column for Model confidence mode.")
        elif monitor_mode == "Feature value" and feature_to_monitor is None:
            st.error("Select a feature to monitor.")
        else:
            st.success("HDSM Drift Monitoring Started")

            baseline = df.iloc[:baseline_size]
            stream   = df.iloc[baseline_size:]

            # -----------------------------------------------
            # TRAIN MODEL / BUILD BASELINE SIGNAL
            # -----------------------------------------------

            model      = None
            baseline_X = None
            stream_X   = None
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

            # -----------------------------------------------
            # AUTO-CALIBRATE WEIGHTS
            # -----------------------------------------------

            with st.spinner("Auto-calibrating weights from baseline..."):
                auto_a, auto_b, auto_g, auto_l = calibrate_weights(
                    baseline_signal=expected,
                    baseline_df=baseline,
                    window_size=window_size,
                    monitor_mode=monitor_mode,
                    feature_to_monitor=feature_to_monitor,
                    mu_0=mu_0,
                    model=model,
                    baseline_X=baseline_X,
                    is_classifier=is_classifier
                )

            if (ov_alpha + ov_beta + ov_gamma + ov_lam) > 0:
                alpha, beta, gamma, lam = ov_alpha, ov_beta, ov_gamma, ov_lam
                w_src = "Manual Override"
            else:
                alpha, beta, gamma, lam = auto_a, auto_b, auto_g, auto_l
                w_src = "Auto-Calibrated from Baseline"

            # -----------------------------------------------
            # AUTO-CALIBRATE THRESHOLD
            # -----------------------------------------------

            with st.spinner("Auto-calibrating threshold from baseline..."):
                auto_thresh = calibrate_threshold(
                    baseline_signal=expected,
                    baseline_df=baseline,
                    window_size=window_size,
                    alpha=alpha, beta=beta, gamma=gamma, lam=lam,
                    monitor_mode=monitor_mode,
                    feature_to_monitor=feature_to_monitor,
                    mu_0=mu_0,
                    model=model,
                    baseline_X=baseline_X,
                    is_classifier=is_classifier
                )

            threshold = ov_thresh if ov_thresh > 0 else auto_thresh
            t_src     = "Manual Override" if ov_thresh > 0 else "Auto-Calibrated from Baseline"

            # -----------------------------------------------
            # SHOW CALIBRATION SUMMARY
            # -----------------------------------------------

            st.subheader("Calibration Summary")

            st.caption(f"Weights: {w_src}   |   Threshold: {t_src}")

            wc1, wc2, wc3, wc4, wc5 = st.columns(5)
            wc1.metric("α  Confidence", alpha)
            wc2.metric("β  PSI",        beta)
            wc3.metric("γ  KS",         gamma)
            wc4.metric("λ  Stability",  lam)
            wc5.metric("δ  Threshold",  threshold)

            # -----------------------------------------------
            # STREAM LOOP
            # -----------------------------------------------

            drift_scores     = []
            severity_list    = []
            conf_drift_list  = []
            psi_list         = []
            ks_list          = []
            stability_list   = []
            drift_deriv_list = []   # computed but shown only in table/graph

            prev_mu = mu_0
            prev_Dt = None

            for i in range(0, len(stream), window_size):
                window = stream.iloc[i:i + window_size]
                if len(window) < window_size:
                    continue

                if monitor_mode == "Model confidence":
                    Xw = stream_X.iloc[i:i + window_size]
                    if is_classifier:
                        actual = model.predict_proba(Xw).max(axis=1)
                    else:
                        actual = model.predict(Xw)
                    mu_t = float(np.mean(actual))
                else:
                    actual = window[feature_to_monitor].values
                    mu_t   = float(np.mean(actual))

                cd    = abs(mu_t - mu_0)
                psi   = calculate_psi(expected, actual)
                ks, _ = ks_2samp(expected, actual)
                stab  = abs(mu_t - prev_mu)

                D_t     = alpha * cd + beta * psi + gamma * ks + lam * stab
                D_prime = D_t - prev_Dt if prev_Dt is not None else 0.0

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

            # -----------------------------------------------
            # RESULTS
            # -----------------------------------------------

            if len(drift_scores) > 0:

                final_score    = drift_scores[-1]
                final_severity = severity_list[-1]
                drift_status   = (
                    "Drift Detected"
                    if final_score > threshold
                    else "No Drift"
                )

                st.subheader("Final Results")

                r1, r2, r3 = st.columns(3)
                r1.metric("Final Drift Score (Dt)", round(final_score, 4))
                r2.metric("Drift Status",           drift_status)
                r3.metric("Final Severity",         final_severity)

                st.subheader("Recommendations")

                if final_severity == "High Drift":
                    st.error(
                        "High drift detected — well above the dataset-calibrated "
                        "threshold. Model retraining is recommended immediately."
                    )
                elif final_severity == "Moderate Drift":
                    st.warning(
                        "Moderate drift detected — above the calibrated threshold. "
                        "Monitor closely and consider retraining soon."
                    )
                else:
                    st.success(
                        "Dataset is stable relative to its own baseline. "
                        "No action required."
                    )

                windows_idx = list(
                    range(0, len(drift_scores) * window_size, window_size)
                )

                result_df = pd.DataFrame({
                    "Window":            windows_idx,
                    "Confidence Drift":  [round(v, 4) for v in conf_drift_list],
                    "PSI":               [round(v, 4) for v in psi_list],
                    "KS":                [round(v, 4) for v in ks_list],
                    "Stability":         [round(v, 4) for v in stability_list],
                    "Drift Score (Dt)":  [round(v, 4) for v in drift_scores],
                    "Drift Speed (D't)": [round(v, 4) for v in drift_deriv_list],
                    "Severity":          severity_list,
                })

                # -----------------------------------------------
                # GRAPH 1 — MAIN DRIFT SCORE WITH BANDS
                # -----------------------------------------------

                st.subheader("Drift Monitoring Graph")

                fig1, ax1 = plt.subplots(figsize=(12, 5))

                y_top = max(max(drift_scores) * 1.4, threshold * 2.5, 0.5)

                ax1.axhspan(0,               threshold,       alpha=0.12,
                            color="green",  label="Stable zone")
                ax1.axhspan(threshold,       threshold * 1.5, alpha=0.12,
                            color="orange", label="Moderate zone")
                ax1.axhspan(threshold * 1.5, y_top,           alpha=0.12,
                            color="red",    label="High zone")

                ax1.axhline(
                    y=threshold, color="red", linestyle="--",
                    linewidth=1.5,
                    label=f"Auto threshold δ = {threshold}"
                )
                ax1.plot(
                    windows_idx, drift_scores,
                    marker="o", linewidth=2, color="steelblue",
                    label="HDSM Drift Score (Dt)"
                )

                ax1.set_ylim(0, y_top)
                ax1.set_xlabel("Streaming Window")
                ax1.set_ylabel("Drift Score (Dt)")
                ax1.set_title("HDSM Dataset Drift Over Time")
                ax1.legend(loc="upper left")
                ax1.grid(True, alpha=0.3)

                st.pyplot(fig1)

                # -----------------------------------------------
                # GRAPH 2 — COMPONENT BREAKDOWN
                # -----------------------------------------------

                st.subheader("Component Breakdown")

                fig2, ax2 = plt.subplots(figsize=(12, 5))

                ax2.plot(windows_idx, conf_drift_list,
                         marker="o", label=f"Confidence Drift  (α={alpha})")
                ax2.plot(windows_idx, psi_list,
                         marker="s", label=f"PSI  (β={beta})")
                ax2.plot(windows_idx, ks_list,
                         marker="^", label=f"KS Statistic  (γ={gamma})")
                ax2.plot(windows_idx, stability_list,
                         marker="D", label=f"Stability Penalty  (λ={lam})")

                ax2.axhline(y=threshold, color="red", linestyle=":",
                            linewidth=1, label=f"Threshold δ={threshold}")

                ax2.set_xlabel("Streaming Window")
                ax2.set_ylabel("Component Score")
                ax2.set_title("Individual Drift Components Over Time")
                ax2.legend(loc="upper left", fontsize=9)
                ax2.grid(True, alpha=0.3)

                st.pyplot(fig2)

                # -----------------------------------------------
                # GRAPH 3 — WEIGHT PIE
                # -----------------------------------------------

                st.subheader("Auto-Calibrated Weight Distribution")

                fig3, ax3 = plt.subplots(figsize=(5, 5))

                ax3.pie(
                    [alpha, beta, gamma, lam],
                    labels=[
                        f"α Confidence\n{alpha}",
                        f"β PSI\n{beta}",
                        f"γ KS\n{gamma}",
                        f"λ Stability\n{lam}",
                    ],
                    colors=["steelblue", "darkorange", "green", "purple"],
                    autopct="%1.1f%%",
                    startangle=140
                )
                ax3.set_title("Weight share per drift component")

                st.pyplot(fig3)

                # -----------------------------------------------
                # RESULTS TABLE
                # -----------------------------------------------

                st.subheader("Detailed Results")
                st.dataframe(result_df)

                # -----------------------------------------------
                # SAVE
                # -----------------------------------------------

                if save_results:
                    result_df.to_csv("hdsm_results.csv", index=False)
                    with open("hdsm_results.csv", "rb") as f:
                        st.download_button(
                            label="Download Results CSV",
                            data=f,
                            file_name="hdsm_results.csv",
                            mime="text/csv"
                        )

            else:
                st.warning(
                    "No full windows found in streaming data. "
                    "Try uploading a larger dataset."
                )