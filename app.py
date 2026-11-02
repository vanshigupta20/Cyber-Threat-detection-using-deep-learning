"""
Cyber Threat Detection - Streamlit SOC dashboard.

Run with:  streamlit run app.py

Reads only saved artefacts (model, scaler, feature list, JSON metrics).
It never loads the raw CIC-IDS2017 dataset and never trains a model.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.data_utils import clean_column_names, find_label_column, normalize_attack_labels
from src.predict import (
    InvalidInputError,
    ModelArtifactsMissingError,
    NIDSPredictor,
    load_csv_safely,
    preprocessing_matches_model,
    summarize,
)

st.set_page_config(page_title="Cyber Threat Detection | NIDS", page_icon="🛡️", layout="wide")

DISCLAIMER = ("This is a machine-learning-based academic prototype trained and evaluated "
              "on the CIC-IDS2017 dataset.")
COLORS = {"normal": "#22c55e", "attack": "#ef4444", "accent": "#00e5ff",
          "warn": "#f59e0b", "muted": "#94a3b8", "purple": "#a78bfa"}
PAGES = ["Dashboard", "Dataset Analysis", "Intrusion Detection", "Model Performance", "About Project"]
MAX_TABLE_ROWS = 5000

# ====================================================================== styling
st.markdown("""
<style>
.block-container {padding-top: 3rem; padding-bottom: 2rem;}
.hero {background: linear-gradient(120deg, #07101f 0%, #0c1f3a 55%, #062a33 100%);
       border: 1px solid #16324f; border-radius: 14px; padding: 22px 28px; margin-bottom: 18px;
       box-shadow: 0 0 24px rgba(0,229,255,0.08);}
.hero h1 {font-family: 'Consolas','JetBrains Mono',monospace; letter-spacing: 4px; margin: 0;
          color: #00e5ff; font-size: 2.0rem;}
.hero h3 {margin: 4px 0 0 0; color: #e2e8f0; font-weight: 500;}
.hero p {margin: 6px 0 0 0; color: #94a3b8; font-family: 'Consolas',monospace; font-size: 0.85rem;
         letter-spacing: 2px;}
.kpi {background: #0f172a; border: 1px solid #1e293b; border-left: 4px solid var(--c);
      border-radius: 10px; padding: 14px 16px; height: 100%;}
.kpi .label {color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px;}
.kpi .value {color: #f8fafc; font-size: 1.7rem; font-weight: 700; font-family: 'Consolas',monospace;}
.kpi .sub {color: #64748b; font-size: 0.75rem;}
.status {border-radius: 12px; padding: 18px; text-align: center; border: 1px solid var(--c);
         background: radial-gradient(circle at center, rgba(255,255,255,0.03), #0b1220);
         box-shadow: 0 0 22px var(--g);}
.status .title {color: #94a3b8; letter-spacing: 3px; font-size: 0.8rem;}
.status .state {color: var(--c); font-size: 2rem; font-weight: 800; letter-spacing: 3px;
                font-family: 'Consolas',monospace;}
.status .detail {color: #cbd5e1; font-size: 0.85rem;}
.section {color: #00e5ff; font-family: 'Consolas',monospace; letter-spacing: 2px;
          font-size: 0.9rem; border-bottom: 1px solid #1e293b; padding-bottom: 4px; margin: 18px 0 10px 0;}
.explain {background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 12px 14px;
          margin-bottom: 8px;}
.explain b {color: var(--c);}
.verdict {border-radius: 12px; padding: 20px 24px; border: 2px solid var(--c); background: #0b1220;}
.verdict .big {font-size: 1.9rem; font-weight: 800; color: var(--c); font-family: 'Consolas',monospace;}
.verdict .line {font-size: 1.1rem; color: #e2e8f0; font-family: 'Consolas',monospace; margin-top: 6px;}
</style>
""", unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", color: str = COLORS["accent"]) -> str:
    return (f'<div class="kpi" style="--c:{color}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


def kpi_row(cards: list) -> None:
    for column, card in zip(st.columns(len(cards)), cards):
        column.markdown(kpi(*card), unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="section">▍{title.upper()}</div>', unsafe_allow_html=True)


def style_figure(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(color="#cbd5e1"), legend=dict(orientation="h", y=-0.1))
    return fig


def fmt(value) -> str:
    return f"{value:,}" if isinstance(value, (int, np.integer)) else str(value)


def pct(value) -> str:
    return f"{value * 100:.2f}%" if value is not None else "—"


# ====================================================================== data access
@st.cache_data(show_spinner=False)
def _read_json(path_str: str, modified_time: float):
    with open(path_str, encoding="utf-8") as f:
        return json.load(f)


def load_json(path: Path):
    """Load a JSON output file; cache refreshes automatically when the file changes."""
    if not path.exists():
        return None
    try:
        return _read_json(str(path), path.stat().st_mtime)
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_resource(show_spinner="Loading trained model ...")
def _load_predictor(model_mtime: float):
    return NIDSPredictor()


def get_predictor():
    """Return (predictor, error_message). Never raises."""
    try:
        mtime = config.MODEL_PATH.stat().st_mtime if config.MODEL_PATH.exists() else 0.0
        return _load_predictor(mtime), None
    except ModelArtifactsMissingError as error:
        return None, str(error)
    except Exception as error:  # corrupted / incompatible model file
        return None, f"The model could not be loaded: {error}"


def run_detection(df: pd.DataFrame, source_name: str) -> bool:
    """Predict on a DataFrame and store the results in session state for all pages."""
    predictor, error = get_predictor()
    if error:
        st.error(error)
        return False
    try:
        with st.spinner(f"Analysing {len(df):,} network flows ..."):
            results, report = predictor.predict_dataframe(df)
    except InvalidInputError as error:
        st.error(str(error))
        return False
    st.session_state["analysis"] = {"results": results, "report": report, "source": source_name,
                                    "summary": summarize(results)}
    return True


def missing_artifacts_notice(what: str, command: str) -> None:
    st.warning(f"{what} not found. Run:\n\n```bash\n{command}\n```")


# ====================================================================== charts
def donut_chart(normal: int, attack: int, title: str) -> go.Figure:
    fig = go.Figure(go.Pie(labels=["Normal", "Attack"], values=[normal, attack], hole=0.62,
                           marker=dict(colors=[COLORS["normal"], COLORS["attack"]]),
                           textinfo="percent", sort=False))
    total = normal + attack
    fig.add_annotation(text=f"<b>{total:,}</b><br>flows", showarrow=False, font=dict(size=15))
    fig.update_layout(title=title)
    return style_figure(fig)


def attack_distribution_chart(label_counts: dict) -> go.Figure | None:
    attacks = {k: v for k, v in label_counts.items() if k.upper() != config.BENIGN_LABEL}
    if not attacks:
        return None
    attacks = dict(sorted(attacks.items(), key=lambda item: item[1]))
    fig = go.Figure(go.Bar(x=list(attacks.values()), y=list(attacks.keys()), orientation="h",
                           marker_color=COLORS["warn"], text=[f"{v:,}" for v in attacks.values()],
                           textposition="outside", cliponaxis=False))
    fig.update_layout(title="Attack Type Distribution (log scale)", xaxis_type="log")
    fig = style_figure(fig, height=max(320, 28 * len(attacks) + 80))
    fig.update_layout(margin=dict(r=60))
    return fig


def metrics_bar_chart(metrics: dict) -> go.Figure:
    names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"]]
    fig = go.Figure(go.Bar(x=names, y=values, text=[f"{v:.4f}" for v in values], textposition="outside",
                           marker_color=[COLORS["accent"], COLORS["purple"], COLORS["warn"], COLORS["normal"]]))
    fig.update_layout(title="Model Performance (unseen test set)", yaxis_range=[0, 1.12])
    return style_figure(fig)


def confusion_matrix_chart(tn: int, fp: int, fn: int, tp: int, title: str) -> go.Figure:
    z = [[tn, fp], [fn, tp]]
    text = [[f"TN<br>{tn:,}", f"FP<br>{fp:,}"], [f"FN<br>{fn:,}", f"TP<br>{tp:,}"]]
    fig = go.Figure(go.Heatmap(z=z, x=["Normal", "Attack"], y=["Normal", "Attack"], text=text,
                               texttemplate="%{text}", textfont=dict(size=18), colorscale="Blues",
                               showscale=False))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual",
                      yaxis_autorange="reversed")
    return style_figure(fig, height=380)


def training_curve_chart(history: dict, metric: str, title: str) -> go.Figure:
    epochs = list(range(1, len(history[metric]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history[metric], mode="lines+markers", name=f"Training {metric}",
                             line=dict(color=COLORS["accent"])))
    fig.add_trace(go.Scatter(x=epochs, y=history[f"val_{metric}"], mode="lines+markers",
                             name=f"Validation {metric}", line=dict(color=COLORS["warn"])))
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title=metric.capitalize())
    return style_figure(fig)


def gauge_chart(probability: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100, number={"suffix": "%"},
        title={"text": "Attack Probability"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#e2e8f0"},
               "steps": [{"range": [0, 30], "color": "#14532d"}, {"range": [30, 70], "color": "#78350f"},
                         {"range": [70, 100], "color": "#7f1d1d"}],
               "threshold": {"line": {"color": COLORS["accent"], "width": 4},
                             "value": config.PREDICTION_THRESHOLD * 100}}))
    return style_figure(fig, height=280)


# ====================================================================== pages
def render_header() -> None:
    st.markdown("""
    <div class="hero">
      <h1>CYBER THREAT DETECTION</h1>
      <h3>Network Intrusion Detection System</h3>
      <p>POWERED BY DEEP LEARNING · CIC-IDS2017 · ANN (TENSORFLOW / KERAS)</p>
    </div>""", unsafe_allow_html=True)


def render_status_panel() -> None:
    analysis = st.session_state.get("analysis")
    if analysis is None:
        color, glow, state = COLORS["muted"], "rgba(148,163,184,0.15)", "STANDBY"
        detail = "No traffic analysed yet. Upload a CSV on the Intrusion Detection page or run the demo scan."
    else:
        s = analysis["summary"]
        if s["attack"] > 0:
            color, glow, state = COLORS["attack"], "rgba(239,68,68,0.35)", "THREAT DETECTED"
        else:
            color, glow, state = COLORS["normal"], "rgba(34,197,94,0.3)", "NORMAL"
        detail = (f"Source: {analysis['source']} · {s['analysed']:,} flows analysed · "
                  f"{s['attack']:,} flagged as attack ({s['attack_percentage']:.2f}%)")
    st.markdown(f'<div class="status" style="--c:{color};--g:{glow}"><div class="title">SYSTEM STATUS</div>'
                f'<div class="state">{state}</div><div class="detail">{detail}</div></div>',
                unsafe_allow_html=True)


def page_dashboard() -> None:
    render_header()
    summary = load_json(config.DATASET_SUMMARY_PATH)
    evaluation = load_json(config.EVALUATION_METRICS_PATH)

    if summary is None:
        missing_artifacts_notice("Dataset summary", "python src/preprocess.py")
    if evaluation is None:
        missing_artifacts_notice("Evaluation results", "python src/train_model.py\npython src/evaluate_model.py")
    if summary and summary.get("mode") == "development":
        st.warning(f"The model was built in DEVELOPMENT MODE on a {summary['sample_fraction']:.0%} sample. "
                   "Re-run `python src/preprocess.py --mode final` for final results.")

    cm = evaluation["confusion_matrix"] if evaluation else None
    kpi_row([
        ("Total Network Records", fmt(summary["final_records"]) if summary else "—",
         "flows after cleaning & de-duplication", COLORS["accent"]),
        ("Normal Traffic", fmt(summary["class_counts"]["Normal"]) if summary else "—",
         "BENIGN flows in dataset", COLORS["normal"]),
        ("Detected Attacks", fmt(cm["tp"]) if cm else "—",
         f"of {evaluation['test_attacks']:,} attacks in unseen test set" if evaluation else "run evaluation",
         COLORS["attack"]),
        ("Model Accuracy", pct(evaluation["accuracy"]) if evaluation else "—",
         f"on {evaluation['test_samples']:,} unseen test flows" if evaluation else "run evaluation",
         COLORS["purple"]),
    ])

    st.write("")
    left, right = st.columns([2, 1])
    with left:
        render_status_panel()
    with right:
        st.caption("Quick demo: scan unseen test flows saved during preprocessing.")
        if st.button("▶ RUN DEMO SCAN", use_container_width=True,
                     disabled=not config.SAMPLE_TEST_CSV_PATH.exists()):
            try:
                if run_detection(load_csv_safely(config.SAMPLE_TEST_CSV_PATH), "sample_test_flows.csv"):
                    st.rerun()
            except InvalidInputError as error:
                st.error(str(error))
        if st.session_state.get("analysis") and st.button("Clear analysis", use_container_width=True):
            st.session_state.pop("analysis")
            st.rerun()

    section("Traffic overview")
    c1, c2, c3 = st.columns(3)
    if summary:
        c1.plotly_chart(donut_chart(summary["class_counts"]["Normal"], summary["class_counts"]["Attack"],
                                    "Dataset: Normal vs Attack"), use_container_width=True)
        attack_fig = attack_distribution_chart(summary["label_counts"])
        if attack_fig:
            c2.plotly_chart(attack_fig.update_layout(height=340), use_container_width=True)
    if evaluation:
        c3.plotly_chart(metrics_bar_chart(evaluation), use_container_width=True)

    analysis = st.session_state.get("analysis")
    if analysis:
        section(f"Latest analysis · {analysis['source']}")
        s = analysis["summary"]
        a1, a2 = st.columns([1, 2])
        a1.plotly_chart(donut_chart(s["normal"], s["attack"], "Analysed traffic"), use_container_width=True)
        attacks = analysis["results"].query("Prediction == 'Attack'")
        a2.markdown("**Highest-probability alerts**")
        a2.dataframe(attacks.sort_values("Probability", ascending=False).head(10),
                     hide_index=True, use_container_width=True)
    st.caption(DISCLAIMER)


def page_dataset_analysis() -> None:
    st.title("📊 Dataset Analysis")
    tab_training, tab_upload = st.tabs(["CIC-IDS2017 training dataset", "Inspect your own CSV"])

    with tab_training:
        summary = load_json(config.DATASET_SUMMARY_PATH)
        if summary is None:
            missing_artifacts_notice("Dataset summary (outputs/metrics/dataset_summary.json)",
                                     "python src/preprocess.py")
        else:
            mode_label = ("DEVELOPMENT MODE (%.0f%% sample)" % (summary["sample_fraction"] * 100)
                          if summary["mode"] == "development" else "FINAL TRAINING MODE (full data)")
            st.caption(f"Processed on {summary['created_at']} · {mode_label}")
            kpi_row([
                ("Records", fmt(summary["final_records"]), "after cleaning", COLORS["accent"]),
                ("Features used", fmt(summary["n_features_used"]), "model inputs", COLORS["purple"]),
                ("Normal records", fmt(summary["class_counts"]["Normal"]), "label 0", COLORS["normal"]),
                ("Attack records", fmt(summary["class_counts"]["Attack"]), "label 1", COLORS["attack"]),
            ])

            section("Class distribution")
            c1, c2 = st.columns([1, 1.4])
            c1.plotly_chart(donut_chart(summary["class_counts"]["Normal"], summary["class_counts"]["Attack"],
                                        "Normal vs Attack"), use_container_width=True)
            attack_fig = attack_distribution_chart(summary["label_counts"])
            if attack_fig:
                c2.plotly_chart(attack_fig, use_container_width=True)
            labels = pd.DataFrame(summary["label_counts"].items(), columns=["Original label", "Records"])
            labels["Binary class"] = np.where(labels["Original label"].str.upper() == config.BENIGN_LABEL,
                                              "0 - Normal", "1 - Attack")
            labels["Share"] = (labels["Records"] / labels["Records"].sum() * 100).round(3).astype(str) + " %"
            st.dataframe(labels, hide_index=True, use_container_width=True)

            section("Data cleaning report")
            cleaning = pd.DataFrame([
                ("Raw rows read from CSV files", summary["raw_records"]),
                ("Empty rows removed", summary["empty_rows_removed"]),
                ("Rows with NaN / Infinity removed", summary["invalid_rows_removed"]),
                ("Duplicate records removed", summary["duplicates_removed"]),
                ("Rows left out by development sampling", summary["rows_sampled_out"]),
                ("Final records used", summary["final_records"]),
            ], columns=["Step", "Rows"])
            c1, c2 = st.columns(2)
            c1.dataframe(cleaning, hide_index=True, use_container_width=True)
            splits = pd.DataFrame([(name.capitalize(), size, summary["split_attack_counts"][name])
                                   for name, size in summary["split_sizes"].items()],
                                  columns=["Split", "Records", "Attacks"])
            c2.dataframe(splits, hide_index=True, use_container_width=True)
            c2.caption("Stratified split 70 / 10 / 20. The scaler was fitted on the training split only.")

            section("Missing / infinite values (before cleaning)")
            missing = summary.get("missing_values_per_column", {})
            if missing:
                st.dataframe(pd.DataFrame([(c, v["nan"], v["inf"]) for c, v in missing.items()],
                                          columns=["Column", "NaN values", "Infinite values"]),
                             hide_index=True, use_container_width=True)
            else:
                st.success("No missing or infinite values were found.")

            section("Feature preparation")
            f1, f2, f3 = st.columns(3)
            f1.markdown("**Identifier columns removed**")
            f1.write(summary["dropped_identifier_columns"] or "None")
            f2.markdown("**Constant columns removed (train split)**")
            f2.write(summary["dropped_constant_columns"] or "None")
            f3.markdown("**Non-numeric columns removed**")
            f3.write(summary["dropped_non_numeric_columns"] or "None")
            with st.expander(f"All {summary['n_features_used']} model features (in model order)"):
                st.write(summary["feature_columns"])
            with st.expander("Per-file statistics"):
                st.dataframe(pd.DataFrame(summary["source_files"]), hide_index=True, use_container_width=True)

    with tab_upload:
        st.info("Uploading a file here only **inspects** it. The model is never retrained from the app.")
        uploaded = st.file_uploader("Upload a network-flow CSV", type=["csv"], key="inspect_upload")
        if uploaded is not None:
            try:
                df = clean_column_names(load_csv_safely(uploaded))
            except InvalidInputError as error:
                st.error(str(error))
                return
            numeric = df.select_dtypes(include=[np.number])
            label_col = find_label_column(df.columns)
            kpi_row([
                ("Records", fmt(len(df)), uploaded.name, COLORS["accent"]),
                ("Columns", fmt(df.shape[1]), f"{numeric.shape[1]} numeric", COLORS["purple"]),
                ("Duplicate rows", fmt(int(df.duplicated().sum())), "exact duplicates", COLORS["warn"]),
                ("Cells with NaN", fmt(int(df.isna().sum().sum())), "missing values", COLORS["attack"]),
            ])
            if label_col:
                labels = normalize_attack_labels(df[label_col].dropna())
                counts = labels.value_counts()
                normal = int(counts.get(config.BENIGN_LABEL, 0))
                c1, c2 = st.columns([1, 1.4])
                c1.plotly_chart(donut_chart(normal, int(counts.sum()) - normal, "Label distribution"),
                                use_container_width=True)
                fig = attack_distribution_chart(counts.to_dict())
                if fig:
                    c2.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No 'Label' column found - class distribution cannot be shown.")

            section("Missing and infinite values per column")
            quality = pd.DataFrame({
                "NaN values": df.isna().sum(),
                "Infinite values": np.isinf(numeric).sum().reindex(df.columns, fill_value=0),
            })
            quality = quality[(quality["NaN values"] > 0) | (quality["Infinite values"] > 0)]
            if quality.empty:
                st.success("No missing or infinite values.")
            else:
                st.dataframe(quality, use_container_width=True)

            predictor, _ = get_predictor()
            if predictor is not None:
                missing, extra = predictor.check_columns(df)
                if missing:
                    st.error(f"Not compatible with the model: {len(missing)} required feature(s) missing, "
                             f"e.g. {missing[:5]}")
                else:
                    st.success(f"Compatible with the model ({len(predictor.feature_columns)} features found; "
                               f"{len(extra)} extra column(s) will be ignored).")
            with st.expander("Preview (first 100 rows)"):
                st.dataframe(df.head(100), use_container_width=True)


def show_detection_results() -> None:
    analysis = st.session_state.get("analysis")
    if not analysis:
        return
    results, report, s = analysis["results"], analysis["report"], analysis["summary"]

    section(f"Detection results · {analysis['source']}")
    kpi_row([
        ("Normal Records", fmt(s["normal"]), "predicted SAFE", COLORS["normal"]),
        ("Attack Records", fmt(s["attack"]), "predicted INTRUSION", COLORS["attack"]),
        ("Attack Percentage", f"{s['attack_percentage']:.2f}%", f"of {s['analysed']:,} analysed flows",
         COLORS["warn"]),
        ("Skipped Rows", fmt(s["skipped"]), "NaN / Infinity / text values", COLORS["muted"]),
    ])
    if report.extra_columns:
        st.caption(f"ℹ️ {len(report.extra_columns)} extra column(s) were ignored "
                   f"(e.g. {', '.join(map(str, report.extra_columns[:4]))}).")
    if report.columns_reordered:
        st.caption("ℹ️ Columns were re-ordered to match the order used during training.")
    if s["skipped"]:
        st.warning(f"{s['skipped']:,} row(s) contained missing, infinite or non-numeric values and were not "
                   f"analysed (records: {report.skipped_records[:10]}{' ...' if s['skipped'] > 10 else ''}).")

    c1, c2 = st.columns(2)
    c1.plotly_chart(donut_chart(s["normal"], s["attack"], "Prediction distribution"), use_container_width=True)
    analysed = results[results["Prediction"] != "Skipped"]
    hist = go.Figure(go.Histogram(x=analysed["Probability"], nbinsx=40, marker_color=COLORS["accent"]))
    hist.add_vline(x=config.PREDICTION_THRESHOLD, line_dash="dash", line_color=COLORS["attack"],
                   annotation_text=f"threshold {config.PREDICTION_THRESHOLD}")
    hist.update_layout(title="Attack probability distribution", xaxis_title="Attack probability",
                       yaxis_title="Flows", yaxis_type="log")
    c2.plotly_chart(style_figure(hist), use_container_width=True)

    if report.has_ground_truth:
        checked = analysed[analysed["Actual Label"].notna() & (analysed["Actual Label"] != "nan")]
        actual = (checked["Actual Class"] == "Attack").to_numpy()
        predicted = (checked["Prediction"] == "Attack").to_numpy()
        tp, tn = int((actual & predicted).sum()), int((~actual & ~predicted).sum())
        fp, fn = int((~actual & predicted).sum()), int((actual & ~predicted).sum())
        with st.expander("✅ This file contains a 'Label' column - compare predictions with the true labels",
                         expanded=True):
            accuracy = (tp + tn) / max(len(checked), 1)
            recall = tp / (tp + fn) if (tp + fn) else float("nan")
            precision = tp / (tp + fp) if (tp + fp) else float("nan")
            m1, m2 = st.columns([1, 1])
            m1.plotly_chart(confusion_matrix_chart(tn, fp, fn, tp, "Confusion matrix (this file)"),
                            use_container_width=True)
            m2.metric("Accuracy on this file", f"{accuracy:.2%}")
            m2.metric("Recall (attacks caught)", f"{recall:.2%}" if not np.isnan(recall) else "—")
            m2.metric("Precision", f"{precision:.2%}" if not np.isnan(precision) else "—")
            m2.caption("The label column was NOT used for prediction - only for this comparison.")

    section("Per-record predictions")
    view = st.radio("Show", ["All records", "Attacks only", "Normal only", "Skipped rows"], horizontal=True)
    filtered = {"All records": results,
                "Attacks only": results[results["Prediction"] == "Attack"],
                "Normal only": results[results["Prediction"] == "Normal"],
                "Skipped rows": results[results["Prediction"] == "Skipped"]}[view]
    if len(filtered) > MAX_TABLE_ROWS:
        st.caption(f"Showing the first {MAX_TABLE_ROWS:,} of {len(filtered):,} rows. "
                   "The download contains every row.")
    st.dataframe(
        filtered.head(MAX_TABLE_ROWS), hide_index=True, use_container_width=True,
        column_config={"Probability": st.column_config.ProgressColumn(
            "Probability", help="Attack probability from the ANN", min_value=0.0, max_value=1.0,
            format="%.4f")})
    st.download_button("⬇ Download prediction results (CSV)", results.to_csv(index=False).encode("utf-8"),
                       file_name=f"{Path(analysis['source']).stem}_predictions.csv", mime="text/csv",
                       use_container_width=True)


def render_single_flow_form(predictor: NIDSPredictor) -> None:
    features = predictor.feature_columns
    st.info(f"The model needs **all {len(features)} network-flow features** below, in the same units as "
            "CIC-IDS2017 (CICFlowMeter output). Typing them by hand is impractical, so **CSV upload is the "
            "primary mode**. Use the loader to fill the form with a real flow from the unseen test set, "
            "then edit any value and analyse it.")

    sample = None
    if config.SAMPLE_TEST_CSV_PATH.exists():
        sample = clean_column_names(pd.read_csv(config.SAMPLE_TEST_CSV_PATH, encoding="latin-1"))
    c1, c2, c3 = st.columns([2, 1, 1])
    choice = c1.selectbox("Pre-fill the form with", ["All zeros", "Random BENIGN flow from test set",
                                                     "Random ATTACK flow from test set"],
                          disabled=sample is None)
    if c2.button("Load values", use_container_width=True):
        if choice == "All zeros" or sample is None:
            row, label = {f: 0.0 for f in features}, None
        else:
            is_benign = sample["Label"].str.upper() == config.BENIGN_LABEL
            pool = sample[is_benign] if "BENIGN" in choice else sample[~is_benign]
            picked = pool.sample(1).iloc[0]
            row, label = {f: float(picked[f]) for f in features}, picked["Label"]
        for i, f in enumerate(features):
            st.session_state[f"feature_{i}"] = row[f]
        st.session_state["loaded_label"] = label
        st.rerun()
    if st.session_state.get("loaded_label"):
        c3.markdown(f"Loaded flow's true label:<br>**{st.session_state['loaded_label']}**",
                    unsafe_allow_html=True)

    with st.form("single_flow_form"):
        columns = st.columns(4)
        values = {}
        for i, feature in enumerate(features):
            key = f"feature_{i}"
            if key not in st.session_state:
                st.session_state[key] = 0.0
            values[feature] = columns[i % 4].number_input(feature, key=key, format="%.4f", step=1.0)
        submitted = st.form_submit_button("🔍 ANALYZE TRAFFIC", use_container_width=True, type="primary")

    if submitted:
        try:
            result = predictor.predict_single(values)
        except InvalidInputError as error:
            st.error(str(error))
            return
        color = COLORS["attack"] if result["prediction"] == "ATTACK" else COLORS["normal"]
        level_color = {"LOW": COLORS["normal"], "MEDIUM": COLORS["warn"], "HIGH": COLORS["attack"]}
        v1, v2 = st.columns([1.2, 1])
        v1.markdown(
            f'<div class="verdict" style="--c:{color}"><div class="big">Prediction: {result["prediction"]}</div>'
            f'<div class="line">Attack Probability: {result["probability"] * 100:.1f}%</div>'
            f'<div class="line">Threat Level: <span style="color:{level_color[result["threat_level"]]}">'
            f'{result["threat_level"]}</span></div></div>', unsafe_allow_html=True)
        v1.caption(f"Decision rule: Attack if probability ≥ {config.PREDICTION_THRESHOLD}. "
                   "Threat level is a project visualisation rule, not an official severity standard.")
        v2.plotly_chart(gauge_chart(result["probability"]), use_container_width=True)


def page_intrusion_detection() -> None:
    st.title("🛡️ Intrusion Detection")
    predictor, error = get_predictor()
    if error:
        st.error(error)
        st.code("python src/preprocess.py\npython src/train_model.py\npython src/evaluate_model.py", "bash")
        return
    st.caption(f"Model input: {len(predictor.feature_columns)} features · decision threshold "
               f"{predictor.threshold} · Threat level (project visualisation rule, not an official "
               f"standard): LOW < {config.THREAT_LEVEL_LOW_MAX:.2f} ≤ MEDIUM < "
               f"{config.THREAT_LEVEL_MEDIUM_MAX:.2f} ≤ HIGH")

    tab_csv, tab_single = st.tabs(["Mode A · Upload CSV (primary)", "Mode B · Single network flow"])
    with tab_csv:
        st.markdown("Upload a **CIC-IDS2017-compatible CSV** (CICFlowMeter features). Extra columns are "
                    "ignored, column order does not matter, and a `Label` column (if present) is only used "
                    "to compare predictions with the truth.")
        uploaded = st.file_uploader("Network traffic CSV", type=["csv"], key="detect_upload")
        b1, b2 = st.columns(2)
        if b1.button("🔍 Analyse uploaded file", use_container_width=True, disabled=uploaded is None,
                     type="primary"):
            try:
                run_detection(load_csv_safely(uploaded), uploaded.name)
            except InvalidInputError as error:
                st.error(str(error))
        if b2.button("🧪 Use sample test flows", use_container_width=True,
                     disabled=not config.SAMPLE_TEST_CSV_PATH.exists()):
            try:
                run_detection(load_csv_safely(config.SAMPLE_TEST_CSV_PATH), "sample_test_flows.csv")
            except InvalidInputError as error:
                st.error(str(error))
        show_detection_results()

    with tab_single:
        render_single_flow_form(predictor)


def page_model_performance() -> None:
    st.title("📈 Model Performance")
    evaluation = load_json(config.EVALUATION_METRICS_PATH)
    training = load_json(config.TRAINING_HISTORY_PATH)
    metadata = load_json(config.MODEL_METADATA_PATH)

    if evaluation is None:
        missing_artifacts_notice("Evaluation results", "python src/evaluate_model.py")
    else:
        st.caption(f"Evaluated on {evaluation['test_samples']:,} UNSEEN test flows "
                   f"({evaluation['test_attacks']:,} attacks) · threshold {evaluation['prediction_threshold']} "
                   f"· {evaluation['evaluated_at']}")
        kpi_row([
            ("Accuracy", pct(evaluation["accuracy"]), "all correct predictions", COLORS["accent"]),
            ("Precision", pct(evaluation["precision"]), "alerts that were real attacks", COLORS["purple"]),
            ("Recall", pct(evaluation["recall"]), "attacks that were caught", COLORS["warn"]),
            ("F1 Score", pct(evaluation["f1_score"]), "balance of precision & recall", COLORS["normal"]),
        ])

        cm = evaluation["confusion_matrix"]
        section("Confusion matrix")
        c1, c2 = st.columns([1.1, 1])
        c1.plotly_chart(confusion_matrix_chart(cm["tn"], cm["fp"], cm["fn"], cm["tp"],
                                               "Confusion matrix (unseen test set)"), use_container_width=True)
        with c2:
            for name, key, color, text in (
                ("True Positive (TP)", "tp", COLORS["normal"], "Attack correctly detected."),
                ("True Negative (TN)", "tn", COLORS["normal"], "Normal traffic correctly identified."),
                ("False Positive (FP)", "fp", COLORS["warn"],
                 "Normal traffic incorrectly classified as attack (a false alarm)."),
                ("False Negative (FN)", "fn", COLORS["attack"],
                 "Attack incorrectly classified as normal (a missed attack)."),
            ):
                st.markdown(f'<div class="explain" style="--c:{color}"><b>{name}: {cm[key]:,}</b><br>{text}</div>',
                            unsafe_allow_html=True)

        st.error(
            f"**Why False Negatives matter most in an IDS** — a false negative is an attack the system "
            f"let through as normal traffic. Nobody is alerted, so the attacker can continue unnoticed. "
            f"A false positive only costs an analyst some time to check a false alarm. "
            f"This model missed **{cm['fn']:,}** of **{cm['fn'] + cm['tp']:,}** test attacks "
            f"(false negative rate {evaluation['false_negative_rate']:.2%}) and raised "
            f"**{cm['fp']:,}** false alarms (false positive rate {evaluation['false_positive_rate']:.2%}). "
            f"Class weights were used during training to reduce missed attacks.")

        section("What the metrics mean")
        e1, e2 = st.columns(2)
        e1.markdown("""
- **Accuracy** = (TP + TN) / all flows — how often the model is right overall. On imbalanced data this
  can look high even when many attacks are missed, so it is not used alone.
- **Precision** = TP / (TP + FP) — when the model raises an alert, how often it is a real attack.""")
        e2.markdown("""
- **Recall (Detection rate)** = TP / (TP + FN) — what share of real attacks were caught.
  This is the most important metric for an IDS.
- **F1 Score** — the harmonic mean of precision and recall; high only when both are high.""")

        c1, c2 = st.columns(2)
        c1.plotly_chart(metrics_bar_chart(evaluation), use_container_width=True)
        per_attack = evaluation.get("detection_rate_by_attack", {})
        if per_attack:
            names = list(per_attack.keys())[::-1]
            rates = [per_attack[n]["detection_rate"] for n in names]
            fig = go.Figure(go.Bar(x=rates, y=names, orientation="h", marker_color=COLORS["attack"],
                                   text=[f"{r:.1%} (n={per_attack[n]['total']:,})" for r, n in zip(rates, names)],
                                   textposition="auto"))
            fig.update_layout(title="Detection rate per attack type (test set)", xaxis_range=[0, 1.05])
            c2.plotly_chart(style_figure(fig, height=max(340, 28 * len(names) + 80)), use_container_width=True)

        section("Classification report")
        report = evaluation["classification_report"]
        rows = [(name, report[name]["precision"], report[name]["recall"], report[name]["f1-score"],
                 int(report[name]["support"])) for name in config.CLASS_NAMES + ["macro avg", "weighted avg"]]
        st.dataframe(pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "F1-score", "Support"])
                     .style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-score": "{:.4f}",
                                    "Support": "{:,}"}), hide_index=True, use_container_width=True)

    section("Training history")
    if training is None:
        missing_artifacts_notice("Training history", "python src/train_model.py")
    else:
        history = training["history"]
        c1, c2 = st.columns(2)
        c1.plotly_chart(training_curve_chart(history, "accuracy", "Training Accuracy vs Validation Accuracy"),
                        use_container_width=True)
        c2.plotly_chart(training_curve_chart(history, "loss", "Training Loss vs Validation Loss"),
                        use_container_width=True)
        st.caption(f"Epochs run: {training['epochs_run']} · best epoch (lowest validation loss): "
                   f"{training['best_epoch']} · EarlyStopping restored the best weights.")
    if metadata:
        with st.expander("Model configuration"):
            st.json(metadata)


def page_about() -> None:
    st.title("ℹ️ About Project")
    st.warning(DISCLAIMER + " It is not a real-time enterprise IDS and cannot detect every cyber attack.")
    st.markdown("""
### Problem
Cyber attacks can be hidden inside large amounts of network traffic. Manually inspecting millions of
network flows is impossible, and signature-based tools can miss attack patterns they have no rule for.

### Proposed Solution
A deep-learning-based Network Intrusion Detection System analyses network-flow characteristics
(packet counts, sizes, timings, TCP flags, ...) and predicts whether traffic is **normal** or **malicious**.

### Dataset
**CIC-IDS2017** from the Canadian Institute for Cybersecurity (University of New Brunswick) —
five days of benign traffic plus attacks such as brute force (FTP/SSH), DoS/DDoS, Heartbleed,
web attacks, infiltration, botnet and port scans. Flows were extracted with CICFlowMeter.
The *MachineLearningCSV* version is used.

### Model
Artificial Neural Network built with TensorFlow/Keras:
`Input → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)`,
Adam optimiser, binary cross-entropy loss, class weights for imbalance, EarlyStopping.

### Output
**Normal (0)** or **Attack (1)**, with the attack probability. A flow is classified as Attack when the
probability is at least the decision threshold.
""")
    st.markdown("### Pipeline")
    st.graphviz_chart("""
    digraph {
      rankdir=LR; bgcolor="transparent";
      node [shape=box style="rounded,filled" fillcolor="#0f172a" color="#00e5ff" fontcolor="#e2e8f0" fontname="Helvetica"];
      edge [color="#64748b"];
      A [label="CIC-IDS2017\\nCSV files"]; B [label="Cleaning\\n(NaN, Inf, duplicates)"];
      C [label="Binary labels\\nBENIGN=0, Attack=1"]; D [label="Train / Val / Test\\nsplit"];
      E [label="StandardScaler\\n(fit on train only)"]; F [label="ANN training\\n(class weights)"];
      G [label="Evaluation\\n(unseen test)"]; H [label="Saved model\\n+ scaler + features"];
      I [label="Streamlit\\ndashboard"];
      A->B->C->D->E->F->G->H->I;
    }""")
    st.markdown("""
### Technologies
`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `TensorFlow/Keras` · `Matplotlib` · `Seaborn` · `Plotly` · `Streamlit`

### Limitations
- Trained on one lab dataset (2017); real networks look different, so performance can drop on other traffic.
- Works on flow features produced by CICFlowMeter — it does not capture or sniff live packets.
- Binary only: it says *whether* a flow looks malicious, not *which* attack it is.
- The LOW / MEDIUM / HIGH threat level is a project visualisation rule, not an official severity standard.
""")


# ====================================================================== navigation
with st.sidebar:
    st.markdown("## 🛡️ NIDS Console")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.divider()
    ready = all(p.exists() for p in (config.MODEL_PATH, config.SCALER_PATH, config.FEATURE_COLUMNS_PATH))
    st.markdown(f"**Model:** {'🟢 loaded from models/' if ready else '🔴 not trained yet'}")
    st.markdown(f"**Threshold:** {config.PREDICTION_THRESHOLD}")
    if ready and not preprocessing_matches_model():
        st.error("preprocess.py was re-run after training. Run `python src/train_model.py` and "
                 "`python src/evaluate_model.py` again so the model matches the scaler.")
    st.divider()
    st.caption(DISCLAIMER)

{"Dashboard": page_dashboard, "Dataset Analysis": page_dataset_analysis,
 "Intrusion Detection": page_intrusion_detection, "Model Performance": page_model_performance,
 "About Project": page_about}[page]()
