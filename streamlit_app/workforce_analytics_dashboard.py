# streamlit_app/workforce_analytics_dashboard.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------------------
# App config
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Workforce Analytics Dashboard", layout="wide")
st.markdown(
    """
    <style>
      /* prevent the page title from getting cut */
      h1, h2, h3 { line-height: 1.2; }
      .block-container { padding-top: 1.2rem; }
      /* tidy legends (bottom, horizontal) */
      .js-plotly-plot .legend { text-transform: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

REPO = Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
DATA = REPO.parent / "data"
ANALYSIS = DATA / "analysis-outputs"
RAW = DATA / "raw"

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def find_first(name_stem: str, exts=(".csv", ".xlsx", ".xls")) -> Optional[Path]:
    """Look in analysis-outputs then data for a file by stem (case-insensitive)."""
    for base in (ANALYSIS, DATA):
        for ext in exts:
            p = (base / f"{name_stem}{ext}")
            if p.exists():
                return p
    return None

@st.cache_data(show_spinner=False)
def read_table_auto(p: Path) -> pd.DataFrame:
    if p.suffix.lower() == ".csv":
        # try utf-8 then latin-1 to be robust
        try:
            return pd.read_csv(p, low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(p, low_memory=False, encoding="latin-1")
    else:
        return pd.read_excel(p)

def clean_percent(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "ifu":
        return s.astype(float)
    return (
        s.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan})
        .astype(float)
        / 100.0
    )

def human_metric_name(col: str) -> str:
    mapping = {
        "Score_Increase": "Change (Post – Pre)",
        "Score_Increase_Rounded": "Change (Post – Pre), rounded",
        "Intake_Proficiency_Score": "Pre-training: Proficiency",
        "Intake_Applications_Score": "Pre-training: Application",
        "Outcome_Proficiency_Score": "Post-training: Proficiency",
        "Outcome_Applications_Score": "Post-training: Application",
    }
    return mapping.get(col, col.replace("_", " ").title())

def parse_pc_names(explained_df: pd.DataFrame) -> Dict[str, str]:
    """
    From a table like:
      Principal Component | Explained Variance
      PC1 (Skill Development) | 31.90%
    build {'PC1':'Skill Development', 'PC2':'Operational Focus', ...}
    """
    comp_map = {}
    name_col = explained_df.columns[0]
    for raw in explained_df[name_col].astype(str):
        m = re.match(r"^(PC\d+)\s*\((.*?)\)", raw)
        if m:
            comp_map[m.group(1)] = m.group(2)
        else:
            # fallback: keep PCn
            tag = raw.split()[0]
            if tag.startswith("PC"):
                comp_map[tag] = tag
    return comp_map

@st.cache_data(show_spinner=False)
def load_survey_map() -> Dict[str, str]:
    """
    Map Q1..Q12 -> full question text from `data/raw/survey_questions.xlsx|csv`.
    """
    p = find_first("survey_questions", exts=(".xlsx", ".csv"))
    if p is None:
        return {}
    df = read_table_auto(p)
    # expected columns: QID, Question Text
    qid_col = next((c for c in df.columns if str(c).strip().lower() in ("qid", "question", "question_id")), None)
    text_col = next((c for c in df.columns if "text" in str(c).lower() or "question" == str(c).strip().lower() or "question text" in str(c).lower()), None)
    if qid_col is None or text_col is None:
        return {}
    mp = {}
    for _, r in df[[qid_col, text_col]].dropna().iterrows():
        qid = str(r[qid_col]).strip()
        if qid.upper().startswith("Q"):
            mp[qid.upper()] = str(r[text_col]).strip()
    return mp

def rename_q_columns(df: pd.DataFrame, qmap: Dict[str, str]) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        if re.fullmatch(r"Q\d+", str(c).strip(), flags=re.IGNORECASE):
            key = str(c).upper()
            label = qmap.get(key, key)
            new_cols[c] = f"{key} — {label}"
    return df.rename(columns=new_cols)

def computed_cluster_labels(centers: pd.DataFrame, pc_names: Dict[str, str]) -> Dict[str, str]:
    """
    Label clusters by the *dominant* absolute component (PC1/PC2/PC3) and its sign.
    Returns mapping 'Cluster 0' -> 'Cluster 0 — Dominant: PC1 (Skill Development ↑)'
    """
    lab = {}
    # ensure columns present
    pcs = [c for c in ["PC1", "PC2", "PC3"] if c in centers.columns]
    for idx, row in centers.iterrows():
        cid = row.get("Cluster", f"Cluster {idx}")
        # pick PC with max absolute value
        if pcs:
            comp = max(pcs, key=lambda k: abs(row[k]))
            pretty = pc_names.get(comp, comp)
            arrow = "↑" if row[comp] >= 0 else "↓"
            lab[cid] = f"{cid} — Dominant: {comp} ({pretty} {arrow})"
        else:
            lab[cid] = cid
    return lab

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------

# Outcomes (per-course)
outcomes_by_course_p = find_first("course_assessment_by_course")
if outcomes_by_course_p is None:
    st.error("Missing `course_assessment_by_course` in `data/analysis-outputs/`.")
    st.stop()
out_by_course = read_table_auto(outcomes_by_course_p)

# Country enrollments
enroll_country_p = find_first("country_enrollment_summary")
enroll_country = read_table_auto(enroll_country_p) if enroll_country_p else pd.DataFrame()

# PCA multi-sheet (recommended)
pca_multi_p = find_first("pca_components", exts=(".xlsx", ".xls"))
pca_loadings = explained = seg_city = pd.DataFrame()
pc_name_map: Dict[str, str] = {}

if pca_multi_p:
    pca_loadings = pd.read_excel(pca_multi_p, sheet_name="Loadings")
    explained = pd.read_excel(pca_multi_p, sheet_name="ExplainedVariance")
    seg_city = pd.read_excel(pca_multi_p, sheet_name="ClusterCenters", dtype={"Cluster": str})
    explained = explained.rename(columns={explained.columns[1]: "Explained Variance"})
    # parse PC names
    pc_name_map = parse_pc_names(explained)

# KMeans centers (alternative sheet/file for 2D/3D plot)
kmeans_alt_p = find_first("pca_kmeans_results", exts=(".xlsx", ".xls"))
kmeans_centers = pd.DataFrame()
if kmeans_alt_p:
    try:
        kmeans_centers = pd.read_excel(kmeans_alt_p, sheet_name="KMeans_Cluster_Centers")
        # If the sheet uses PC1/PC2/PC3 as columns without a 'Cluster' column, add it:
        if "Cluster" not in kmeans_centers.columns:
            kmeans_centers = kmeans_centers.copy()
            kmeans_centers.insert(0, "Cluster", [f"Cluster {i}" for i in range(len(kmeans_centers))])
    except Exception:
        kmeans_centers = pd.DataFrame()

# Map survey questions to full text
QMAP = load_survey_map()
if not pca_loadings.empty and QMAP:
    pca_loadings = rename_q_columns(pca_loadings, QMAP)

# ------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------
st.title("Workforce Analytics Dashboard")

# ------------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------------
tab_outcomes, tab_pca = st.tabs(
    ["Training Outcomes", "PCA & Segmentation"]
)

# ------------------------------------------------------------------------------
# TRAINING OUTCOMES
# ------------------------------------------------------------------------------
with tab_outcomes:
    col_left, col_right = st.columns([2, 5], gap="large")
    with col_left:
        st.subheader("Controls")
        # Metric
        metric_options = [
            "Outcome_Applications_Score",
            "Outcome_Proficiency_Score",
            "Intake_Applications_Score",
            "Intake_Proficiency_Score",
            "Score_Increase",
        ]
        metric = st.selectbox(
            "Metric",
            options=metric_options,
            index=0,
            format_func=human_metric_name,
            key="metric_sel",
        )

        # Optional course filter (multi)
        courses = st.multiselect(
            "Courses (optional)",
            options=sorted(out_by_course["Course_Title"].astype(str).unique()),
            key="course_filter",
        )

        # Per-course view only (we removed per-enrollment path to keep it simple & clean)
        st.caption(
            "Intake = pre-training • Outcome = post-training • Change = post minus pre"
        )

    with col_right:
        st.subheader("Training Outcomes by Course & Delivery Mode")
        df = out_by_course.copy()

        # Optional filter
        if courses:
            df = df[df["Course_Title"].astype(str).isin(courses)]

        if df.empty or metric not in df.columns:
            st.info("No rows with numeric values for the selected metric/courses.")
        else:
            # Prepare a tidy table for plotting
            plot_df = df.loc[:, ["Course_Title", metric]].copy()
            # Wrap long course names for readability (but DON'T show the internal column name)
            plot_df["Course"] = plot_df["Course_Title"].astype(str).apply(
                lambda s: "<br>".join(re.findall(".{1,35}(?:\\s|$)", s))
            )

            fig = px.bar(
                plot_df.sort_values(metric, ascending=False).head(15),
                x="Course",
                y=metric,
                labels={"Course": "Course", metric: human_metric_name(metric)},
            )
            fig.update_layout(
                height=480,
                margin=dict(l=10, r=10, t=30, b=100),
                xaxis_title="Course",
                yaxis_title=human_metric_name(metric),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5, xanchor="center", title_text=""),
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_outcomes")

# ------------------------------------------------------------------------------
# PCA & SEGMENTATION
# ------------------------------------------------------------------------------
with tab_pca:
    st.subheader("PCA — Top Contributing Survey Questions")

    if pca_loadings.empty:
        st.info("Add PCA loadings to `data/analysis-outputs/pca_components.xlsx` (sheet: **Loadings**).")
    else:
        # Columns that are questions after renaming (they contain " — " once mapped)
        q_cols = [c for c in pca_loadings.columns if c.startswith("Q") or " — " in c]
        # Users pick a component (row) to view
        comp_choices = pca_loadings.iloc[:, 0].astype(str).tolist()
        sel = st.selectbox("Component", options=comp_choices, index=0, key="pc_pick")

        row = pca_loadings[pca_loadings.iloc[:, 0].astype(str) == sel]
        if row.empty:
            st.info("No data for the selected component.")
        else:
            series = row[q_cols].T.rename(columns=row.columns[:1].tolist()).iloc[:, 0]
            top = series.abs().sort_values(ascending=False).head(10)
            plot_df = pd.DataFrame({"Question": top.index, "Loading (abs)": top.values})
            fig = px.bar(
                plot_df,
                x="Question",
                y="Loading (abs)",
                labels={"Question": "Survey Question", "Loading (abs)": "Absolute Loading"},
            )
            fig.update_layout(
                height=500,
                margin=dict(l=10, r=10, t=30, b=150),
                xaxis_tickangle=35,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0.5, xanchor="center", title_text=""),
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_pca_top")

    # --- KMeans centers: table + 2D & 3D views ---
    st.markdown("---")
    st.subheader("K-Means Cluster Centers in PCA Space")

    # Use the alternate centers sheet if present; else fall back to 'ClusterCenters' from pca_components.xlsx
    centers = pd.DataFrame()
    if not kmeans_centers.empty:
        centers = kmeans_centers.copy()
    elif not seg_city.empty and {"Cluster", "PC1", "PC2"}.issubset(seg_city.columns):
        centers = seg_city.rename(columns={"Cluster": "Cluster"}).copy()

    if centers.empty or "PC1" not in centers.columns or "PC2" not in centers.columns:
        st.info("Add centers with columns **Cluster, PC1, PC2** (and optionally **PC3**) to either "
                "`data/analysis-outputs/pca_kmeans_results.xlsx` (sheet `KMeans_Cluster_Centers`) "
                "or `data/analysis-outputs/pca_components.xlsx` (sheet `ClusterCenters`).")
    else:
        # Normalize cluster labels
        centers = centers.copy()
        centers["Cluster"] = centers["Cluster"].astype(str).apply(lambda s: s if s.startswith("Cluster") else f"Cluster {s}")

        # Build pretty labels using explained-variance names (if available)
        pretty_pc_names = {"PC1": "PC1", "PC2": "PC2", "PC3": "PC3"}
        if not explained.empty:
            pretty_pc_names.update(parse_pc_names(explained))

        label_map = computed_cluster_labels(centers, pretty_pc_names)
        centers["Cluster Label"] = centers["Cluster"].map(label_map)

        # Table
        show_cols = ["Cluster", "PC1", "PC2"] + (["PC3"] if "PC3" in centers.columns else [])
        st.dataframe(centers[show_cols], use_container_width=True)

        # 2D plots
        st.markdown("##### 2D Views")
        c1, c2, c3 = st.columns(3)
        with c1:
            if {"PC1", "PC2"}.issubset(centers.columns):
                fig12 = px.scatter(
                    centers, x="PC1", y="PC2", color="Cluster Label",
                    hover_name="Cluster", text="Cluster",
                )
                fig12.update_traces(textposition="top center")
                fig12.update_layout(
                    height=360,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5, xanchor="center", title_text=""),
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title=f"PC1 ({pretty_pc_names.get('PC1','PC1')})",
                    yaxis_title=f"PC2 ({pretty_pc_names.get('PC2','PC2')})",
                )
                st.plotly_chart(fig12, use_container_width=True, key="pc12")
        with c2:
            if {"PC1", "PC3"}.issubset(centers.columns):
                fig13 = px.scatter(
                    centers, x="PC1", y="PC3", color="Cluster Label",
                    hover_name="Cluster", text="Cluster",
                )
                fig13.update_traces(textposition="top center")
                fig13.update_layout(
                    height=360,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5, xanchor="center", title_text=""),
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title=f"PC1 ({pretty_pc_names.get('PC1','PC1')})",
                    yaxis_title=f"PC3 ({pretty_pc_names.get('PC3','PC3')})",
                )
                st.plotly_chart(fig13, use_container_width=True, key="pc13")
        with c3:
            if {"PC2", "PC3"}.issubset(centers.columns):
                fig23 = px.scatter(
                    centers, x="PC2", y="PC3", color="Cluster Label",
                    hover_name="Cluster", text="Cluster",
                )
                fig23.update_traces(textposition="top center")
                fig23.update_layout(
                    height=360,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5, xanchor="center", title_text=""),
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title=f"PC2 ({pretty_pc_names.get('PC2','PC2')})",
                    yaxis_title=f"PC3 ({pretty_pc_names.get('PC3','PC3')})",
                )
                st.plotly_chart(fig23, use_container_width=True, key="pc23")

        # 3D plot (if we have PC3)
        if "PC3" in centers.columns:
            st.markdown("##### 3D View")
            fig3d = px.scatter_3d(
                centers, x="PC1", y="PC2", z="PC3",
                color="Cluster Label", hover_name="Cluster",
            )
            fig3d.update_layout(
                height=520,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, x=0.5, xanchor="center", title_text=""),
                margin=dict(l=10, r=10, t=10, b=10),
                scene=dict(
                    xaxis_title=f"PC1 ({pretty_pc_names.get('PC1','PC1')})",
                    yaxis_title=f"PC2 ({pretty_pc_names.get('PC2','PC2')})",
                    zaxis_title=f"PC3 ({pretty_pc_names.get('PC3','PC3')})",
                ),
            )
            st.plotly_chart(fig3d, use_container_width=True, key="pc3d")
