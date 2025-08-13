# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------------------
# App config + light CSS (prevents giant title clipping on some browsers)
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Workforce Analytics Dashboard", layout="wide")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.25rem; }
      h1, h2, h3 { letter-spacing: 0.2px; }
      .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
      .stMetric { text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Paths & small helpers
# --------------------------------------------------------------------------------------
REPO = Path(__file__).parent if "__file__" in globals() else Path(".")
DATA_DIRS = [
    REPO.parent / "data" / "analysis-outputs",
    REPO.parent / "data" / "processed",
    REPO.parent / "data" / "raw",
    REPO / "data" / "analysis-outputs",
    REPO / "data" / "processed",
    REPO / "data" / "raw",
]

def find_path(basename_no_ext: str) -> Optional[Path]:
    """Return first matching path (csv/xlsx) in expected data dirs."""
    for d in DATA_DIRS:
        for ext in (".csv", ".xlsx", ".xls"):
            p = d / f"{basename_no_ext}{ext}"
            if p.exists():
                return p
    return None

def ensure_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def tidy_legend(fig, title=""):
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        legend_title_text=title,
        margin=dict(t=90, b=40)
    )
    return fig

# --------------------------------------------------------------------------------------
# Display names (metrics) — cleaned & non-repetitive
# --------------------------------------------------------------------------------------
PRETTY_METRIC = {
    "Intake_Proficiency_Score":   "Pre-training Proficiency (mean)",
    "Outcome_Proficiency_Score":  "Post-training Proficiency (mean)",
    "Intake_Applications_Score":  "Pre-training Application (mean)",
    "Outcome_Applications_Score": "Post-training Application (mean)",
    "Score_Increase":             "Average Change (post − pre)",
    "Score_Increase_Rounded":     "Average Change (rounded)",
}

# --------------------------------------------------------------------------------------
# Survey question map (Q1..Q12 -> full text) for PCA visuals
# --------------------------------------------------------------------------------------
SURVEY_Q_MAP = {
    "Q1":  "I prefer training that helps me enhance my job performance, without the need for group interaction.",
    "Q2":  "I’m motivated to learn new things that expand my abilities and knowledge beyond my day-to-day responsibilities.",
    "Q3":  "I am more interested in training that enhances my individual skills than in activities focused on team-building.",
    "Q4":  "I’m not interested in using training for networking or career changes; I value it more for improving my day-to-day work.",
    "Q5":  "I enjoy training that broadens my knowledge and helps me grow, even if it’s not directly related to my current role.",
    "Q6":  "I am motivated by training that not only helps me perform better in my current role but also enhances my overall skills and knowledge.",
    "Q7":  "I value training that makes me more effective in my job while also helping me grow professionally.",
    "Q8":  "I find value in training that supports my personal development and inspires me to grow in new directions.",
    "Q9":  "I look for training opportunities that allow me to develop my role-specific skills and learn new concepts that can help me in the future.",
    "Q10": "I’m more interested in improving my current role than in preparing for a new position or career change.",
    "Q11": "I find the most value in training that directly impacts my role, rather than in sessions involving group discussions.",
    "Q12": "I don’t see training as a way to transition into a new job; I prefer to use it to build on what I’m already doing.",
}

# Component names used in your analysis/reporting
PC_LABELS = {
    "PC1": "PC1 (Skill Development)",
    "PC2": "PC2 (Operational Focus)",
    "PC3": "PC3 (Career Advancement)",
}

# Cluster explanations (derived from your write-ups)
CLUSTER_EXPLAIN = {
    0: "Career-focused (PC3↑, slight PC2+, PC1↓)",
    1: "Operational-focused (PC2↑ / PC3↓)",
    2: "Skill-development focused (PC1↑ / PC2↓)",
    3: "Lower across themes",
}
def pretty_cluster(value) -> str:
    try:
        i = int(pd.to_numeric(str(value), errors="coerce"))
    except Exception:
        return str(value)
    return f"Cluster {i} — {CLUSTER_EXPLAIN.get(i, '').strip()}".rstrip(" —")

# --------------------------------------------------------------------------------------
# Data loaders
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv_any(basename_no_ext: str) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    p = find_path(basename_no_ext)
    if p is None:
        return None, None
    if p.suffix.lower() == ".csv":
        try:
            return pd.read_csv(p, low_memory=False), p
        except UnicodeDecodeError:
            # Windows-1252 fallback if CSV has % signs/other encodings
            return pd.read_csv(p, low_memory=False, encoding="cp1252"), p
    elif p.suffix.lower() in (".xlsx", ".xls"):
        # Return first sheet (or caller will read a specific sheet)
        return pd.read_excel(p), p
    return None, None

@st.cache_data(show_spinner=False)
def load_excel_sheet(basename_no_ext: str, sheet: str) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    p = find_path(basename_no_ext)
    if p is None or p.suffix.lower() not in (".xlsx", ".xls"):
        return None, None
    try:
        return pd.read_excel(p, sheet_name=sheet), p
    except Exception:
        return None, p

# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.title("Workforce Analytics Dashboard")

# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
tab_outcomes, tab_pca = st.tabs(["Training Outcomes", "PCA & Segmentation"])

# --------------------------------------------------------------------------------------
# TRAINING OUTCOMES TAB
# --------------------------------------------------------------------------------------
with tab_outcomes:
    with st.container():
        st.markdown(
            "**Methodology & Definitions** · "
            "Proficiency = knowledge mastery. Application = ability to apply skills on the job. "
            "Each measure is captured **pre-training (Intake)** and **post-training (Outcome)**; "
            "**Change = post − pre** on a 0–1 scaled score."
        )

        # Load course outcomes (aggregated by course)
        df_by_course, _p = load_csv_any("course_assessment_by_course")
        if df_by_course is None:
            st.warning("Add `course_assessment_by_course.csv` to `data/analysis-outputs/`.")
            st.stop()

        # Controls & chart in one container to reduce page jump
        st.markdown("### Course Outcomes by Delivery Mode")
        col_left, col_right = st.columns([2, 3])

        with col_left:
            metric_col = st.selectbox(
                "Metric",
                options=[
                    "Outcome_Proficiency_Score",
                    "Outcome_Applications_Score",
                    "Intake_Proficiency_Score",
                    "Intake_Applications_Score",
                    "Score_Increase",
                    "Score_Increase_Rounded",
                ],
                index=0,
                format_func=lambda c: PRETTY_METRIC.get(c, c),
            )
            # Optional multi-select for courses
            all_courses = sorted(df_by_course["Course_Title"].astype(str).unique().tolist())
            selected_courses = st.multiselect("Courses (optional)", all_courses)

        # Filter
        view_df = df_by_course.copy()
        if selected_courses:
            view_df = view_df[view_df["Course_Title"].astype(str).isin(selected_courses)]

        # Only numeric
        view_df[metric_col] = ensure_numeric(view_df[metric_col])
        view_df = view_df.dropna(subset=[metric_col, "Course_Title"])

        # Top-N sorted by metric (descending for change/post, sensible for pre as well)
        view_df = view_df.sort_values(metric_col, ascending=False).head(30)

        # Horizontal bars (readable course names)
        y_label = PRETTY_METRIC.get(metric_col, metric_col)
        fig_out = px.bar(
            view_df,
            x=metric_col,
            y="Course_Title",
            orientation="h",
            labels={metric_col: y_label, "Course_Title": "Course"},
            title=f"{y_label} — Top Courses",
        )
        fig_out.update_layout(margin=dict(t=80, b=40))
        st.plotly_chart(fig_out, use_container_width=True)

# --------------------------------------------------------------------------------------
# PCA & SEGMENTATION TAB
# --------------------------------------------------------------------------------------
with tab_pca:
    with st.container():
        st.markdown(
            "PCA condenses survey responses into three themes: **Skill Development (PC1)**, "
            "**Operational Focus (PC2)**, and **Career Advancement (PC3)**."
        )

        # 1) Loadings (Q1..Q12)
        loadings_df, _ = load_excel_sheet("pca_components", "Loadings")
        if loadings_df is None:
            st.warning(
                "Add `pca_components.xlsx` with a sheet named **Loadings** (columns: Response, "
                "Employee_ID, Q1..Q12)."
            )
            st.stop()

        # 2) Explained variance (optional but nice)
        ev_df, _ = load_excel_sheet("pca_components", "ExplainedVariance")
        if ev_df is not None and set(ev_df.columns[:2]) >= {"Principal Component", "Explained Variance"}:
            # show as small table
            ev_show = ev_df.copy()
            # Convert string percentages if needed
            ev_show["Explained Variance"] = ev_show["Explained Variance"].astype(str).str.rstrip("%")
            ev_show["Explained Variance"] = pd.to_numeric(ev_show["Explained Variance"], errors="coerce") / 100.0
            st.dataframe(
                ev_show.rename(columns={"Principal Component": "Component"}),
                use_container_width=True,
            )

        # --- PCA: Top contributing survey questions ---
        st.markdown("### PCA — Top Contributing Survey Questions")

        pc_options = [
            "PC1 (Skill Development)",
            "PC2 (Operational Focus)",
            "PC3 (Career Advancement)",
        ]
        selected_pc = st.selectbox("Component", pc_options, index=0)

        # Pull the single row for the chosen PC
        row = loadings_df.loc[
            loadings_df["Response"].astype(str).str.startswith(selected_pc)
        ].head(1)

        if row.empty:
            st.info("No data for the selected component.")
        else:
            q_cols = [c for c in row.columns if c.startswith("Q")]
            series = pd.to_numeric(row[q_cols].iloc[0], errors="coerce")
            top = series.abs().sort_values(ascending=False).head(10)
            plot_df = pd.DataFrame({
                "Survey Question": [SURVEY_Q_MAP.get(q, q) for q in top.index],
                "|Loading|": top.values
            })
            fig_topq = px.bar(
                plot_df, x="|Loading|", y="Survey Question", orientation="h",
                labels={"|Loading|": "Absolute Loading", "Survey Question": ""},
                title=f"Top Questions Influencing {selected_pc}"
            )
            tidy_legend(fig_topq, "")
            st.plotly_chart(fig_topq, use_container_width=True)

        st.markdown("### K-Means Cluster Centers in PCA Space")

        # Centers from your Excel (preferred)
        centers_df, centers_p = load_excel_sheet("pca_kmeans_results", "KMeans_Cluster_Centers")
        if centers_df is None:
            st.warning(
                "Add `pca_kmeans_results.xlsx` with sheet **KMeans_Cluster_Centers** "
                "(columns: PC1, PC2, PC3)."
            )
        else:
            centers = centers_df.copy()

            # If cluster label column is missing, create it 0..n-1
            if "Cluster" not in centers.columns:
                centers.insert(0, "Cluster", [f"{i}" for i in range(len(centers))])

            # Normalize cluster labels to int then pretty text
            centers["Cluster"] = (
                centers["Cluster"].astype(str).str.extract(r"(\d+)").astype(int).squeeze()
            )
            centers["Cluster"] = centers["Cluster"].map(pretty_cluster)

            # Show a neat table
            show_tbl = centers.rename(columns={
                "PC1": "PC1 (Skill Dev.)",
                "PC2": "PC2 (Operational)",
                "PC3": "PC3 (Career)",
            })
            st.dataframe(show_tbl, use_container_width=True)

            # 2D plots
            fig12 = px.scatter(
                centers, x="PC1", y="PC2", color="Cluster",
                labels={"PC1": PC_LABELS["PC1"], "PC2": PC_LABELS["PC2"]},
                title="Cluster Centers: PC1 vs PC2"
            )
            tidy_legend(fig12, "Cluster")
            st.plotly_chart(fig12, use_container_width=True)

            fig13 = px.scatter(
                centers, x="PC1", y="PC3", color="Cluster",
                labels={"PC1": PC_LABELS["PC1"], "PC3": PC_LABELS["PC3"]},
                title="Cluster Centers: PC1 vs PC3"
            )
            tidy_legend(fig13, "Cluster")
            st.plotly_chart(fig13, use_container_width=True)

            fig23 = px.scatter(
                centers, x="PC2", y="PC3", color="Cluster",
                labels={"PC2": PC_LABELS["PC2"], "PC3": PC_LABELS["PC3"]},
                title="Cluster Centers: PC2 vs PC3"
            )
            tidy_legend(fig23, "Cluster")
            st.plotly_chart(fig23, use_container_width=True)

            # 3D plot
            fig3d = px.scatter_3d(
                centers, x="PC1", y="PC2", z="PC3", color="Cluster",
                labels={"PC1": "PC1", "PC2": "PC2", "PC3": "PC3"},
                title="Cluster Centers: 3D (PC1–PC2–PC3)"
            )
            tidy_legend(fig3d, "Cluster")
            st.plotly_chart(fig3d, use_container_width=True)

# --------------------------------------------------------------------------------------
# Footer (subtle, not intrusive)
# --------------------------------------------------------------------------------------
st.caption("© Your Name — Workforce Analytics Portfolio")
