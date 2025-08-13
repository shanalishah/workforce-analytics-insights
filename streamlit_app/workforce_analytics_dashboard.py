# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------------------
# App config + small CSS polish (prevents title clipping; tidier spacing)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Workforce Analytics Dashboard", layout="wide")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; }
      h1, h2, h3 { letter-spacing: .2px; }
      .stTabs [data-baseweb="tab-list"] { gap: .5rem; }
      .stMetric { text-align: center; }
      .legend-compact .legendtitle { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# Paths & helpers
# ------------------------------------------------------------------------------
REPO = Path(__file__).parent if "__file__" in globals() else Path(".")
DATA_DIRS = [
    REPO.parent / "data" / "analysis-outputs",
    REPO.parent / "data" / "processed",
    REPO.parent / "data" / "raw",
    REPO / "data" / "analysis-outputs",
    REPO / "data" / "processed",
    REPO / "data" / "raw",
]

def _find(basename_no_ext: str) -> Optional[Path]:
    for d in DATA_DIRS:
        for ext in (".csv", ".xlsx", ".xls"):
            p = d / f"{basename_no_ext}{ext}"
            if p.exists():
                return p
    return None

@st.cache_data(show_spinner=False)
def load_any(basename_no_ext: str) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    p = _find(basename_no_ext)
    if not p:
        return None, None
    if p.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(p, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(p, low_memory=False, encoding="cp1252")
        return df, p
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p), p
    return None, None

@st.cache_data(show_spinner=False)
def load_excel_sheet(basename_no_ext: str, sheet: str) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    p = _find(basename_no_ext)
    if not p or p.suffix.lower() not in (".xlsx", ".xls"):
        return None, None
    try:
        return pd.read_excel(p, sheet_name=sheet), p
    except Exception:
        return None, p

def ensure_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def tidy_legend(fig, title=""):
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        legend_title_text=title,
        margin=dict(t=80, b=40)
    )
    return fig

# ------------------------------------------------------------------------------
# Friendly metric names
# ------------------------------------------------------------------------------
PRETTY_METRIC = {
    "Intake_Proficiency_Score":   "Pre-training Proficiency (mean)",
    "Outcome_Proficiency_Score":  "Post-training Proficiency (mean)",
    "Intake_Applications_Score":  "Pre-training Application (mean)",
    "Outcome_Applications_Score": "Post-training Application (mean)",
    "Score_Increase":             "Average Change (post − pre)",
    "Score_Increase_Rounded":     "Average Change (rounded)",
}

# ------------------------------------------------------------------------------
# Survey question map (Q1..Q12 → full text)
# ------------------------------------------------------------------------------
SURVEY_Q_MAP = {
    "Q1":  "Enhance job performance (no group interaction needed).",
    "Q2":  "Motivated to learn beyond day-to-day responsibilities.",
    "Q3":  "Prefer individual skill-building over team activities.",
    "Q4":  "Use training to improve current work, not networking/career change.",
    "Q5":  "Enjoy broadening knowledge even if not job-specific.",
    "Q6":  "Training should improve current role and overall skills.",
    "Q7":  "Value training that boosts job effectiveness and growth.",
    "Q8":  "Training should support personal development & inspiration.",
    "Q9":  "Look for role-specific skills and new concepts for future.",
    "Q10": "Prefer improving current role over preparing for a new one.",
    "Q11": "Value training that directly impacts role (vs. group sessions).",
    "Q12": "Don’t use training to transition jobs; build on current role.",
}

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
def pretty_cluster(v) -> str:
    try:
        i = int(pd.to_numeric(str(v), errors="coerce"))
    except Exception:
        return str(v)
    return f"Cluster {i} — {CLUSTER_EXPLAIN.get(i, '').strip()}".rstrip(" —")

# ------------------------------------------------------------------------------
# Header + KPIs (the “nice” summary row)
# ------------------------------------------------------------------------------
st.title("Workforce Analytics Dashboard")

# Load quick sources for KPIs (best-effort)
courses_df, _ = load_any("course_assessment_by_course")
enroll_df, _ = load_any("country_enrollment_summary")

k1, k2, k3, k4 = st.columns(4)
with k1:
    val = (courses_df["Course_Title"].nunique() if isinstance(courses_df, pd.DataFrame) else 0)
    st.metric("Courses Analyzed", f"{val:,}")
with k2:
    val = (enroll_df["Country_Regional_Center"].nunique() if isinstance(enroll_df, pd.DataFrame) else 0)
    st.metric("Countries Covered", f"{val:,}")
with k3:
    if isinstance(courses_df, pd.DataFrame) and "Score_Increase_Rounded" in courses_df.columns:
        st.metric("Avg Change (post − pre)", f"{courses_df['Score_Increase_Rounded'].mean():.2f}")
    else:
        st.metric("Avg Change (post − pre)", "—")
with k4:
    if isinstance(courses_df, pd.DataFrame) and "Score_Increase_Rounded" in courses_df.columns:
        row = courses_df.sort_values("Score_Increase_Rounded", ascending=False).head(1)
        label = row["Course_Title"].iloc[0] if not row.empty else "—"
        val = row["Score_Increase_Rounded"].iloc[0] if not row.empty else np.nan
        st.metric("Best Course Improvement", f"{val:.2f}" if not np.isnan(val) else "—", help=label)
    else:
        st.metric("Best Course Improvement", "—")

# ------------------------------------------------------------------------------
# Tabs (put “nice” views back + keep the cleaned ones)
# ------------------------------------------------------------------------------
tab_overview, tab_outcomes, tab_pca, tab_experiment = st.tabs(
    ["Enrollments", "Training Outcomes", "PCA & Segmentation", "Curriculum Experiment"]
)

# -------------------------------------------------------------------- Enrollments
with tab_overview:
    st.subheader("Training Utilization by Country")
    df, _ = load_any("country_enrollment_summary")
    if df is None or df.empty:
        st.info("Add `country_enrollment_summary.csv` to data/analysis-outputs/.")
    else:
        df = df.rename(columns={"Country_Regional_Center": "Country", "Total_Enrollments": "Enrollments"})
        df["Enrollments"] = ensure_num(df["Enrollments"])
        topn = st.slider("Top N countries (by enrollments)", 5, min(25, len(df)), min(10, len(df)))
        view = df.sort_values("Enrollments", ascending=False).head(topn)
        fig = px.bar(view, x="Enrollments", y="Country", orientation="h",
                     labels={"Enrollments": "Total Enrollments", "Country": ""})
        tidy_legend(fig, "")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------- Training Outcomes
with tab_outcomes:
    with st.container():
        st.markdown(
            "**Definitions** · Proficiency = knowledge mastery. "
            "Application = on-the-job application. "
            "Measured **pre (Intake)** and **post (Outcome)**. "
            "**Change = post − pre** on a 0–1 scale."
        )

        df, _ = load_any("course_assessment_by_course")
        if df is None or df.empty:
            st.warning("Add `course_assessment_by_course.csv` to data/analysis-outputs/.")
        else:
            st.markdown("### Course Outcomes by Delivery Mode")
            c1, c2 = st.columns([2, 3])
            with c1:
                metric = st.selectbox(
                    "Metric",
                    [
                        "Outcome_Proficiency_Score",
                        "Outcome_Applications_Score",
                        "Intake_Proficiency_Score",
                        "Intake_Applications_Score",
                        "Score_Increase",
                        "Score_Increase_Rounded",
                    ],
                    index=0,
                    format_func=lambda x: PRETTY_METRIC.get(x, x),
                )
                all_courses = sorted(df["Course_Title"].astype(str).unique())
                picked = st.multiselect("Courses (optional)", all_courses)

            view = df.copy()
            if picked:
                view = view[view["Course_Title"].astype(str).isin(picked)]

            # Make a delivery flag so in-person vs virtual are comparable
            view["Delivery"] = np.where(view["Course_Title"].str.contains("Virtual", case=False, na=False),
                                        "Virtual", "In-Person")
            # Clean metric and plot top rows
            view[metric] = ensure_num(view[metric])
            view = view.dropna(subset=[metric, "Course_Title"])
            view = view.sort_values(metric, ascending=False).head(30)

            fig = px.bar(
                view, x=metric, y="Course_Title", color="Delivery", orientation="h",
                labels={metric: PRETTY_METRIC.get(metric, metric), "Course_Title": "Course"},
                title=f"{PRETTY_METRIC.get(metric, metric)} — Top Courses"
            )
            tidy_legend(fig, "Delivery")
            st.plotly_chart(fig, use_container_width=True)

            # Optional download of the current view
            csv_bytes = view[["Course_Title", "Delivery", metric]].to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download current table (CSV)", csv_bytes, "course_outcomes_view.csv", "text/csv")

# --------------------------------------------------------------- PCA & Segmentation
with tab_pca:
    st.markdown(
        "PCA condenses survey responses into: **Skill Development (PC1)**, "
        "**Operational Focus (PC2)**, **Career Advancement (PC3)**."
    )

    # 1) Loadings (Q1..Q12)
    loadings, _ = load_excel_sheet("pca_components", "Loadings")
    if loadings is None or loadings.empty:
        st.warning("Add `pca_components.xlsx` with sheet **Loadings** (Response, Employee_ID, Q1..Q12).")
    else:
        # Explained variance (optional)
        ev, _ = load_excel_sheet("pca_components", "ExplainedVariance")
        if isinstance(ev, pd.DataFrame) and set(ev.columns[:2]) >= {"Principal Component", "Explained Variance"}:
            ev_show = ev.copy()
            ev_show["Explained Variance"] = (
                ev_show["Explained Variance"].astype(str).str.rstrip("%").astype(float) / 100.0
            )
            st.dataframe(ev_show.rename(columns={"Principal Component": "Component"}), use_container_width=True)

        st.markdown("### PCA — Top Contributing Survey Questions")
        pc_choice = st.selectbox(
            "Component",
            ["PC1 (Skill Development)", "PC2 (Operational Focus)", "PC3 (Career Advancement)"],
            index=0
        )
        row = loadings.loc[loadings["Response"].astype(str).str.startswith(pc_choice)].head(1)
        if row.empty:
            st.info("No data for the selected component.")
        else:
            q_cols = [c for c in row.columns if c.startswith("Q")]
            series = pd.to_numeric(row[q_cols].iloc[0], errors="coerce")
            top = series.abs().sort_values(ascending=False).head(10)
            plot_df = pd.DataFrame({"Survey Question": [SURVEY_Q_MAP.get(q, q) for q in top.index],
                                    "|Loading|": top.values})
            fig = px.bar(plot_df, x="|Loading|", y="Survey Question", orientation="h",
                         labels={"|Loading|": "Absolute Loading", "Survey Question": ""},
                         title=f"Top Questions Influencing {pc_choice}")
            tidy_legend(fig, "")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### K-Means Cluster Centers in PCA Space")

    # Preferred source: Excel with sheet KMeans_Cluster_Centers (PC1, PC2, PC3, optional Cluster)
    centers, _ = load_excel_sheet("pca_kmeans_results", "KMeans_Cluster_Centers")
    if centers is None or centers.empty:
        st.warning("Add `pca_kmeans_results.xlsx` with sheet **KMeans_Cluster_Centers** (PC1, PC2, PC3).")
    else:
        dfc = centers.copy()
        if "Cluster" not in dfc.columns:
            dfc.insert(0, "Cluster", range(len(dfc)))
        dfc["Cluster"] = (
            dfc["Cluster"].astype(str).str.extract(r"(\d+)").astype(int).squeeze()
        )
        dfc["Cluster"] = dfc["Cluster"].map(pretty_cluster)

        # Show compact table
        st.dataframe(
            dfc.rename(columns={"PC1": "PC1 (Skill Dev.)", "PC2": "PC2 (Operational)", "PC3": "PC3 (Career)"}),
            use_container_width=True
        )

        # 2D plots (PC1-PC2, PC1-PC3, PC2-PC3)
        def scatter2d(x, y, ttl):
            fig = px.scatter(
                dfc, x=x, y=y, color="Cluster",
                labels={x: PC_LABELS[x], y: PC_LABELS[y]},
                title=ttl
            )
            tidy_legend(fig, "Cluster")
            return fig

        st.plotly_chart(scatter2d("PC1", "PC2", "Cluster Centers: PC1 vs PC2"), use_container_width=True)
        st.plotly_chart(scatter2d("PC1", "PC3", "Cluster Centers: PC1 vs PC3"), use_container_width=True)
        st.plotly_chart(scatter2d("PC2", "PC3", "Cluster Centers: PC2 vs PC3"), use_container_width=True)

        # 3D
        fig3d = px.scatter_3d(
            dfc, x="PC1", y="PC2", z="PC3", color="Cluster",
            labels={"PC1": "PC1", "PC2": "PC2", "PC3": "PC3"},
            title="Cluster Centers: 3D (PC1–PC2–PC3)"
        )
        tidy_legend(fig3d, "Cluster")
        st.plotly_chart(fig3d, use_container_width=True)

    # City distribution view (the “nice” stacked share by city)
    st.markdown("### Segment Distribution by City")
    city_df, _ = load_any("city_cluster_distribution")
    if isinstance(city_df, pd.DataFrame) and not city_df.empty:
        # expected: City_y, columns 0..3 are counts
        city = city_df.rename(columns={"City_y": "City"}).copy()
        cluster_cols = [c for c in city.columns if str(c).strip().isdigit()]
        # To shares:
        city["Total"] = city[cluster_cols].sum(axis=1)
        for c in cluster_cols:
            city[str(c)] = city[c] / city["Total"]
        melted = city.melt(id_vars=["City"], value_vars=[str(c) for c in cluster_cols],
                           var_name="ClusterID", value_name="Share")
        melted["Cluster"] = melted["ClusterID"].astype(int).map(pretty_cluster)
        fig_city = px.bar(
            melted, x="City", y="Share", color="Cluster", barmode="stack",
            labels={"Share": "Employee Share", "City": ""},
            title="Cluster Share by City"
        )
        tidy_legend(fig_city, "Cluster")
        st.plotly_chart(fig_city, use_container_width=True)
    else:
        st.info("Add `city_cluster_distribution.csv` to show segment shares by city (optional).")

# -------------------------------------------------------------- Curriculum Experiment
with tab_experiment:
    st.subheader("Curriculum Experiment — Outcomes")
    exp, _ = load_any("experiment_curriculum_cleaned")
    if exp is None or exp.empty:
        st.info("Add `experiment_curriculum_cleaned.csv` to data/analysis-outputs/ (optional).")
    else:
        # Expect: Intake_Proficiency_Score / Outcome_Proficiency_Score / Proficiency_Improvement etc.
        for c in ["Intake_Proficiency_Score", "Outcome_Proficiency_Score",
                  "Intake_Applications_Score", "Outcome_Applications_Score",
                  "Proficiency_Improvement", "Applications_Improvement"]:
            if c in exp.columns:
                exp[c] = ensure_num(exp[c])

        # Summaries (mean improvements)
        s1, s2 = st.columns(2)
        if "Proficiency_Improvement" in exp.columns:
            s1.metric("Avg Proficiency Change", f"{exp['Proficiency_Improvement'].mean():.2f}")
        if "Applications_Improvement" in exp.columns:
            s2.metric("Avg Application Change", f"{exp['Applications_Improvement'].mean():.2f}")

        # Per-course mean change bars (if Course column exists)
        if "Course" in exp.columns and "Proficiency_Improvement" in exp.columns:
            view = exp.groupby("Course", as_index=False)["Proficiency_Improvement"].mean()
            view = view.sort_values("Proficiency_Improvement", ascending=False).head(25)
            fig_e = px.bar(view, x="Proficiency_Improvement", y="Course", orientation="h",
                           labels={"Proficiency_Improvement": "Mean Proficiency Change", "Course": ""},
                           title="Proficiency Change by Course (mean)")
            tidy_legend(fig_e, "")
            st.plotly_chart(fig_e, use_container_width=True)

# ------------------------------------------------------------------------------
st.caption("© Your Name — Workforce Analytics Portfolio")
