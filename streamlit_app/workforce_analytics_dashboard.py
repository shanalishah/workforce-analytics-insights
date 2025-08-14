# workforce_analytics_dashboard.py
from pathlib import Path
import re
from typing import Dict, Tuple, Optional, List

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page config + subtle typography / spacing tweaks
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Workforce Analytics Dashboard", page_icon="📊", layout="wide")

# ── Inline Theme (replaces .streamlit/config.toml) ────────────────────────────
THEME = {
    "base_bg": "#FFFFFF",       # light background
    "text":    "#222222",
    "primary": "#0E7C86",       # your primaryColor
    "muted":   "#4A4A4A",
    "grid":    "#E9ECEF",
}

def apply_inline_theme():
    # Plotly: set a simple, consistent template that matches the theme
    template = dict(
        layout=dict(
            font=dict(color=THEME["text"]),
            paper_bgcolor=THEME["base_bg"],
            plot_bgcolor=THEME["base_bg"],
            colorway=[THEME["primary"]],
            xaxis=dict(gridcolor=THEME["grid"]),
            yaxis=dict(gridcolor=THEME["grid"]),
            legend=dict(bordercolor=THEME["grid"]),
        )
    )
    pio.templates["inline_theme"] = template
    pio.templates.default = "plotly+inline_theme"

    # Streamlit UI tweaks via CSS
    st.markdown(
        f"""
        <style>
        :root {{
            --primary-color: {THEME["primary"]};
            --text-color: {THEME["text"]};
            --bg-color: {THEME["base_bg"]};
        }}
        .stApp {{ background-color: var(--bg-color); color: var(--text-color); }}

        /* reduce top padding so H1 isn't clipped */
        .block-container {{ padding-top: 1.6rem !important; }}

        /* consistent heading spacing */
        h1, h2, h3 {{ line-height: 1.25 !important; margin-top: 0.35rem !important; margin-bottom: 0.35rem !important; }}

        /* Primary-colored buttons */
        .stButton>button, .stDownloadButton>button {{
            background: var(--primary-color) !important;
            color: white !important;
            border: 0 !important;
        }}

        /* Give charts a bit more breathing room from legends */
        .plot-container, .echarts-container {{ padding-bottom: 6px !important; }}

        /* Axis titles a touch bolder for clarity */
        .xtitle, .ytitle {{ font-weight: 600 !important; }}

        /* Metric color */
        [data-testid="stMetricValue"] {{ color: var(--primary-color) !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_inline_theme()

st.title("Workforce Analytics Dashboard")
st.caption("Enrollments • Training Outcomes • PCA (Dimensionality Reduction) • K-Means Segmentation • A/B Testing")

# ──────────────────────────────────────────────────────────────────────────────
# Paths & filenames
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
SEARCH_DIRS = [
    ROOT / "data" / "analysis-outputs",
    ROOT / "data" / "processed",
    ROOT / "data",
    ROOT / "data" / "raw",  # survey_questions.xlsx, nls_experiment.csv, nls_local_offices.csv
]

FILES = {
    "enroll":        ["country_enrollment_summary.csv"],
    "ass_by_course": ["course_assessment_by_course.csv"],
    "ass_summed":    ["course_assessment_summed.csv"],       # optional
    "improve":       ["assessment_improvement.csv", "AssessmentImprovement.csv"],   # optional
    "city_clusters": ["city_cluster_distribution.csv", "City_Cluster_Distribution.csv"],  # optional (wide format)
    "experiment":    ["nls_experiment.csv", "experiment_curriculum_cleaned.csv"],  # A/B (prefer the new file)
    "offices":       ["nls_local_offices.csv"],               # Office_ID → City
    "survey_loc":    ["emp_survey_with_locations.csv"],       # optional (Employee_ID → City/Country)
    "pca_workbook":  ["pca_components.xlsx"],                 # sheets: Loadings, ExplainedVariance, (optional) CityClusterDistribution
    "centers_xlsx":  ["pca_kmeans_results.xlsx"],             # sheet: KMeans_Cluster_Centers (Cluster, PC1, PC2, PC3, Percentage)
    "survey_qs":     ["survey_questions.xlsx", "survey_questions.csv"],  # QID, Question Text
}

# Optional friendly names for clusters (leave empty to show “Cluster 0/1/…”)
CLUSTER_LABELS: Dict[str, str] = {
    # "Cluster 0": "Career-Oriented Implementers",
    # "Cluster 1": "Operational Specialists",
    # "Cluster 2": "Skill Growth Seekers",
    # "Cluster 3": "Foundation Builders",
}

# A/B group mapping (by City)
AB_GROUPS = {
    # Control
    "New York": "Control",
    "Los Angeles": "Control",
    # A
    "Miami": "A",
    "Houston": "A",
    # B
    "Detroit": "B",
    "Denver": "B",
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def find_first(candidates: List[str]) -> Optional[Path]:
    for base in SEARCH_DIRS:
        for name in candidates:
            p = base / name
            if p.exists():
                return p
    return None

def read_csv_any(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            try:
                return pd.read_csv(path, low_memory=False, encoding=enc, engine="python")
            except Exception:
                continue
    return pd.read_csv(path, low_memory=False, encoding="utf-8", engine="python", on_bad_lines="skip", dtype=str)

def ensure_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def pc_order_val(label: str) -> int:
    m = re.search(r"PC\s*(\d+)", str(label), re.I)
    return int(m.group(1)) if m else 10_000

def wrap_text(s: str, width: int = 28) -> str:
    words, line, out = str(s).split(), "", []
    for w in words:
        if len(line) + len(w) + (1 if line else 0) <= width:
            line = (line + " " + w).strip()
        else:
            out.append(line); line = w
    if line: out.append(line)
    return "<br>".join(out)

def cluster_sort_key(val: str) -> int:
    m = re.search(r"(\d+)", str(val))
    return int(m.group(1)) if m else 9_999

def tidy_legend_bottom(fig, title_text=""):
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5, title=title_text),
        margin=dict(l=12, r=12, t=54, b=80),
    )

# ──────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_df(kind: str) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    p = find_first(FILES[kind])
    if not p:
        return None, None
    if p.suffix.lower() == ".csv":
        return read_csv_any(p), p
    try:
        return pd.read_excel(p), p
    except Exception:
        return None, p

@st.cache_data(show_spinner=False)
def load_qmap() -> Dict[str, str]:
    """Map Q1..Qn → full question text from survey_questions.* (supports CSV/XLSX)."""
    p = find_first(FILES["survey_qs"])
    if not p:
        return {}
    try:
        df = read_csv_any(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        qid_col = cols_lower.get("qid") or list(df.columns)[0]
        qtxt_col = next((c for c in df.columns if "question" in str(c).lower()), list(df.columns)[1])

        out = {}
        for _, r in df[[qid_col, qtxt_col]].dropna().iterrows():
            key = str(r[qid_col]).strip().upper()
            if re.match(r"^Q\d+$", key):
                out[key] = str(r[qtxt_col]).strip()
        return out
    except Exception:
        return {}

@st.cache_data(show_spinner=False)
def load_pca_workbook():
    """Return dict with: loadings, explained, city_pct (any may be None)."""
    p = find_first(FILES["pca_workbook"])
    res = {"loadings": None, "explained": None, "city_pct": None}
    if not p:
        return res

    # Loadings
    try:
        ld = pd.read_excel(p, sheet_name="Loadings")
        if "Response" not in ld.columns:
            ld = ld.rename(columns={ld.columns[0]: "Response"})
        res["loadings"] = ld
    except Exception:
        pass

    # Explained variance (accept multiple names)
    def read_ev(sheet_name):
        try:
            ev = pd.read_excel(p, sheet_name=sheet_name)
            pc_col = next((c for c in ev.columns if "principal" in str(c).lower()), ev.columns[0])
            var_col = next((c for c in ev.columns if "variance"  in str(c).lower()), ev.columns[1])
            ev = ev[[pc_col, var_col]].rename(columns={pc_col: "Principal Component", var_col: "Explained Variance (%)"})
            ev["Explained Variance (%)"] = (
                ev["Explained Variance (%)"].astype(str).str.replace("%","",regex=False).str.replace(",","",regex=False).str.strip()
            )
            ev["Explained Variance (%)"] = pd.to_numeric(ev["Explained Variance (%)"], errors="coerce")
            if ev["Explained Variance (%)"].max(skipna=True) <= 1.5:
                ev["Explained Variance (%)"] *= 100.0
            ev = ev.dropna(subset=["Explained Variance (%)"])
            if ev.empty:
                return None
            ev["__o"] = ev["Principal Component"].map(pc_order_val)
            ev = ev.sort_values("__o").drop(columns="__o")
            return ev
        except Exception:
            return None

    for name in ("ExplainedVariance", "Explained Variance", "EV", "Variance"):
        ev = read_ev(name)
        if ev is not None:
            res["explained"] = ev
            break

    # City cluster percentage (optional)
    try:
        city_pct = pd.read_excel(p, sheet_name="CityClusterDistribution")
        # Expect columns: City | Cluster | Percentage (percentage can be 0.5 or 50%)
        city_pct = city_pct.rename(columns={
            city_pct.columns[0]: "City",
            city_pct.columns[1]: "Cluster",
            city_pct.columns[2]: "Percentage",
        })
        city_pct["Cluster"] = city_pct["Cluster"].apply(
            lambda x: f"Cluster {int(x)}" if str(x).strip().isdigit() else str(x)
        )
        city_pct["Percentage"] = city_pct["Percentage"].astype(str).str.replace("%","",regex=False).str.strip()
        city_pct["Percentage"] = pd.to_numeric(city_pct["Percentage"], errors="coerce")
        if city_pct["Percentage"].max(skipna=True) > 1.5:
            city_pct["Percentage"] = city_pct["Percentage"] / 100.0
        res["city_pct"] = city_pct
    except Exception:
        pass

    return res

@st.cache_data(show_spinner=False)
def load_kmeans_centers():
    p = find_first(FILES["centers_xlsx"])
    if not p:
        return None, None
    df = pd.read_excel(p, sheet_name="KMeans_Cluster_Centers")
    if "Cluster" not in df.columns:
        df.insert(0, "Cluster", [f"Cluster {i}" for i in range(len(df))])
    else:
        df["Cluster"] = df["Cluster"].astype(str)
        df.loc[~df["Cluster"].str.contains("Cluster", case=False), "Cluster"] = "Cluster " + df["Cluster"]

    for col in ("PC1","PC2","PC3","Percentage"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if CLUSTER_LABELS:
        df["Cluster"] = df["Cluster"].map(lambda x: CLUSTER_LABELS.get(x, x))

    df = df.sort_values("Cluster", key=lambda s: s.map(cluster_sort_key)).reset_index(drop=True)
    return df, p

# ──────────────────────────────────────────────────────────────────────────────
# Load core data
# ──────────────────────────────────────────────────────────────────────────────
enr, _         = load_df("enroll")
ass_course, _  = load_df("ass_by_course")
ass_sum, _     = load_df("ass_summed")          # optional
improve, _     = load_df("improve")             # optional
city_pivot, _  = load_df("city_clusters")       # optional
experiment, _  = load_df("experiment")          # A/B
offices, _     = load_df("offices")             # Office_ID → City
survey_loc, _  = load_df("survey_loc")          # Employee_ID → City/Country (optional)

QTEXT  = load_qmap()
PCAWB  = load_pca_workbook()
CENTERS, centers_path = load_kmeans_centers()

# ──────────────────────────────────────────────────────────────────────────────
# KPI Row (use Explained Variance if available)
# ──────────────────────────────────────────────────────────────────────────────
kpi = {}
if enr is not None and not enr.empty:
    c_country = "Country_Regional_Center" if "Country_Regional_Center" in enr.columns else enr.columns[0]
    c_enroll  = "Total_Enrollments"       if "Total_Enrollments"       in enr.columns else enr.columns[1]
    enr[c_enroll] = ensure_num(enr[c_enroll])
    kpi["Total Enrollments"]     = f"{int(enr[c_enroll].sum(skipna=True)):,}"
    kpi["Countries"]             = enr[c_country].astype(str).nunique()

if ass_course is not None and "Course_Title" in ass_course.columns:
    kpi["Courses Analyzed"] = ass_course["Course_Title"].astype(str).nunique()

ev = PCAWB.get("explained")
if isinstance(ev, pd.DataFrame) and not ev.empty:
    total_var = float(ensure_num(ev["Explained Variance (%)"]).sum())
    total_var = max(0.0, min(total_var, 100.0))
    kpi["Explained Variance (PCA)"] = f"{total_var:.1f}%"

if kpi:
    cols = st.columns(min(4, len(kpi)))
    for (label, value), col in zip(kpi.items(), cols):
        col.metric(label, value)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📍 Enrollments", "🎯 Training Outcomes", "🧩 PCA & Segmentation", "🧪 A/B Testing"])

# ── Enrollments
with tab1:
    st.subheader("Enrollments by Country")
    if enr is None or enr.empty:
        st.info("Add `country_enrollment_summary.csv` to `data/analysis-outputs/`.")
    else:
        c_country = "Country_Regional_Center" if "Country_Regional_Center" in enr.columns else enr.columns[0]
        c_enroll  = "Total_Enrollments"       if "Total_Enrollments"       in enr.columns else enr.columns[1]
        enr[c_enroll] = ensure_num(enr[c_enroll])
        view = enr[[c_country, c_enroll]].dropna().copy().rename(columns={c_country: "Country", c_enroll: "Enrollments"})

        # Default: all countries (only ~11)
        picks = st.multiselect("Countries", options=sorted(view["Country"]), default=sorted(view["Country"]), key="enr_picks_all")
        order = st.radio("Sort by", ["Enrollments (desc)", "Country (A–Z)"], horizontal=True, key="enr_sort")

        if picks: view = view[view["Country"].isin(picks)]
        view = view.sort_values("Enrollments", ascending=False) if order.startswith("Enrollments") else view.sort_values("Country")

        if view.empty:
            st.info("No countries selected.")
        else:
            fig = px.bar(view, x="Country", y="Enrollments", title="Enrollments for Selected Countries", height=420)
            fig.update_layout(yaxis_title_standoff=12, title=dict(text="Enrollments for Selected Countries", pad=dict(t=8, b=2)))
            tidy_legend_bottom(fig, "")
            st.plotly_chart(fig, use_container_width=True, key="enr_plot")

# ── Training Outcomes
with tab2:
    st.subheader("Training Outcomes by Course and Delivery Mode")

    with st.expander("Methodology & Definitions", expanded=False):
        st.markdown(
            "- **Proficiency**: Learners’ self-rated skill level in the training domain.\n"
            "- **Application**: Learners’ confidence in applying those skills in real scenarios.\n"
            "- **Intake**: Baseline measurement before training.\n"
            "- **Outcome**: Post-training measurement.\n"
            "- **Change**: Improvement from Intake to Outcome (Outcome − Intake)."
        )

    if ass_course is None or ass_course.empty or "Course_Title" not in ass_course.columns:
        st.info("Add `course_assessment_by_course.csv`.")
    else:
        df = ass_course.copy()
        df["Delivery Mode"] = df["Course_Title"].apply(lambda t: "Virtual" if isinstance(t, str) and "virtual" in t.lower() else "In-Person")
        df["Δ Proficiency"]      = ensure_num(df.get("Outcome_Proficiency_Score"))  - ensure_num(df.get("Intake_Proficiency_Score"))
        df["Δ Application"]      = ensure_num(df.get("Outcome_Applications_Score")) - ensure_num(df.get("Intake_Applications_Score"))
        df["Proficiency (post)"] = ensure_num(df.get("Outcome_Proficiency_Score"))
        df["Application (post)"] = ensure_num(df.get("Outcome_Applications_Score"))

        metric_options = [
            "Proficiency — Change",
            "Application — Change",
            "Proficiency — Post-training score",
            "Application — Post-training score",
        ]
        col_map = {
            "Proficiency — Change": "Δ Proficiency",
            "Application — Change": "Δ Application",
            "Proficiency — Post-training score": "Proficiency (post)",
            "Application — Post-training score": "Application (post)",
        }

        c1, c2 = st.columns([1.05, 1])
        with c1:
            metric_label = st.selectbox("Metric", metric_options, index=1, key="metric_pick")
        with c2:
            course_sel = st.multiselect("Courses (optional)", options=sorted(df["Course_Title"].dropna().unique()), default=[], key="course_filter")

        metric_col = col_map[metric_label]
        df_plot = df if not course_sel else df[df["Course_Title"].isin(course_sel)]
        df_plot = df_plot.dropna(subset=[metric_col])

        if df_plot.empty:
            st.info("No rows with numeric values for the selected metric/courses.")
        else:
            g1, g2 = st.columns([1.05, 1])
            with g1:
                by_mode = df_plot.groupby("Delivery Mode", as_index=False)[metric_col].mean()
                fig = px.bar(by_mode, x="Delivery Mode", y=metric_col, title=f"{metric_label} by Delivery Mode", height=410)
                fig.update_layout(yaxis_title_standoff=14, title=dict(text=f"{metric_label} by Delivery Mode", pad=dict(t=8, b=2)))
                tidy_legend_bottom(fig, "")
                st.plotly_chart(fig, use_container_width=True, key="outcomes_mode")

            with g2:
                top = (df_plot.groupby("Course_Title", as_index=False)[metric_col]
                       .mean()
                       .sort_values(metric_col, ascending=False)
                       .head(15))
                top["_Course_Wrapped"] = top["Course_Title"].apply(lambda s: wrap_text(s, 28))
                fig2 = px.bar(top, y="_Course_Wrapped", x=metric_col, orientation="h",
                              title=f"{metric_label} — Top 15 Courses", height=540)
                fig2.update_traces(text=top[metric_col].round(2), textposition="outside", cliponaxis=False)
                fig2.update_yaxes(title="Course")
                fig2.update_layout(margin=dict(l=160, r=30, t=60, b=10),
                                   title=dict(text=f"{metric_label} — Top 15 Courses", pad=dict(t=8, b=2)))
                tidy_legend_bottom(fig2, "")
                st.plotly_chart(fig2, use_container_width=True, key="outcomes_top")

# ── PCA & Segmentation
with tab3:
    st.subheader("PCA Summary & K-Means Segmentation")

    # Explained variance
    st.markdown("#### PCA — Explained Variance")
    if isinstance(ev, pd.DataFrame) and not ev.empty:
        fig_ev = px.bar(ev, x="Principal Component", y="Explained Variance (%)",
                        title="Explained Variance by Component", height=320)
        fig_ev.update_layout(margin=dict(l=12, r=12, t=56, b=10),
                             title=dict(text="Explained Variance by Component", pad=dict(t=8, b=2)))
        st.plotly_chart(fig_ev, use_container_width=True, key="pca_ev")
    else:
        st.info("Add `ExplainedVariance` sheet to `pca_components.xlsx` with columns: Principal Component, Explained Variance (%).")

    # Top contributing survey questions (loadings) with QID → full text mapping
    st.markdown("#### PCA — Top Contributing Survey Questions")
    ld = PCAWB.get("loadings")
    if isinstance(ld, pd.DataFrame) and not ld.empty:
        # Identify component labels from the Loadings "Response" column if present
        if "Response" not in ld.columns:
            ld = ld.rename(columns={ld.columns[0]: "Response"})
        labels_from_ld = ld["Response"].astype(str).tolist()
        labels_from_ev = (ev["Principal Component"].astype(str).tolist()
                          if isinstance(ev, pd.DataFrame) and not ev.empty else [])
        # Prefer EV labels if they match count; else use loadings "Response"
        labels = labels_from_ev if labels_from_ev and len(labels_from_ev) <= len(labels_from_ld) else labels_from_ld

        pc_pick = st.selectbox("Component", labels, index=0,
                               help="Shows strongest contributing survey questions for the selected component.",
                               key="pc_pick_loadings")

        # Try exact match on Response; fallback to index position
        if pc_pick in ld["Response"].values:
            row = ld[ld["Response"] == pc_pick].iloc[0]
        else:
            row = ld.iloc[0]

        qcols = [c for c in ld.columns if re.match(r"^Q\d+$", str(c), re.I)]
        if not qcols:
            st.info("No Q1..Qn columns found in the Loadings sheet.")
        else:
            series = row[qcols].astype(float)
            # Map to full text where available
            series.index = [QTEXT.get(q.upper(), q) for q in series.index]
            top = series.abs().sort_values(ascending=False).head(10)
            display_vals = series.loc[top.index]  # keep sign

            plot_df = pd.DataFrame({
                "Question": top.index,
                "Influence (Loading)": display_vals.values
            })

            fig_ld = px.bar(plot_df, x="Influence (Loading)", y="Question", orientation="h",
                            title=f"Top Questions Influencing {pc_pick}", height=520)
            fig_ld.update_layout(margin=dict(l=40, r=18, t=60, b=10),
                                 title=dict(text=f"Top Questions Influencing {pc_pick}", pad=dict(t=8, b=2)))
            st.plotly_chart(fig_ld, use_container_width=True, key="pca_top_qs")
    else:
        st.info("Add `Loadings` sheet to `pca_components.xlsx` with a row per component and columns Q1..Qn.")

    # Segment distribution by city (optional)
    st.markdown("#### Segment Distribution by City")
    city_df = PCAWB.get("city_pct")
    if (city_df is None or city_df.empty) and (city_pivot is not None and not city_pivot.empty):
        dfc = city_pivot.copy()
        city_col = "City_y" if "City_y" in dfc.columns else dfc.columns[0]
        clust_cols = [c for c in dfc.columns if str(c).strip().isdigit()]
        if clust_cols:
            city_df = dfc.melt(id_vars=[city_col], value_vars=clust_cols,
                               var_name="Cluster", value_name="Employees").rename(columns={city_col: "City"})
            city_df["Cluster"] = city_df["Cluster"].apply(lambda x: f"Cluster {int(x)}" if str(x).isdigit() else str(x))

    if city_df is None or city_df.empty:
        st.info("Provide city distribution in `CityClusterDistribution` sheet or `city_cluster_distribution.csv`.")
    else:
        if "Percentage" in city_df.columns:
            city_df = city_df.copy()
            city_df["Percentage"] = ensure_num(city_df["Percentage"])
            fig_c = px.bar(city_df, x="City", y="Percentage", color="Cluster",
                           title="Segment Share by City", height=420)
            fig_c.update_layout(margin=dict(l=12, r=12, t=56, b=100),
                                title=dict(text="Segment Share by City", pad=dict(t=8, b=2)))
            tidy_legend_bottom(fig_c, "Cluster")
            st.plotly_chart(fig_c, use_container_width=True, key="city_pct")
        elif "Employees" in city_df.columns:
            fig_c = px.bar(city_df, x="City", y="Employees", color="Cluster",
                           title="Segment Counts by City", height=420)
            fig_c.update_layout(margin=dict(l=12, r=12, t=56, b=100),
                                title=dict(text="Segment Counts by City", pad=dict(t=8, b=2)))
            tidy_legend_bottom(fig_c, "Cluster")
            st.plotly_chart(fig_c, use_container_width=True, key="city_cnt")

    # K-Means centers: table + 2D pairs (NO 3D)
    st.markdown("#### K-Means Cluster Centers in PCA Space")
    if CENTERS is None or CENTERS.empty:
        st.warning("Add `pca_kmeans_results.xlsx` with sheet `KMeans_Cluster_Centers` (columns: Cluster, PC1, PC2, PC3, Percentage).")
    else:
        cols_to_show = ["Cluster"] + [c for c in ("PC1","PC2","PC3","Percentage") if c in CENTERS.columns]
        st.dataframe(CENTERS[cols_to_show], use_container_width=True, hide_index=True)

        have_pc12 = {"PC1","PC2"}.issubset(CENTERS.columns)
        have_pc13 = {"PC1","PC3"}.issubset(CENTERS.columns)
        have_pc23 = {"PC2","PC3"}.issubset(CENTERS.columns)

        figs = []
        if have_pc12:
            f = px.scatter(CENTERS, x="PC1", y="PC2", color="Cluster", text="Cluster", title="PC1 vs PC2 (Cluster Centers)")
            figs.append(("pc12", f))
        if have_pc13:
            f = px.scatter(CENTERS, x="PC1", y="PC3", color="Cluster", text="Cluster", title="PC1 vs PC3 (Cluster Centers)")
            figs.append(("pc13", f))
        if have_pc23:
            f = px.scatter(CENTERS, x="PC2", y="PC3", color="Cluster", text="Cluster", title="PC2 vs PC3 (Cluster Centers)")
            figs.append(("pc23", f))

        if figs:
            cols = st.columns(len(figs))
            for (k, fig), col in zip(figs, cols):
                fig.update_traces(marker=dict(size=12, opacity=0.9), textposition="top center")
                # Extra breathing room so axes labels don’t touch the legend
                fig.update_layout(margin=dict(l=14, r=14, t=64, b=110),
                                  height=420,
                                  title=dict(text=fig.layout.title.text, pad=dict(t=8, b=2)))
                tidy_legend_bottom(fig, "Cluster")
                col.plotly_chart(fig, use_container_width=True, key=f"kmeans_2d_{k}")

# ── A/B Testing
with tab4:
    st.subheader("A/B Testing — Curriculum Experiment")
    st.caption("Comparing **Control** (current program) vs **Curriculum A** vs **Curriculum B** using changes in Proficiency and Application scores. Δ = Outcome − Intake.")

    # Load experiment and attach cities
    if experiment is None or experiment.empty:
        st.info("Add `nls_experiment.csv` (preferably in `data/raw/`) to show A/B results.")
    else:
        df_exp = experiment.copy()

        # Make sure expected columns exist
        needed_cols = {"Employee_ID","Office_ID","Intake_Proficiency_Score","Intake_Applications_Score",
                       "Outcome_Proficiency_Score","Outcome_Applications_Score"}
        missing = needed_cols - set(df_exp.columns)
        if missing:
            st.error(f"Experiment file is missing columns: {', '.join(sorted(missing))}")
        else:
            # Merge city via Office_ID if available
            if offices is not None and not offices.empty and "Office_ID" in offices.columns and "City" in offices.columns:
                office_city = offices[["Office_ID","City"]].copy()
                office_city["Office_ID"] = pd.to_numeric(office_city["Office_ID"], errors="coerce")
                df_exp["Office_ID"] = pd.to_numeric(df_exp["Office_ID"], errors="coerce")
                df_exp = df_exp.merge(office_city, on="Office_ID", how="left")
            else:
                # fallback: if survey_loc is present
                if survey_loc is not None and not survey_loc.empty and {"Employee_ID","City_y"}.issubset(survey_loc.columns):
                    df_exp = df_exp.merge(
                        survey_loc[["Employee_ID","City_y"]].rename(columns={"City_y":"City"}),
                        on="Employee_ID", how="left"
                    )
                else:
                    df_exp["City"] = None

            # Normalize city names (strip, title-case; drop state abbreviations if present)
            def norm_city(x: Optional[str]) -> Optional[str]:
                if pd.isna(x): return None
                s = str(x).strip()
                # drop trailing ", XX"
                s = re.sub(r",\s*[A-Z]{2}$", "", s)
                return s

            df_exp["City"] = df_exp["City"].apply(norm_city)

            # Assign groups from city
            def city_to_group(city: Optional[str]) -> str:
                if not city:
                    return "Other / Not Mapped"
                for key, grp in AB_GROUPS.items():
                    if city == key:
                        return grp
                return "Other / Not Mapped"

            df_exp["Group"] = df_exp["City"].apply(city_to_group)

            # Compute deltas
            df_exp["Δ Proficiency"] = ensure_num(df_exp["Outcome_Proficiency_Score"]) - ensure_num(df_exp["Intake_Proficiency_Score"])
            df_exp["Δ Application"] = ensure_num(df_exp["Outcome_Applications_Score"]) - ensure_num(df_exp["Intake_Applications_Score"])

            # Summary table: Group, N, Avg Intake/Outcome, Δ, and % lift vs Control (when Control present)
            agg = df_exp.groupby("Group", as_index=False).agg(
                N=("Employee_ID","count"),
                Avg_Intake_Prof=("Intake_Proficiency_Score","mean"),
                Avg_Outcome_Prof=("Outcome_Proficiency_Score","mean"),
                Avg_Intake_App=("Intake_Applications_Score","mean"),
                Avg_Outcome_App=("Outcome_Applications_Score","mean"),
                Avg_Delta_Prof=("Δ Proficiency","mean"),
                Avg_Delta_App=("Δ Application","mean"),
            ).round(3)

            # Percent lift vs Control (Δ)
            ctrl_prof_val = agg.loc[agg["Group"]=="Control", "Avg_Delta_Prof"]
            ctrl_app_val  = agg.loc[agg["Group"]=="Control", "Avg_Delta_App"]
            ctrl_prof_val = float(ctrl_prof_val.iloc[0]) if not ctrl_prof_val.empty else None
            ctrl_app_val  = float(ctrl_app_val.iloc[0])  if not ctrl_app_val.empty  else None

            def pct_lift(v, base):
                if base is None or base == 0 or pd.isna(v):
                    return None
                return (v - base) / abs(base) * 100.0

            agg["Δ Prof vs Control (%)"] = agg["Avg_Delta_Prof"].apply(lambda v: round(pct_lift(v, ctrl_prof_val), 1) if pct_lift(v, ctrl_prof_val) is not None else None)
            agg["Δ App vs Control (%)"]  = agg["Avg_Delta_App"].apply(lambda v: round(pct_lift(v, ctrl_app_val), 1) if pct_lift(v, ctrl_app_val) is not None else None)

            # Order Control, A, B, Other
            order = pd.CategoricalDtype(categories=["Control","A","B","Other / Not Mapped"], ordered=True)
            agg["Group"] = agg["Group"].astype(order)
            agg = agg.sort_values("Group")

            st.markdown("**Experiment Summary**")
            st.dataframe(
                agg.rename(columns={
                    "N": "Participants",
                    "Avg_Intake_Prof": "Avg Intake (Prof.)",
                    "Avg_Outcome_Prof": "Avg Outcome (Prof.)",
                    "Avg_Intake_App": "Avg Intake (App.)",
                    "Avg_Outcome_App": "Avg Outcome (App.)",
                    "Avg_Delta_Prof": "Avg Δ Proficiency",
                    "Avg_Delta_App": "Avg Δ Application",
                }),
                use_container_width=True,
                hide_index=True
            )

            # Bar charts (Δ) — only if any group besides "Other / Not Mapped" exists
            assigned_mask = agg["Group"].isin(["Control","A","B"])
            if assigned_mask.any():
                plot_df = agg[assigned_mask].copy()

                c1, c2 = st.columns(2)
                with c1:
                    figA = px.bar(plot_df, x="Group", y="Avg_Delta_Prof",
                                  title="Average Δ Proficiency by Group", height=420,
                                  labels={"Avg_Delta_Prof": "Average Change in Proficiency"})
                    figA.update_layout(margin=dict(l=14, r=14, t=64, b=80),
                                       title=dict(text="Average Δ Proficiency by Group", pad=dict(t=8, b=2)))
                    tidy_legend_bottom(figA, "")
                    st.plotly_chart(figA, use_container_width=True, key="ab_prof")

                with c2:
                    figB = px.bar(plot_df, x="Group", y="Avg_Delta_App",
                                  title="Average Δ Application by Group", height=420,
                                  labels={"Avg_Delta_App": "Average Change in Application"})
                    figB.update_layout(margin=dict(l=14, r=14, t=64, b=80),
                                       title=dict(text="Average Δ Application by Group", pad=dict(t=8, b=2)))
                    tidy_legend_bottom(figB, "")
                    st.plotly_chart(figB, use_container_width=True, key="ab_app")
            else:
                st.info("Learners in this sample aren’t located in the six experimental cities. Showing table only.")

            # Short, recruiter-friendly note
            st.caption(
                "Definitions: **Proficiency** = self-rated skill; **Application** = confidence applying skills. "
                "Δ = Outcome − Intake. Groups are assigned by employee city using office mapping (NY/LA = Control; Miami/Houston = A; Detroit/Denver = B)."
            )

# ──────────────────────────────────────────────────────────────────────────────
# Footer (portfolio tag)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("© **Shan Ali Shah** — Workforce Analytics Portfolio")
