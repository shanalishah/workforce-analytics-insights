from pathlib import Path
import re
from typing import Dict, Tuple, Optional, List

import pandas as pd
import plotly.express as px
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page config + subtle typography / spacing tweaks
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Workforce Analytics Dashboard", page_icon="📊", layout="wide")
st.title("Workforce Analytics Dashboard")
st.caption("Enrollments • Training Outcomes • PCA (Dimensionality Reduction) • K-Means Segmentation • A/B Testing")

st.markdown("""
<style>
/* reduce top padding so H1 isn't clipped */
.block-container { padding-top: 1.6rem !important; }
/* consistent heading spacing */
h1, h2, h3 { line-height: 1.25 !important; margin-top: 0.35rem !important; margin-bottom: 0.35rem !important; }
/* keep legends from overlapping axes by adding a bit of bottom space */
.echarts-container, .plot-container { padding-bottom: 6px !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Paths & filenames
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
SEARCH_DIRS = [
    ROOT / "data" / "analysis-outputs",
    ROOT / "data" / "processed",
    ROOT / "data",
    ROOT / "data" / "raw",  # survey_questions lives here
]

FILES = {
    "enroll":        ["country_enrollment_summary.csv"],
    "ass_by_course": ["course_assessment_by_course.csv"],
    "ass_summed":    ["course_assessment_summed.csv"],       # optional
    "improve":       ["assessment_improvement.csv", "AssessmentImprovement.csv"],   # optional
    "city_clusters": ["city_cluster_distribution.csv", "City_Cluster_Distribution.csv"],  # optional (wide format)
    "experiment":    ["experiment_curriculum_cleaned.csv"],  # for A/B
    "survey_loc":    ["emp_survey_with_locations.csv"],      # Employee_ID → City/Country
    "pca_workbook":  ["pca_components.xlsx"],                # sheets: Loadings, ExplainedVariance, (optional) CityClusterDistribution
    "centers_xlsx":  ["pca_kmeans_results.xlsx"],            # sheet: KMeans_Cluster_Centers (Cluster, PC1, PC2, PC3, Percentage)
    "survey_qs":     ["survey_questions.xlsx", "survey_questions.csv"],  # QID, Question Text
}

# Optional friendly names for clusters (leave empty to show “Cluster 0/1/…”)
CLUSTER_LABELS: Dict[str, str] = {
    # "Cluster 0": "Career-Oriented Implementers",
    # "Cluster 1": "Operational Specialists",
    # "Cluster 2": "Skill Growth Seekers",
    # "Cluster 3": "Foundation Builders",
}

# # A/B group mapping (by City)
# AB_GROUPS = {
#     # Control
#     "New York": "Control",
#     "Los Angeles": "Control",
#     # A
#     "Miami": "A",
#     "Houston": "A",
#     # B
#     "Detroit": "B",
#     "Denver": "B",
# }

# A/B group mapping — flexible, recruiter-friendly
AB_CITY_MAP = {
    # Control
    "new york": "Control",
    "ny": "Control",
    "nyc": "Control",
    "los angeles": "Control",
    "la": "Control",
    # A
    "miami": "A",
    "houston": "A",
    # B
    "detroit": "B",
    "denver": "B",
}

def normalize_city_name(raw: Optional[str]) -> Optional[str]:
    """Lowercase, strip, remove state/country tails and punctuation variants."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip().lower()

    # Remove common state tails (", ca", ", ny", " - usa", etc.)
    # Keep only the first token before comma / dash
    for sep in [",", " - ", " — ", " – "]:
        if sep in s:
            s = s.split(sep)[0].strip()

    # Normalize a few short aliases
    alias = {
        "la": "los angeles",
        "nyc": "new york",
        "ny": "new york",
    }
    s = alias.get(s, s)
    return s

def assign_ab_group(city: Optional[str]) -> str:
    """Return Control / A / B / Other based on normalized city."""
    base = normalize_city_name(city)
    if not base:
        return "Other / Not Mapped"
    # exact match first
    if base in AB_CITY_MAP:
        return AB_CITY_MAP[base]
    # startswith / fuzzy-ish fallback (covers e.g. "los angeles downtown")
    for key in AB_CITY_MAP:
        if base.startswith(key):
            return AB_CITY_MAP[key]
    return "Other / Not Mapped"


# Common aliases & normalization for cities
AB_ALIASES = {
    "nyc": "New York",
    "new york": "New York",
    "la": "Los Angeles",
    "l.a.": "Los Angeles",
    "los angeles": "Los Angeles",
    "los angeles ca": "Los Angeles",
    "miami": "Miami",
    "houston": "Houston",
    "detroit": "Detroit",
    "denver": "Denver",
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
        margin=dict(l=16, r=16, t=64, b=100),
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
    """Map Q1..Q12 → full question text from survey_questions.* (supports CSV/XLSX)."""
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
survey_loc, _  = load_df("survey_loc")          # Employee_ID → City/Country

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
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Enrollments",
    "🎯 Training Outcomes",
    "🧩 PCA & Segmentation",
    "🧪 A/B Testing",
])

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
            "- **Proficiency (skill level)**: Learners’ self-rated skill in the training domain.\n"
            "- **Application (confidence to apply)**: Learners’ confidence in using those skills on the job.\n"
            "- **Intake**: Baseline before training.\n"
            "- **Outcome**: Post-training measurement.\n"
            "- **Change (Δ)**: Outcome − Intake."
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
        fig_ev.update_layout(margin=dict(l=16, r=16, t=56, b=14),
                             title=dict(text="Explained Variance by Component", pad=dict(t=8, b=2)))
        st.plotly_chart(fig_ev, use_container_width=True, key="pca_ev")
    else:
        st.info("Add `ExplainedVariance` sheet to `pca_components.xlsx` with columns: Principal Component, Explained Variance (%).")

    # Top contributing survey questions (loadings) with QID → full text mapping
    st.markdown("#### PCA — Top Contributing Survey Questions")

    def select_loadings_row(ldf: pd.DataFrame, label: str) -> pd.Series:
        """Find the row for the selected component by exact label, then by PC number, else first row."""
        if "Response" in ldf.columns:
            exact = ldf.index[ldf["Response"].astype(str).str.casefold() == str(label).casefold()].tolist()
            if exact:
                return ldf.iloc[exact[0]]
        m = re.search(r"PC\s*(\d+)", str(label), re.I)
        if m:
            n = int(m.group(1))
            cand = ldf.index[ldf["Response"].astype(str).str.contains(fr"PC\s*{n}\b", case=False, regex=True)].tolist()
            if cand:
                return ldf.iloc[cand[0]]
            if 0 <= n-1 < len(ldf):
                return ldf.iloc[n-1]
        return ldf.iloc[0]

    ld = PCAWB.get("loadings")
    if isinstance(ld, pd.DataFrame) and not ld.empty:
        if "Response" not in ld.columns:
            ld = ld.rename(columns={ld.columns[0]: "Response"})

        labels = (ev["Principal Component"].astype(str).tolist()
                  if isinstance(ev, pd.DataFrame) and not ev.empty
                  else [str(x) for x in ld["Response"].tolist()])

        pc_pick = st.selectbox(
            "Component",
            labels,
            index=0,
            help="Shows strongest contributing survey questions for the selected component.",
            key="pc_pick_loadings"
        )

        row = select_loadings_row(ld, pc_pick)

        qcols = [c for c in ld.columns if re.match(r"^Q\d+$", str(c), re.I)]
        if not qcols:
            st.info("No Q1..Qn columns found in the Loadings sheet.")
        else:
            series = row[qcols].T.iloc[:, 0] if isinstance(row, pd.DataFrame) else row[qcols].T
            # Map QIDs to full text where available
            idx = []
            for q in series.index:
                q_up = str(q).upper().strip()
                label = QTEXT.get(q_up, q_up)
                idx.append(label)
            series.index = idx

            # top absolute contributors but keep sign for display
            abs_top = series.abs().sort_values(ascending=False).head(10)
            display_vals = series.loc[abs_top.index]

            plot_df = pd.DataFrame({"Question": abs_top.index, "Influence (Loading)": display_vals.values})
            fig_ld = px.bar(plot_df, x="Influence (Loading)", y="Question", orientation="h",
                            title=f"Top Questions Influencing {pc_pick}", height=520)
            fig_ld.update_layout(margin=dict(l=40, r=18, t=60, b=10),
                                 title=dict(text=f"Top Questions Influencing {pc_pick}", pad=dict(t=8, b=2)))
            st.plotly_chart(fig_ld, use_container_width=True, key="pca_top_qs")
    else:
        st.info("Add `Loadings` sheet to `pca_components.xlsx` with a row per component and columns Q1..Q12.")

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
            fig_c.update_layout(margin=dict(l=16, r=16, t=56, b=110),
                                title=dict(text="Segment Share by City", pad=dict(t=8, b=2)))
            tidy_legend_bottom(fig_c, "Cluster")
            st.plotly_chart(fig_c, use_container_width=True, key="city_pct")
        elif "Employees" in city_df.columns:
            fig_c = px.bar(city_df, x="City", y="Employees", color="Cluster",
                           title="Segment Counts by City", height=420)
            fig_c.update_layout(margin=dict(l=16, r=16, t=56, b=110),
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
                fig.update_traces(marker=dict(size=11, opacity=0.95), textposition="top center")
                fig.update_xaxes(automargin=True)
                fig.update_yaxes(automargin=True)
                fig.update_layout(margin=dict(l=18, r=18, t=70, b=110),
                                  height=440,
                                  title=dict(text=fig.layout.title.text, pad=dict(t=10, b=2)))
                tidy_legend_bottom(fig, "Cluster")
                col.plotly_chart(fig, use_container_width=True, key=f"kmeans_2d_{k}")

# # ── A/B Testing
# with tab4:
#     st.subheader("A/B Testing — Curriculum Experiment")
#     with st.expander("What this shows", expanded=False):
#         st.markdown(
#             "- **Goal**: Compare **Control** (current program) vs **Curriculum A** vs **Curriculum B**.\n"
#             "- **Metrics**:\n"
#             "  - **Proficiency (skill level)** — average learner self-rating of skill.\n"
#             "  - **Application (confidence to apply)** — average confidence to use the skill at work.\n"
#             "  - **Δ (change)** = Outcome − Intake (how much learners improved).\n"
#             "- **Grouping**: Learners are assigned to Control/A/B by office city."
#         )

#     if experiment is None or experiment.empty:
#         st.info("Add `experiment_curriculum_cleaned.csv` to show A/B results.")
#     else:
#         df_exp = experiment.copy()

#         # Attach City by Employee_ID (primary) and Office_ID (fallback)
#         if survey_loc is not None and not survey_loc.empty:
#             merged = False
#             if "Employee_ID" in df_exp.columns and "Employee_ID" in survey_loc.columns:
#                 df_exp = df_exp.merge(
#                     survey_loc[["Employee_ID", "City_y"]].rename(columns={"City_y": "City"}),
#                     on="Employee_ID", how="left"
#                 )
#                 merged = True
#             if (not merged) and "Office_ID" in df_exp.columns and "Office_ID" in survey_loc.columns:
#                 df_exp = df_exp.merge(
#                     survey_loc[["Office_ID", "City_y"]].drop_duplicates().rename(columns={"City_y": "City"}),
#                     on="Office_ID", how="left"
#                 )
#         if "City" not in df_exp.columns:
#             df_exp["City"] = None

#         # Normalize city strings → City_norm
#         df_exp["City"] = df_exp["City"].astype(str).str.strip()
#         # drop trailing ", ST"
#         df_exp["City_norm"] = df_exp["City"].str.replace(r",\s*[A-Za-z]{2}$", "", regex=True).str.lower().str.strip()
#         # apply aliases
#         df_exp["City_norm"] = df_exp["City_norm"].map(lambda x: AB_ALIASES.get(x, x) if isinstance(x, str) else x)

#         # Auto map by normalized city
#         AB_LOWER = {k.lower(): v for (k, v) in AB_GROUPS.items()}
#         def auto_group(cn: str) -> str:
#             if not isinstance(cn, str):
#                 return "Other / Not Mapped"
#             return AB_LOWER.get(cn.lower(), "Other / Not Mapped")
#         df_exp["Group"] = df_exp["City_norm"].apply(auto_group)

#         # ── Manual mapping UI (optional, fixes “Other / Not Mapped” quickly)
#         st.markdown("##### Optional: Adjust city-to-group mapping")
#         unique_cities = sorted(set(df_exp["City_norm"].dropna()))
#         other_cities = sorted([c for c in unique_cities if auto_group(c) == "Other / Not Mapped"])

#         with st.expander("Assign cities to Control / A / B (only if needed)", expanded=False):
#             c1, c2, c3 = st.columns(3)
#             default_ctrl = [c for c in unique_cities if auto_group(c) == "Control"]
#             default_a    = [c for c in unique_cities if auto_group(c) == "A"]
#             default_b    = [c for c in unique_cities if auto_group(c) == "B"]

#             sel_ctrl = c1.multiselect("Control cities", options=unique_cities, default=default_ctrl, key="ab_ctrl")
#             sel_a    = c2.multiselect("A cities",       options=unique_cities, default=default_a,    key="ab_a")
#             sel_b    = c3.multiselect("B cities",       options=unique_cities, default=default_b,    key="ab_b")

#             manual_map = {c: "Control" for c in sel_ctrl}
#             manual_map.update({c: "A" for c in sel_a})
#             manual_map.update({c: "B" for c in sel_b})

#             if manual_map:
#                 df_exp["Group"] = df_exp["City_norm"].map(lambda c: manual_map.get(c, auto_group(c)))

#         # Ensure numeric deltas exist or compute from intake/outcome
#         prof_col = "Proficiency_Improvement" if "Proficiency_Improvement" in df_exp.columns else "Proficiency_Delta"
#         app_col  = "Applications_Improvement" if "Applications_Improvement" in df_exp.columns else "Applications_Delta"
#         if prof_col not in df_exp.columns and {"Outcome_Proficiency_Score","Intake_Proficiency_Score"}.issubset(df_exp.columns):
#             df_exp["Proficiency_Improvement"] = ensure_num(df_exp["Outcome_Proficiency_Score"]) - ensure_num(df_exp["Intake_Proficiency_Score"])
#             prof_col = "Proficiency_Improvement"
#         if app_col not in df_exp.columns and {"Outcome_Applications_Score","Intake_Applications_Score"}.issubset(df_exp.columns):
#             df_exp["Applications_Improvement"] = ensure_num(df_exp["Outcome_Applications_Score"]) - ensure_num(df_exp["Intake_Applications_Score"])
#             app_col = "Applications_Improvement"

#         # Prepare intake/outcome numeric series (if present)
#         df_exp["Intake_Proficiency_Score"]   = ensure_num(df_exp.get("Intake_Proficiency_Score"))
#         df_exp["Outcome_Proficiency_Score"]  = ensure_num(df_exp.get("Outcome_Proficiency_Score"))
#         df_exp["Intake_Applications_Score"]  = ensure_num(df_exp.get("Intake_Applications_Score"))
#         df_exp["Outcome_Applications_Score"] = ensure_num(df_exp.get("Outcome_Applications_Score"))

#         df_exp[prof_col] = ensure_num(df_exp[prof_col]) if prof_col in df_exp.columns else pd.NA
#         df_exp[app_col]  = ensure_num(df_exp[app_col])  if app_col  in df_exp.columns else pd.NA

#         # Participants = simple count per group
#         df_exp["_participant"] = 1

#         # Summary by group (means + counts)
#         gb = df_exp.groupby("Group", as_index=False)
#         summary = gb.agg({
#             "_participant": "count",
#             "Intake_Proficiency_Score": "mean",
#             "Outcome_Proficiency_Score":"mean",
#             prof_col: "mean",
#             "Intake_Applications_Score": "mean",
#             "Outcome_Applications_Score":"mean",
#             app_col: "mean",
#         }).rename(columns={
#             "_participant": "Participants",
#             "Intake_Proficiency_Score":  "Avg Intake (Skill)",
#             "Outcome_Proficiency_Score": "Avg Outcome (Skill)",
#             prof_col:                     "Avg Δ Skill",
#             "Intake_Applications_Score":  "Avg Intake (Confidence)",
#             "Outcome_Applications_Score": "Avg Outcome (Confidence)",
#             app_col:                      "Avg Δ Confidence",
#         })

#         # Safe ordering: Control, A, B, Other / Not Mapped
#         order = pd.CategoricalDtype(categories=["Control", "A", "B", "Other / Not Mapped"], ordered=True)
#         summary["Group"] = summary["Group"].astype(order)
#         summary = summary.sort_values("Group")

#         # Δ sanity if Δ columns missing
#         if ("Avg Δ Skill" not in summary.columns or summary["Avg Δ Skill"].isna().all()) and \
#            {"Avg Outcome (Skill)","Avg Intake (Skill)"}.issubset(summary.columns):
#             summary["Avg Δ Skill"] = ensure_num(summary["Avg Outcome (Skill)"]) - ensure_num(summary["Avg Intake (Skill)"])
#         if ("Avg Δ Confidence" not in summary.columns or summary["Avg Δ Confidence"].isna().all()) and \
#            {"Avg Outcome (Confidence)","Avg Intake (Confidence)"}.issubset(summary.columns):
#             summary["Avg Δ Confidence"] = ensure_num(summary["Avg Outcome (Confidence)"]) - ensure_num(summary["Avg Intake (Confidence)"])

#         # % change vs Control (safe if control exists & non-zero)
#         ctrl_skill = summary.loc[summary["Group"]=="Control", "Avg Δ Skill"]
#         ctrl_conf  = summary.loc[summary["Group"]=="Control", "Avg Δ Confidence"]
#         ctrl_skill_val = float(ctrl_skill.iloc[0]) if len(ctrl_skill) and pd.notna(ctrl_skill.iloc[0]) else None
#         ctrl_conf_val  = float(ctrl_conf.iloc[0])  if len(ctrl_conf)  and pd.notna(ctrl_conf.iloc[0])  else None

#         def pct_lift_safe(v, base):
#             if base is None or pd.isna(base) or base == 0 or v is None or pd.isna(v):
#                 return None
#             return (v - base) / abs(base) * 100.0

#         def fmt_pct(x):
#             return f"{x:.1f}%" if x is not None else "—"

#         summary["% vs Control (Skill Δ)"]      = summary["Avg Δ Skill"].apply(lambda v: fmt_pct(pct_lift_safe(v, ctrl_skill_val)))
#         summary["% vs Control (Confidence Δ)"] = summary["Avg Δ Confidence"].apply(lambda v: fmt_pct(pct_lift_safe(v, ctrl_conf_val)))

#         # Final recruiter-friendly table
#         display_cols = [
#             "Group",
#             "Participants",
#             "Avg Intake (Skill)","Avg Outcome (Skill)","Avg Δ Skill","% vs Control (Skill Δ)",
#             "Avg Intake (Confidence)","Avg Outcome (Confidence)","Avg Δ Confidence","% vs Control (Confidence Δ)",
#         ]
#         st.dataframe(summary[display_cols].round(3), use_container_width=True, hide_index=True)

#         # Bars for Δ
#         mean_df = summary[["Group","Avg Δ Skill","Avg Δ Confidence"]].copy()
#         c1, c2 = st.columns(2)
#         with c1:
#             figA = px.bar(mean_df.sort_values("Group"), x="Group", y="Avg Δ Skill",
#                           title="Average Change in Skill (Proficiency)", height=420,
#                           labels={"Avg Δ Skill": "Average Δ (Skill)"})
#             figA.update_layout(margin=dict(l=16, r=16, t=64, b=80),
#                                title=dict(text="Average Change in Skill (Proficiency)", pad=dict(t=8, b=2)))
#             tidy_legend_bottom(figA, "")
#             st.plotly_chart(figA, use_container_width=True, key="ab_prof")

#         with c2:
#             figB = px.bar(mean_df.sort_values("Group"), x="Group", y="Avg Δ Confidence",
#                           title="Average Change in Confidence (Application)", height=420,
#                           labels={"Avg Δ Confidence": "Average Δ (Confidence to apply)"})
#             figB.update_layout(margin=dict(l=16, r=16, t=64, b=80),
#                                title=dict(text="Average Change in Confidence (Application)", pad=dict(t=8, b=2)))
#             tidy_legend_bottom(figB, "")
#             st.plotly_chart(figB, use_container_width=True, key="ab_app")

#         st.caption("Definitions: **Skill** = Proficiency (self-rated skill level). **Confidence** = Application (confidence to apply the skill). **Δ** = Outcome – Intake.")

# ── A/B Testing
with tab4:
    st.subheader("A/B Testing — Curriculum Experiment")
    st.caption("Comparing Control vs. Treatment groups using pre/post assessments: **Skill Knowledge** (formerly “Proficiency”) and **Practical Application**.")

    if experiment is None or experiment.empty:
        st.info("Add `experiment_curriculum_cleaned.csv` to show A/B results.")
    else:
        df_exp = experiment.copy()

        # Ensure City is present (merge from survey_loc if needed)
        if "City" not in df_exp.columns:
            if survey_loc is not None and not survey_loc.empty and "Employee_ID" in survey_loc.columns:
                df_exp = df_exp.merge(
                    survey_loc[["Employee_ID", "City_y"]].rename(columns={"City_y": "City"}),
                    on="Employee_ID",
                    how="left"
                )
            else:
                df_exp["City"] = None

        # --- Robust group assignment (inline fallback if not defined globally) ---
        if "assign_ab_group" not in globals():
            AB_CITY_MAP = {
                # Control
                "new york": "Control", "ny": "Control", "nyc": "Control",
                "los angeles": "Control", "la": "Control",
                # A
                "miami": "A", "houston": "A",
                # B
                "detroit": "B", "denver": "B",
            }
            def normalize_city_name(raw):
                if raw is None or (isinstance(raw, float) and pd.isna(raw)): return None
                s = str(raw).strip().lower()
                for sep in [",", " - ", " — ", " – "]:
                    if sep in s:
                        s = s.split(sep)[0].strip()
                alias = {"la": "los angeles", "nyc": "new york", "ny": "new york"}
                s = alias.get(s, s)
                return s
            def assign_ab_group(city):
                base = normalize_city_name(city)
                if not base:
                    return "Other / Not Mapped"
                if base in AB_CITY_MAP:
                    return AB_CITY_MAP[base]
                for key in AB_CITY_MAP:
                    if base.startswith(key):
                        return AB_CITY_MAP[key]
                return "Other / Not Mapped"

        # Assign group
        df_exp["Group"] = df_exp["City"].apply(assign_ab_group)

        # Column names (robust to variants)
        prof_pre  = "Intake_Proficiency_Score"
        prof_post = "Outcome_Proficiency_Score"
        app_pre   = "Intake_Applications_Score"
        app_post  = "Outcome_Applications_Score"

        # Compute deltas if missing
        if "Proficiency_Improvement" not in df_exp.columns and {prof_post, prof_pre}.issubset(df_exp.columns):
            df_exp["Proficiency_Improvement"] = ensure_num(df_exp[prof_post]) - ensure_num(df_exp[prof_pre])
        if "Applications_Improvement" not in df_exp.columns and {app_post, app_pre}.issubset(df_exp.columns):
            df_exp["Applications_Improvement"] = ensure_num(df_exp[app_post]) - ensure_num(df_exp[app_pre])

        # Make numeric
        for c in [prof_pre, prof_post, app_pre, app_post, "Proficiency_Improvement", "Applications_Improvement"]:
            if c in df_exp.columns:
                df_exp[c] = ensure_num(df_exp[c])

        # Labels for recruiters
        LAB = {
            "prof_pre":  "Avg Skill Knowledge (Pre)",
            "prof_post": "Avg Skill Knowledge (Post)",
            "prof_delta":"Avg Δ Skill Knowledge",
            "app_pre":   "Avg Practical Application (Pre)",
            "app_post":  "Avg Practical Application (Post)",
            "app_delta": "Avg Δ Practical Application",
            "n_prof":    "Participants (Skill Knowledge)",
            "n_app":     "Participants (Application)",
        }

        # Safe mean helper
        def safe_mean(s: pd.Series):
            s = ensure_num(s)
            return float(s.mean()) if s.notna().sum() > 0 else None

        # Group summary
        grp = (df_exp
               .groupby("Group", as_index=False)
               .agg({
                   prof_pre:  ["count", safe_mean],
                   prof_post: safe_mean,
                   "Proficiency_Improvement": safe_mean,
                   app_pre:   ["count", safe_mean],
                   app_post:  safe_mean,
                   "Applications_Improvement": safe_mean
               }))

        # Flatten columns
        grp.columns = [
            "Group",
            LAB["n_prof"], LAB["prof_pre"], LAB["prof_post"], LAB["prof_delta"],
            LAB["n_app"],  LAB["app_pre"],  LAB["app_post"],  LAB["app_delta"],
        ]

        # Order groups & drop empty rows
        order = pd.CategoricalDtype(categories=["Control", "A", "B", "Other / Not Mapped"], ordered=True)
        grp["Group"] = grp["Group"].astype(order)
        grp = grp.sort_values("Group")
        grp["__has_data__"] = grp[[LAB["n_prof"], LAB["n_app"]]].fillna(0).sum(axis=1) > 0
        grp = grp[grp["__has_data__"]].drop(columns="__has_data__")

        # % Lift vs Control (only if Control exists with valid means)
        ctrl = grp[grp["Group"] == "Control"]
        def pct_lift(x, base):
            if x is None or base is None or pd.isna(x) or pd.isna(base) or base == 0:
                return None
            return (x - base) / abs(base) * 100.0
        if not ctrl.empty:
            base_prof = ctrl.iloc[0][LAB["prof_delta"]]
            base_app  = ctrl.iloc[0][LAB["app_delta"]]
            grp["Δ Skill Knowledge vs Control (%)"] = grp[LAB["prof_delta"]].apply(lambda v: round(pct_lift(v, base_prof), 1) if pct_lift(v, base_prof) is not None else None)
            grp["Δ Practical Application vs Control (%)"] = grp[LAB["app_delta"]].apply(lambda v: round(pct_lift(v, base_app), 1) if pct_lift(v, base_app) is not None else None)
        else:
            grp["Δ Skill Knowledge vs Control (%)"] = None
            grp["Δ Practical Application vs Control (%)"] = None

        # Clear, minimal test summary
        st.markdown(
            "- **Goal**: Measure whether curriculum variants improve **Skill Knowledge** and **Practical Application**.\n"
            "- **Design**: By city — Control (New York, Los Angeles), A (Miami, Houston), B (Detroit, Denver).\n"
            "- **Measures**: **Pre**, **Post**, and **Δ (Change = Post − Pre)** for each outcome."
        )

        # Results table
        st.dataframe(
            grp,
            use_container_width=True,
            hide_index=True
        )

        # Visuals (only if there’s at least one non-“Other” group)
        non_other = grp[grp["Group"].isin(["Control", "A", "B"])]
        if non_other.empty:
            st.info("Learners in this sample aren’t located in the six experimental cities. Showing table only.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                figA = px.bar(
                    non_other,
                    x="Group",
                    y=LAB["prof_delta"],
                    title="Average Change — Skill Knowledge",
                    labels={LAB["prof_delta"]: "Δ Skill Knowledge"},
                    height=420
                )
                figA.update_layout(margin=dict(l=14, r=14, t=64, b=90),
                                   title=dict(text="Average Change — Skill Knowledge", pad=dict(t=8, b=2)),
                                   legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))
                st.plotly_chart(figA, use_container_width=True, key="ab_prof")

            with c2:
                figB = px.bar(
                    non_other,
                    x="Group",
                    y=LAB["app_delta"],
                    title="Average Change — Practical Application",
                    labels={LAB["app_delta"]: "Δ Practical Application"},
                    height=420
                )
                figB.update_layout(margin=dict(l=14, r=14, t=64, b=90),
                                   title=dict(text="Average Change — Practical Application", pad=dict(t=8, b=2)),
                                   legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))
                st.plotly_chart(figB, use_container_width=True, key="ab_app")

        st.caption("Δ = Post − Pre. Groups are assigned by employee city (Control: New York & Los Angeles; A: Miami & Houston; B: Detroit & Denver).")

# ──────────────────────────────────────────────────────────────────────────────
# Footer (portfolio tag)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("© **Shan Ali Shah** — Workforce Analytics Portfolio")
