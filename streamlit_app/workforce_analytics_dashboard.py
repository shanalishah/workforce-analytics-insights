# streamlit_app/workforce_analytics_dashboard.py
from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ─────────────────────────────
# Page config + typography
# ─────────────────────────────
st.set_page_config(page_title="Workforce Analytics Dashboard", page_icon="📊", layout="wide")
st.title("Workforce Analytics Dashboard")
st.caption("Enrollments • Training Outcomes • PCA (Dimensionality Reduction) • K-Means Segmentation • A/B Testing • GenAI")

# Reduce title clipping/margins for all headings
st.markdown("""
<style>
.block-container { padding-top: 2.0rem !important; }
h1, h2, h3 { line-height: 1.25 !important; margin-top: 0.35rem !important; margin-bottom: 0.35rem !important; }
/* Extra breathing room around legends & axes */
.plotly .legend { margin-top: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# Paths & filenames
# ─────────────────────────────
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
SEARCH_DIRS = [
    ROOT / "data" / "analysis-outputs",
    ROOT / "data" / "processed",
    ROOT / "data",
    ROOT / "reports",
    ROOT,  # as last resort
]

FILES = {
    "enroll":        ["country_enrollment_summary.csv"],
    "ass_by_course": ["course_assessment_by_course.csv"],
    "ass_summed":    ["course_assessment_summed.csv"],
    "improve":       ["assessment_improvement.csv", "AssessmentImprovement.csv"],
    "city_clusters": ["city_cluster_distribution.csv", "City_Cluster_Distribution.csv"],
    "experiment":    ["experiment_curriculum_cleaned.csv"],
    # PCA: Excel workbook with Loadings + ExplainedVariance (+ optional CityClusterDistribution)
    "pca_workbook":  ["pca_components.xlsx"],
    # KMeans centers in PCA space
    "centers_xlsx":  ["pca_kmeans_results.xlsx"],  # sheet: KMeans_Cluster_Centers (Cluster, PC1, PC2, PC3, Percentage)
    # Survey question dictionary (Q1..Qn -> full text)
    "survey_qs":     ["survey_questions.xlsx", "survey_questions.csv"],  # in data/raw/
    # GenAI artifacts
    "genai_exec":    ["genai_executive_summary.pdf"],
    "genai_doc":     ["genai_custom_gpt_documentation.pdf"],
    "genai_flyers":  ["genai_flyers.pdf"],
}

# Optional friendly names for clusters (edit to brand the personas)
CLUSTER_LABELS = {
    # "Cluster 0": "Career-Oriented Implementers",
    # "Cluster 1": "Operational Specialists",
    # "Cluster 2": "Skill Growth Seekers",
    # "Cluster 3": "Foundation Builders",
}

# ─────────────────────────────
# Helpers
# ─────────────────────────────
def find_first(candidates):
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
        if len(line) + len(w) + 1 <= width:
            line = (line + " " + w).strip()
        else:
            out.append(line); line = w
    if line: out.append(line)
    return "<br>".join(out)

def cluster_sort_key(val: str) -> int:
    m = re.search(r"(\d+)", str(val))
    return int(m.group(1)) if m else 9_999

def tidy_legend_bottom(fig, title=""):
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=60, b=80),
        title=dict(text=title),
    )

# ─────────────────────────────
# Cached loaders
# ─────────────────────────────
@st.cache_data(show_spinner=False)
def load_df(kind):
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
def load_qmap():
    """Q1..Qn -> full question text mapping from data/raw/survey_questions.(xlsx|csv)."""
    p = find_first(FILES["survey_qs"])
    if not p:
        return {}
    try:
        df = read_csv_any(p) if p.suffix.lower()==".csv" else pd.read_excel(p)
        # Find columns
        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        qid = cols_lower.get("qid") or list(df.columns)[0]
        qtxt = next((c for c in df.columns if "question" in str(c).lower()), list(df.columns)[1])
        out = {}
        for _, r in df[[qid, qtxt]].dropna().iterrows():
            key = str(r[qid]).strip().upper()
            if re.match(r"^Q\d+$", key):
                out[key] = str(r[qtxt]).strip()
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

# ─────────────────────────────
# Load core data
# ─────────────────────────────
enr, _        = load_df("enroll")
ass_course, _ = load_df("ass_by_course")
ass_sum, _    = load_df("ass_summed")
improve, _    = load_df("improve")
city_pivot, _ = load_df("city_clusters")
experiment, _ = load_df("experiment")

QTEXT  = load_qmap()
PCAWB  = load_pca_workbook()
CENTERS, centers_path = load_kmeans_centers()

# ─────────────────────────────
# KPI Row
# ─────────────────────────────
kpi = {}
if enr is not None and not enr.empty:
    c_country = "Country_Regional_Center" if "Country_Regional_Center" in enr.columns else enr.columns[0]
    c_enroll  = "Total_Enrollments"       if "Total_Enrollments"       in enr.columns else enr.columns[1]
    enr[c_enroll] = ensure_num(enr[c_enroll])
    kpi["Total Enrollments"]     = f"{int(enr[c_enroll].sum(skipna=True)):,}"
    kpi["Countries Represented"] = enr[c_country].astype(str).nunique()

if ass_course is not None and "Course_Title" in ass_course.columns:
    kpi["Courses Analyzed"] = ass_course["Course_Title"].astype(str).nunique()

if isinstance(PCAWB.get("explained"), pd.DataFrame) and not PCAWB["explained"].empty:
    total_var = float(ensure_num(PCAWB["explained"]["Explained Variance (%)"]).sum())
    total_var = max(0.0, min(total_var, 100.0))
    kpi["Variance Explained (PC1–PC3)"] = f"{total_var:.1f}%"

if kpi:
    cols = st.columns(min(4, len(kpi)))
    for (label, value), col in zip(kpi.items(), cols):
        col.metric(label, value)

st.markdown("---")

# ─────────────────────────────
# Tabs (now 5 tabs: + A/B, + GenAI)
# ─────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 Enrollments",
    "🎯 Training Outcomes",
    "🧩 PCA & Segmentation",
    "📊 A/B Testing",
    "🧠 GenAI—Targeted Messaging",
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

        # Default = ALL (you asked to remove the "(default: All)" text)
        all_countries = sorted(view["Country"])
        picks = st.multiselect("Countries", options=all_countries, default=all_countries, key="enr_picks")
        order = st.radio("Sort by", ["Enrollments (desc)", "Country (A–Z)"], horizontal=True, key="enr_sort")

        if picks: view = view[view["Country"].isin(picks)]
        view = view.sort_values("Enrollments", ascending=False) if order.startswith("Enrollments") else view.sort_values("Country")

        if view.empty:
            st.info("No countries selected.")
        else:
            fig = px.bar(view, x="Country", y="Enrollments", title="Enrollments for Selected Countries", height=420)
            fig.update_layout(yaxis_title_standoff=12, title={'pad': {'t': 14}})
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
                fig = px.bar(by_mode, x="Delivery Mode", y=metric_col, title=f"{metric_label} by Delivery Mode", height=400)
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=60), yaxis_title_standoff=14, title={'pad': {'t': 14}})
                tidy_legend_bottom(fig, "")
                st.plotly_chart(fig, use_container_width=True, key="outcomes_mode")

            with g2:
                top = (df_plot.groupby("Course_Title", as_index=False)[metric_col]
                       .mean()
                       .sort_values(metric_col, ascending=False)
                       .head(15))
                top["_Course_Label"] = top["Course_Title"].apply(lambda s: wrap_text(s, 28))
                fig2 = px.bar(top, y="_Course_Label", x=metric_col, orientation="h",
                              title=f"{metric_label} — Top 15 Courses", height=520)
                fig2.update_traces(text=top[metric_col].round(2), textposition="outside", cliponaxis=False)
                fig2.update_layout(margin=dict(l=170, r=30, t=60, b=10), yaxis={"categoryorder": "total ascending"}, title={'pad': {'t': 14}})
                st.plotly_chart(fig2, use_container_width=True, key="outcomes_top")

# ── PCA & Segmentation
with tab3:
    st.subheader("PCA Summary & K-Means Segmentation")

    # Explained variance
    st.markdown("#### PCA — Explained Variance")
    ev = PCAWB.get("explained")
    if isinstance(ev, pd.DataFrame) and not ev.empty:
        fig_ev = px.bar(ev, x="Principal Component", y="Explained Variance (%)",
                        title="Explained Variance by Component", height=320)
        fig_ev.update_layout(margin=dict(l=10, r=10, t=60, b=60), title={'pad': {'t': 14}})
        st.plotly_chart(fig_ev, use_container_width=True, key="pca_ev")
    else:
        st.info("Add `ExplainedVariance` sheet to `pca_components.xlsx` with columns: Principal Component, Explained Variance (%).")

    # Top contributing survey questions (loadings)
    st.markdown("#### PCA — Top Contributing Survey Questions")
    ld = PCAWB.get("loadings")
    if isinstance(ld, pd.DataFrame) and not ld.empty:
        if "Response" not in ld.columns:
            ld = ld.rename(columns={ld.columns[0]: "Response"})
        # Component labels (prefer EV names if present; else derive PC1..)
        labels = ev["Principal Component"].astype(str).tolist() if isinstance(ev, pd.DataFrame) and not ev.empty else [f"PC{i+1}" for i in range(len(ld))]

        pc_pick = st.selectbox("Component", labels, index=0, key="pc_pick",
                               help="Shows the strongest contributing survey questions for the selected component.")
        # Locate the matching row
        row_idx = labels.index(pc_pick) if pc_pick in labels else 0
        row = ld.iloc[row_idx]

        # Columns that look like Q1..Qn
        qcols = [c for c in ld.columns if re.match(r"^Q\d+$", str(c), re.I)]
        if not qcols:
            st.info("No Q1..Qn columns found in the Loadings sheet.")
        else:
            contrib = sorted(((q, float(row[q])) for q in qcols), key=lambda x: abs(x[1]), reverse=True)[:10]
            # Map QIDs -> question text for y-axis
            def q_label(qid: str) -> str:
                return QTEXT.get(qid.upper(), qid)

            disp = pd.DataFrame({
                "Survey Question": [q_label(q) for q, _ in contrib],
                "Loading (magnitude)": [abs(v) for _, v in contrib],
            })
            # Keep the signed loading for tooltip
            disp["Loading (signed)"] = [v for _, v in contrib]

            fig_load = px.bar(
                disp,
                y="Survey Question",
                x="Loading (magnitude)",
                orientation="h",
                title=f"Top Questions Influencing {pc_pick}",
                hover_data={"Loading (signed)": True, "Loading (magnitude)": False},
                height=520,
            )
            fig_load.update_layout(margin=dict(l=220, r=20, t=60, b=20), title={'pad': {'t': 14}})
            st.plotly_chart(fig_load, use_container_width=True, key="pca_top_questions")
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
            fig_c.update_layout(
                margin=dict(l=60, r=20, t=60, b=120),
                xaxis_title_standoff=18,
                title={'pad': {'t': 14}}
            )
            tidy_legend_bottom(fig_c, "")
            st.plotly_chart(fig_c, use_container_width=True, key="city_pct")
        elif "Employees" in city_df.columns:
            fig_c = px.bar(city_df, x="City", y="Employees", color="Cluster",
                           title="Segment Counts by City", height=420)
            fig_c.update_layout(
                margin=dict(l=60, r=20, t=60, b=120),
                xaxis_title_standoff=18,
                title={'pad': {'t': 14}}
            )
            tidy_legend_bottom(fig_c, "")
            st.plotly_chart(fig_c, use_container_width=True, key="city_cnt")

    # K-Means centers: table + 2D pairs + 3D
    st.markdown("#### K-Means Cluster Centers in PCA Space")
    if CENTERS is None or CENTERS.empty:
        st.warning("Add `pca_kmeans_results.xlsx` with sheet `KMeans_Cluster_Centers` (columns: Cluster, PC1, PC2, PC3, Percentage).")
    else:
        cols_to_show = ["Cluster"] + [c for c in ("PC1","PC2","PC3","Percentage") if c in CENTERS.columns]
        st.dataframe(CENTERS[cols_to_show], use_container_width=True, hide_index=True)

        have_pc12 = {"PC1","PC2"}.issubset(CENTERS.columns)
        have_pc13 = {"PC1","PC3"}.issubset(CENTERS.columns)
        have_pc23 = {"PC2","PC3"}.issubset(CENTERS.columns)
        have_pc123 = {"PC1","PC2","PC3"}.issubset(CENTERS.columns)

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
                fig.update_layout(
                    margin=dict(l=40, r=20, t=60, b=120),
                    height=380,
                    xaxis_title_standoff=18,
                    yaxis_title_standoff=14,
                    title={'pad': {'t': 14}}
                )
                tidy_legend_bottom(fig, "")
                col.plotly_chart(fig, use_container_width=True, key=f"kmeans_2d_{k}")

        if have_pc123:
            fig3d = px.scatter_3d(CENTERS, x="PC1", y="PC2", z="PC3", color="Cluster", text="Cluster",
                                  title="PC1 • PC2 • PC3 (Cluster Centers — 3D)")
            fig3d.update_traces(marker=dict(size=6), textposition="top center")
            fig3d.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=520, legend=dict(orientation="h", y=-0.1), title={'pad': {'t': 14}})
            st.plotly_chart(fig3d, use_container_width=True, key="kmeans_3d")

# ── A/B Testing
with tab4:
    st.subheader("A/B Testing — Curriculum Delivery Comparison")

    st.markdown(
        "This view compares **In-Person (Control)** vs **Virtual (Treatment)** delivery for the curriculum experiment. "
        "Metrics show average change from intake (pre) to outcome (post)."
    )

    if experiment is None or experiment.empty:
        st.info("Add `experiment_curriculum_cleaned.csv` to show A/B results.")
    else:
        df_exp = experiment.copy()

        # Bring in course titles (and infer delivery mode) via assessment_improvement mapping if available
        # assessment_improvement has Course & Course_Title we can merge on (Course numeric/string)
        course_map = None
        if improve is not None and not improve.empty:
            tmp = improve[["Course","Course_Title"]].dropna().copy()
            tmp["Course"] = tmp["Course"].astype(str).str.strip()
            course_map = tmp.drop_duplicates("Course")
        if course_map is not None:
            df_exp["Course"] = df_exp["Course"].astype(str).str.strip()
            df_exp = df_exp.merge(course_map, on="Course", how="left")
        else:
            # Fallback: create a generic title
            df_exp["Course_Title"] = df_exp.get("Course", "Course")

        # Infer Group using delivery keywords in title (Virtual vs In-Person)
        def infer_group(title: str) -> str:
            t = str(title or "").lower()
            if "virtual" in t:
                return "Treatment (Virtual)"
            return "Control (In-Person)"

        df_exp["Group"] = df_exp["Course_Title"].apply(infer_group)

        # Numeric metrics
        df_exp["Proficiency_Improvement"]  = ensure_num(df_exp.get("Proficiency_Improvement"))
        df_exp["Applications_Improvement"] = ensure_num(df_exp.get("Applications_Improvement"))

        metric_options = {
            "Proficiency — Change (post − pre)":  "Proficiency_Improvement",
            "Application — Change (post − pre)":  "Applications_Improvement",
        }
        metric_label = st.selectbox("Metric", list(metric_options.keys()), index=0, key="ab_metric")
        metric_col   = metric_options[metric_label]

        # Optional course filter (now that titles are present)
        courses_avail = sorted(df_exp["Course_Title"].dropna().unique())
        course_sel = st.multiselect("Courses (optional)", options=courses_avail, default=[], key="ab_courses")
        df_view = df_exp if not course_sel else df_exp[df_exp["Course_Title"].isin(course_sel)]
        df_view = df_view.dropna(subset=[metric_col])

        if df_view.empty:
            st.info("No rows available for the selected filters/metric.")
        else:
            # Summary stats per group
            g = df_view.groupby("Group")[metric_col].agg(["count","mean","std"]).reset_index()
            # Ensure both groups appear even if one is absent in filtered data
            if "Control (In-Person)" not in g["Group"].values:
                g = pd.concat([g, pd.DataFrame([{"Group":"Control (In-Person)","count":0,"mean":np.nan,"std":np.nan}])], ignore_index=True)
            if "Treatment (Virtual)" not in g["Group"].values:
                g = pd.concat([g, pd.DataFrame([{"Group":"Treatment (Virtual)","count":0,"mean":np.nan,"std":np.nan}])], ignore_index=True)
            g = g.set_index("Group").loc[["Control (In-Person)","Treatment (Virtual)"]].reset_index()

            # Compute lift (%) if control mean exists
            ctrl_mean = g.loc[g["Group"]=="Control (In-Person)","mean"].values[0]
            trt_mean  = g.loc[g["Group"]=="Treatment (Virtual)","mean"].values[0]
            if pd.notna(ctrl_mean) and ctrl_mean != 0 and pd.notna(trt_mean):
                lift_pct = (trt_mean - ctrl_mean) / abs(ctrl_mean) * 100.0
            else:
                lift_pct = np.nan

            # Approximate 95% CI (normal) for bars
            def ci95(row):
                n = row["count"]
                s = row["std"]
                if pd.isna(n) or pd.isna(s) or n <= 1:
                    return np.nan
                return 1.96 * (s / math.sqrt(n))

            g["ci95"] = g.apply(ci95, axis=1)

            c1, c2 = st.columns([1.05, 1])
            with c1:
                fig_ab = px.bar(
                    g, x="Group", y="mean", error_y="ci95",
                    title=f"{metric_label} — Control vs Treatment",
                    labels={"mean":"Average Change", "Group":"Group"},
                    height=420
                )
                fig_ab.update_layout(margin=dict(l=20, r=20, t=60, b=60), title={'pad': {'t': 14}})
                tidy_legend_bottom(fig_ab, "")
                st.plotly_chart(fig_ab, use_container_width=True, key="ab_bar")

            with c2:
                # Clean presentation table
                table = g.rename(columns={
                    "count":"Participants (n)",
                    "mean":"Average Change",
                    "std":"Std. Dev.",
                })[["Group","Participants (n)","Average Change","Std. Dev."]]
                st.markdown("##### Summary")
                st.dataframe(table, use_container_width=True, hide_index=True)

                # Lift statement
                if pd.notna(lift_pct):
                    st.success(f"Estimated Lift vs Control: **{lift_pct:+.1f}%**")
                else:
                    st.info("Lift could not be computed for the current selection.")

            with st.expander("Methodology (A/B) — What you're seeing", expanded=False):
                st.markdown(
                    "- **Groups**: In-Person is treated as **Control**, Virtual as **Treatment** (inferred from course titles).\n"
                    "- **Metric**: Change = Outcome − Intake (per learner), aggregated by group.\n"
                    "- **Error bars**: Approximate 95% confidence intervals (normal) for the mean.\n"
                    "- **Lift**: Relative difference of treatment vs control on the selected metric."
                )

# ── GenAI
with tab5:
    st.subheader("Generative AI for Targeted EDP Messaging")

    st.markdown(
        "A **Custom GPT** was configured to generate segment-specific EDP flyers aligned with PCA themes "
        "(Skill Development, Operational Focus, Career Advancement) and K-Means clusters."
    )
    NLS_GPT_URL = "https://chat.openai.com/"  # replace with your public GPT link if you wish
    st.link_button("Open NLS Assistant (Custom GPT)", NLS_GPT_URL, help="Launch the configured Custom GPT.")

    st.markdown("### Deliverables")
    col_a, col_b, col_c = st.columns(3)

    def _dl_button(col, key, label, file_key):
        p = find_first(FILES[file_key])
        if p and p.exists():
            with open(p, "rb") as f:
                col.download_button(label=label, data=f.read(), file_name=p.name, mime="application/pdf", key=key)
        else:
            col.info(f"Add `{FILES[file_key][0]}` to your repo to enable download.")

    _dl_button(col_a, "dl_flyers", "⬇️ Flyers (All Segments, PDF)", "genai_flyers")
    _dl_button(col_b, "dl_exec",   "⬇️ Executive Summary (PDF)",     "genai_exec")
    _dl_button(col_c, "dl_doc",    "⬇️ Custom GPT Docs (PDF)",       "genai_doc")

    st.markdown("### Method (brief)")
    st.markdown(
        "- Provided the GPT with segmentation context and artifacts (memos, survey, PCA/K-Means results).\n"
        "- Prompted with **persona prompts** per segment to draft concise, professional copy.\n"
        "- Iterated on tone and content; packaged outputs (flyers + executive summary)."
    )

# Footer (tiny)
st.markdown("---")
st.caption("© Shan Ali Shah — Workforce Analytics Portfolio")
