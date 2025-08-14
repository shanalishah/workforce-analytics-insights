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
/* give charts breathing room at bottom so legends don't overlap axes */
.plot-container, .echart-container { padding-bottom: 8px !important; }
/* wrap long select options */
.stMultiSelect [data-baseweb="tag"] { white-space: normal !important; }
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
    ROOT / "data" / "raw",   # <- survey_questions lives here
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

# A/B group mapping (by City startswith)
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
        legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="center", x=0.5, title=title_text),
        margin=dict(l=16, r=16, t=64, b=110),
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
    """Map Q1..Q12 → full question text from survey_questions.* (robust to spacing/case)."""
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
            key = re.sub(r"\s+", "", str(r[qid_col]).strip().upper())  # "Q 1" -> "Q1"
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
        # Normalize Q columns: strip, drop spaces ("Q 1" -> "Q1")
        ren = {}
        for c in ld.columns:
            cs = re.sub(r"\s+", "", str(c).strip())
            if re.match(r"^Q\d+$", cs, re.I):
                ren[c] = cs.upper()
        if ren:
            ld = ld.rename(columns=ren)
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
# KPI Row (use Explained Variance label = PCA)
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
        fig_ev.update_layout(margin=dict(l=16, r=16, t=64, b=20),
                             title=dict(text="Explained Variance by Component", pad=dict(t=8, b=2)))
        st.plotly_chart(fig_ev, use_container_width=True, key="pca_ev")
    else:
        st.info("Add `ExplainedVariance` sheet to `pca_components.xlsx` with columns: Principal Component, Explained Variance (%).")

    # Top contributing survey questions (loadings) with QID → full text mapping
    st.markdown("#### PCA — Top Contributing Survey Questions")
    ld = PCAWB.get("loadings")
    if isinstance(ld, pd.DataFrame) and not ld.empty:
        if "Response" not in ld.columns:
            ld = ld.rename(columns={ld.columns[0]: "Response"})

        # Component labels: prefer EV names (e.g., "PC1 (Skill Development)")
        labels = (ev["Principal Component"].astype(str).tolist()
                  if isinstance(ev, pd.DataFrame) and not ev.empty
                  else [str(x) for x in ld["Response"].astype(str).tolist()])

        pc_pick = st.selectbox(
            "Component",
            labels,
            index=0,
            help="Shows strongest contributing survey questions for the selected component.",
            key="pc_pick_loadings"
        )

        # Find matching row by label text; fallback by numeric PC order
        try:
            row_idx = labels.index(pc_pick)
        except ValueError:
            # attempt match by number
            num = pc_order_val(pc_pick)
            row_idx = min(num-1, len(ld)-1) if num >= 1 else 0

        row = ld.iloc[row_idx:row_idx+1].copy()

        # Robust Q-column detection: "Q1", "Q 1", " q1 "
        qcols = []
        for c in ld.columns:
            cs = re.sub(r"\s+", "", str(c).strip()).upper()
            if re.match(r"^Q\d+$", cs):
                qcols.append(c)
        if not qcols:
            st.info("No Q1..Qn columns found in the Loadings sheet.")
        else:
            # series of loadings for that PC
            s = row[qcols].T.iloc[:, 0]
            # map index to full question text when available
            mapped_index = []
            for raw in s.index:
                key = re.sub(r"\s+", "", str(raw).strip()).upper()  # normalize
                mapped_index.append(QTEXT.get(key, raw))
            s.index = mapped_index

            # Top 10 by absolute loading, keep sign for readability
            abs_top = s.abs().sort_values(ascending=False).head(10)
            display_vals = s.loc[abs_top.index]

            plot_df = pd.DataFrame({
                "Question": abs_top.index,
                "Influence (Loading)": display_vals.values
            })

            fig_ld = px.bar(plot_df, x="Influence (Loading)", y="Question", orientation="h",
                            title=f"Top Questions Influencing {pc_pick}", height=540)
            fig_ld.update_layout(margin=dict(l=80, r=18, t=64, b=20),
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
                           title="Segment Share by City", height=430)
            fig_c.update_layout(margin=dict(l=16, r=16, t=64, b=120),
                                title=dict(text="Segment Share by City", pad=dict(t=8, b=2)))
            tidy_legend_bottom(fig_c, "Cluster")
            st.plotly_chart(fig_c, use_container_width=True, key="city_pct")
        elif "Employees" in city_df.columns:
            fig_c = px.bar(city_df, x="City", y="Employees", color="Cluster",
                           title="Segment Counts by City", height=430)
            fig_c.update_layout(margin=dict(l=16, r=16, t=64, b=120),
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
                fig.update_layout(margin=dict(l=18, r=18, t=72, b=135),
                                  height=440,
                                  xaxis_title_standoff=22,
                                  yaxis_title_standoff=22,
                                  title=dict(text=fig.layout.title.text, pad=dict(t=8, b=2)))
                tidy_legend_bottom(fig, "Cluster")
                col.plotly_chart(fig, use_container_width=True, key=f"kmeans_2d_{k}")

# ── A/B Testing
with tab4:
    st.subheader("A/B Testing — Curriculum Experiment")
    st.caption("Comparing **Control** (current program) vs **Curriculum A** vs **Curriculum B** using changes in Proficiency and Application scores.")

    if experiment is None or experiment.empty:
        st.info("Add `experiment_curriculum_cleaned.csv` to show A/B results.")
    else:
        df_exp = experiment.copy()

        # Coerce Employee_ID types in both frames to improve join match (reduces Unassigned)
        for frame in (df_exp, survey_loc):
            if frame is not None and "Employee_ID" in frame.columns:
                # Keep a copy in case of leading zeros (rare here)
                frame["Employee_ID"] = pd.to_numeric(frame["Employee_ID"], errors="coerce").astype("Int64")

        # Attach City by Employee_ID
        if survey_loc is not None and not survey_loc.empty and "Employee_ID" in survey_loc.columns:
            df_exp = df_exp.merge(
                survey_loc[["Employee_ID", "City_y"]].rename(columns={"City_y": "City"}),
                on="Employee_ID",
                how="left"
            )
        else:
            df_exp["City"] = pd.NA

        # Map City → Group (Control/A/B)
        def city_to_group(city) -> str:
            if pd.isna(city):
                return "Unassigned"
            c = str(city)
            for key, grp in AB_GROUPS.items():
                if c.startswith(key):
                    return grp
            return "Unassigned"

        df_exp["Group"] = df_exp["City"].apply(city_to_group)

        # Ensure numeric deltas exist
        prof_col = "Proficiency_Improvement" if "Proficiency_Improvement" in df_exp.columns else "Proficiency_Delta"
        app_col  = "Applications_Improvement" if "Applications_Improvement" in df_exp.columns else "Applications_Delta"
        if prof_col not in df_exp.columns and {"Outcome_Proficiency_Score","Intake_Proficiency_Score"}.issubset(df_exp.columns):
            df_exp["Proficiency_Improvement"] = ensure_num(df_exp["Outcome_Proficiency_Score"]) - ensure_num(df_exp["Intake_Proficiency_Score"])
            prof_col = "Proficiency_Improvement"
        if app_col not in df_exp.columns and {"Outcome_Applications_Score","Intake_Applications_Score"}.issubset(df_exp.columns):
            df_exp["Applications_Improvement"] = ensure_num(df_exp["Outcome_Applications_Score"]) - ensure_num(df_exp["Intake_Applications_Score"])
            app_col = "Applications_Improvement"

        df_exp[prof_col] = ensure_num(df_exp[prof_col])
        df_exp[app_col]  = ensure_num(df_exp[app_col])

        # Summary by group
        order = pd.CategoricalDtype(categories=["Control", "A", "B", "Unassigned"], ordered=True)
        gmean = (df_exp.groupby("Group", as_index=False)[[prof_col, app_col]].mean().round(3))
        gmean["Group"] = gmean["Group"].astype(order)

        gcount = df_exp.groupby("Group", as_index=False)[[prof_col, app_col]].count()
        gcount["Group"] = gcount["Group"].astype(order)

        summary = gmean.merge(gcount, on="Group", suffixes=("_Mean", "_N")).sort_values("Group")
        summary = summary.rename(columns={
            f"{prof_col}_Mean": "Avg Δ Proficiency",
            f"{app_col}_Mean":  "Avg Δ Application",
            f"{prof_col}_N":    "N (Proficiency)",
            f"{app_col}_N":     "N (Application)",
        })

        st.dataframe(summary, use_container_width=True, hide_index=True)

        # Uplift vs Control bars (if Control present)
        ctrl_prof = gmean.loc[gmean["Group"]=="Control", prof_col].squeeze() if "Control" in gmean["Group"].values else None
        ctrl_app  = gmean.loc[gmean["Group"]=="Control", app_col].squeeze() if "Control" in gmean["Group"].values else None

        if pd.notna(ctrl_prof) and pd.notna(ctrl_app):
            uplift = gmean.copy()
            uplift["Δ Proficiency vs Control"] = uplift[prof_col] - float(ctrl_prof)
            uplift["Δ Application vs Control"] = uplift[app_col]  - float(ctrl_app)

            c1, c2 = st.columns(2)
            with c1:
                figU1 = px.bar(uplift.sort_values("Group"), x="Group", y="Δ Proficiency vs Control",
                               title="Uplift vs Control — Proficiency", height=420,
                               labels={"Δ Proficiency vs Control":"Uplift (Δ Proficiency)"})
                figU1.update_layout(margin=dict(l=16, r=16, t=64, b=90),
                                    title=dict(text="Uplift vs Control — Proficiency", pad=dict(t=8, b=2)))
                tidy_legend_bottom(figU1, "")
                st.plotly_chart(figU1, use_container_width=True, key="uplift_prof")

            with c2:
                figU2 = px.bar(uplift.sort_values("Group"), x="Group", y="Δ Application vs Control",
                               title="Uplift vs Control — Application", height=420,
                               labels={"Δ Application vs Control":"Uplift (Δ Application)"})
                figU2.update_layout(margin=dict(l=16, r=16, t=64, b=90),
                                    title=dict(text="Uplift vs Control — Application", pad=dict(t=8, b=2)))
                tidy_legend_bottom(figU2, "")
                st.plotly_chart(figU2, use_container_width=True, key="uplift_app")

        # Concise, recruiter-friendly note
        assigned = int((df_exp["Group"] != "Unassigned").sum())
        unassigned = int((df_exp["Group"] == "Unassigned").sum())
        st.caption(f"Notes: Groups assigned by employee city (Control: New York & Los Angeles; A: Miami & Houston; B: Detroit & Denver). "
                   f"{assigned} learners assigned, {unassigned} unassigned (no matching city). Δ = Outcome − Intake.")

# ──────────────────────────────────────────────────────────────────────────────
# Footer (portfolio tag)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("© **Shan Ali Shah** — Workforce Analytics Portfolio")
