from pathlib import Path
import re
import pandas as pd
import plotly.express as px
import streamlit as st

# ─────────────────────────────
# Page config + typography
# ─────────────────────────────
st.set_page_config(page_title="Workforce Analytics Dashboard", page_icon="📊", layout="wide")
st.title("Workforce Analytics Dashboard")
st.caption("Enrollments • Training Outcomes • PCA (Dimensionality Reduction) • K-Means Segmentation")

# Reduce title clipping/margins for all headings
st.markdown("""
<style>
.block-container { padding-top: 2.0rem !important; }
h1, h2, h3 { line-height: 1.25 !important; margin-top: 0.35rem !important; margin-bottom: 0.35rem !important; }
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
    ROOT / "data" / "raw",           # make sure we also look in data/raw (for survey_questions.xlsx)
]

FILES = {
    "enroll":        ["country_enrollment_summary.csv"],
    "ass_by_course": ["course_assessment_by_course.csv"],
    "ass_summed":    ["course_assessment_summed.csv"],
    "improve":       ["assessment_improvement.csv", "AssessmentImprovement.csv"],
    "city_clusters": ["city_cluster_distribution.csv", "City_Cluster_Distribution.csv"],
    "experiment":    ["experiment_curriculum_cleaned.csv"],
    "pca_workbook":  ["pca_components.xlsx"],           # sheets: Loadings, ExplainedVariance, (optional) CityClusterDistribution
    "centers_xlsx":  ["pca_kmeans_results.xlsx"],       # sheet: KMeans_Cluster_Centers (Cluster, PC1, PC2, PC3, Percentage)
    "survey_qs":     ["survey_questions.xlsx", "survey_questions.csv"],  # QID, Question Text
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
    # Last-chance fallback
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
    # "Cluster 2" -> 2 ; unknown -> 9999
    m = re.search(r"(\d+)", str(val))
    return int(m.group(1)) if m else 9_999

def tidy_legend_bottom(fig, title="Cluster"):
    # Push legend down and add breathing room so axis labels don't collide
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5, title_text=title),
        margin=dict(l=10, r=10, t=60, b=120),  # generous bottom margin to avoid overlap
    )
    # Extra axis label breathing room
    fig.update_layout(xaxis_title_standoff=24, yaxis_title_standoff=24)

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
    """Return dict like {'Q1': 'Full question text', ...}"""
    p = find_first(FILES["survey_qs"])
    if not p:
        return {}
    try:
        df = read_csv_any(p) if p.suffix.lower()==".csv" else pd.read_excel(p)
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
            # Normalize percentages (handle "31.90%", "31.9", 0.319)
            ev["Explained Variance (%)"] = (
                ev["Explained Variance (%)"].astype(str).str.replace("%","",regex=False).str.replace(",","",regex=False).str.strip()
            )
            ev["Explained Variance (%)"] = pd.to_numeric(ev["Explained Variance (%)"], errors="coerce")
            # If values are decimals (<=1.5), convert to %
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
    # Ensure required columns
    if "Cluster" not in df.columns:
        df.insert(0, "Cluster", [f"Cluster {i}" for i in range(len(df))])
    else:
        df["Cluster"] = df["Cluster"].astype(str)
        df.loc[~df["Cluster"].str.contains("Cluster", case=False), "Cluster"] = "Cluster " + df["Cluster"]

    for col in ("PC1","PC2","PC3","Percentage"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Optional friendly labels
    if CLUSTER_LABELS:
        df["Cluster"] = df["Cluster"].map(lambda x: CLUSTER_LABELS.get(x, x))

    # Sort by numeric cluster id
    df = df.sort_values("Cluster", key=lambda s: s.map(cluster_sort_key)).reset_index(drop=True)
    return df, p

# ─────────────────────────────
# Load core data
# ─────────────────────────────
enr, _        = load_df("enroll")
ass_course, _ = load_df("ass_by_course")
ass_sum, _    = load_df("ass_summed")   # not displayed, but kept for future use
improve, _    = load_df("improve")      # not displayed, but kept for future use
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
# Tabs
# ─────────────────────────────
tab1, tab2, tab3 = st.tabs(["📍 Enrollments", "🎯 Training Outcomes", "🧩 PCA & Segmentation"])

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

        # Default = all countries (you asked to show all by default)
        default_all = sorted(view["Country"].unique().tolist())
        picks = st.multiselect("Countries (default: All)", options=default_all, default=default_all, key="enr_picks")
        order = st.radio("Sort by", ["Enrollments (desc)", "Country (A–Z)"], horizontal=True, key="enr_sort")

        if picks: view = view[view["Country"].isin(picks)]
        view = view.sort_values("Enrollments", ascending=False) if order.startswith("Enrollments") else view.sort_values("Country")

        if view.empty:
            st.info("No countries selected.")
        else:
            fig = px.bar(view, x="Country", y="Enrollments", title="Enrollments for Selected Countries", height=420)
            fig.update_layout(yaxis_title_standoff=12)     # padding only; no title_pad to avoid Plotly errors
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
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=20), yaxis_title_standoff=14)
                st.plotly_chart(fig, use_container_width=True, key="outcomes_mode")

            with g2:
                top = (df_plot.groupby("Course_Title", as_index=False)[metric_col]
                       .mean()
                       .sort_values(metric_col, ascending=False)
                       .head(15))
                top["_Course_Label"] = top["Course_Title"].apply(lambda s: wrap_text(s, 28))
                fig2 = px.bar(top, y="_Course_Label", x=metric_col, orientation="h",
                              title=f"{metric_label} — Top 15 Courses", height=520,
                              labels={metric_col: metric_label, "_Course_Label": "Course"})
                fig2.update_traces(text=top[metric_col].round(2), textposition="outside", cliponaxis=False)
                fig2.update_layout(margin=dict(l=180, r=30, t=60, b=20), yaxis={"categoryorder": "total ascending"})
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
        fig_ev.update_layout(margin=dict(l=10, r=10, t=60, b=20))
        st.plotly_chart(fig_ev, use_container_width=True, key="pca_ev")
    else:
        st.info("Add `ExplainedVariance` sheet to `pca_components.xlsx` with columns: Principal Component, Explained Variance (%).")

    # Top contributing survey questions (loadings)
    st.markdown("#### PCA — Top Contributing Survey Questions")
    ld = PCAWB.get("loadings")
    if isinstance(ld, pd.DataFrame) and not ld.empty:
        if "Response" not in ld.columns:
            ld = ld.rename(columns={ld.columns[0]: "Response"})

        # Available PC labels, prefer the ones from EV sheet if present
        labels = ev["Principal Component"].astype(str).tolist() if isinstance(ev, pd.DataFrame) and not ev.empty else [f"PC{i+1}" for i in range(len(ld))]

        pc_pick = st.selectbox("Component", labels, index=0, key="pc_pick",
                               help="Shows strongest contributing survey questions for the selected component.")

        # Pick the row based on the label position
        row = ld.iloc[labels.index(pc_pick)] if labels and len(ld) >= len(labels) else None
        qcols = [c for c in ld.columns if re.match(r"^Q\d+$", str(c), re.I)]

        if row is None or not qcols:
            st.info("No data for the selected component.")
        else:
            # Compute absolute loadings and map QIDs to full text from QTEXT
            contrib = sorted(((QTEXT.get(q.upper(), q), abs(float(row[q]))) for q in qcols), key=lambda x: x[1], reverse=True)[:8]
            plot_df = pd.DataFrame(contrib, columns=["Survey Question", "Influence (absolute loading)"])

            fig_ld = px.bar(
                plot_df.sort_values("Influence (absolute loading)"),
                x="Influence (absolute loading)",
                y="Survey Question",
                orientation="h",
                title=f"Top Questions Influencing {pc_pick}",
                height=520,
            )
            # Wider left margin so long questions don't get clipped
            fig_ld.update_layout(margin=dict(l=220, r=20, t=60, b=20), xaxis_title_standoff=18, yaxis_title=None)
            st.plotly_chart(fig_ld, use_container_width=True, key="pca_loadings")
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
            tidy_legend_bottom(fig_c, "Cluster")
            st.plotly_chart(fig_c, use_container_width=True, key="city_pct")
        elif "Employees" in city_df.columns:
            fig_c = px.bar(city_df, x="City", y="Employees", color="Cluster",
                           title="Segment Counts by City", height=420)
            tidy_legend_bottom(fig_c, "Cluster")
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
            tidy_legend_bottom(f, "Cluster")
            figs.append(("pc12", f))
        if have_pc13:
            f = px.scatter(CENTERS, x="PC1", y="PC3", color="Cluster", text="Cluster", title="PC1 vs PC3 (Cluster Centers)")
            tidy_legend_bottom(f, "Cluster")
            figs.append(("pc13", f))
        if have_pc23:
            f = px.scatter(CENTERS, x="PC2", y="PC3", color="Cluster", text="Cluster", title="PC2 vs PC3 (Cluster Centers)")
            tidy_legend_bottom(f, "Cluster")
            figs.append(("pc23", f))

        if figs:
            cols = st.columns(len(figs))
            for (k, fig), col in zip(figs, cols):
                fig.update_traces(marker=dict(size=12, opacity=0.9), textposition="top center")
                col.plotly_chart(fig, use_container_width=True, key=f"kmeans_2d_{k}")

        if have_pc123:
            fig3d = px.scatter_3d(CENTERS, x="PC1", y="PC2", z="PC3", color="Cluster", text="Cluster",
                                  title="PC1 • PC2 • PC3 (Cluster Centers — 3D)")
            fig3d.update_traces(marker=dict(size=6), textposition="top center")
            fig3d.update_layout(margin=dict(l=10, r=10, t=50, b=20),
                                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
            st.plotly_chart(fig3d, use_container_width=True, key="kmeans_3d")

# Footer
st.markdown("---")
st.caption("© Shan Ali Shah — Workforce Analytics Portfolio")
