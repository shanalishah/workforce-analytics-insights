# streamlit_app/app.py
from pathlib import Path
import re
import io
import pandas as pd
import plotly.express as px
import streamlit as st

# ───────────────────────────
# Page config
# ───────────────────────────
st.set_page_config(page_title="Workforce Analytics Dashboard", page_icon="📊", layout="wide")
st.title("Workforce Analytics Dashboard")
st.caption("Enrollments · Training Outcomes · PCA (Dimensionality Reduction) · K-Means Clustering (Segmentation)")

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
SEARCH_DIRS = [
    ROOT / "data" / "analysis-outputs",
    ROOT / "data" / "processed",
    ROOT / "data",
    ROOT / "data" / "raw",
]

FILENAMES = {
    "enroll": ["country_enrollment_summary.csv", "Country-wise_Enrollment_Summary.csv"],
    "ass_course": ["course_assessment_by_course.csv"],
    "ass_summed": ["course_assessment_summed.csv"],
    "ass_improve": ["assessment_improvement.csv", "AssessmentImprovement.csv"],
    "seg_city_csv": ["city_cluster_distribution.csv", "City_Cluster_Distribution.csv"],
    "experiment": ["experiment_curriculum_cleaned.csv", "nls_experiment_cleaned.csv"],
    # Excel-first PCA bundle (your new workbook)
    "pca_workbook": ["pca_components.xlsx"],
    # Optional fallbacks
    "pca_components_csv": ["pca_components.csv"],
    "survey_qs": ["survey_questions.xlsx", "survey_questions.csv"],
}

# ───────────────────────────
# Robust file helpers
# ───────────────────────────
@st.cache_data(show_spinner=False)
def find_first(candidates):
    for base in SEARCH_DIRS:
        for name in candidates:
            p = base / name
            if p.exists():
                return p
        if base.exists():
            for p in base.glob("*"):
                if p.is_file() and p.suffix.lower() in (".csv", ".xlsx", ".xls"):
                    for name in candidates:
                        if p.name.lower() == name.lower():
                            return p
    return None

def _read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        with open(path, "rb") as fh:
            sig = fh.read(4)
        if sig.startswith(b"PK\x03\x04"):
            return pd.read_excel(path)
    except Exception:
        pass
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
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

@st.cache_data(show_spinner=False)
def read_any_csv(kind):
    p = find_first(FILENAMES[kind])
    if p is None:
        return None, None
    try:
        return _read_csv_robust(p), p
    except Exception:
        try:
            return pd.read_excel(p), p
        except Exception:
            return None, p

# ───────────────────────────
# Small utilities
# ───────────────────────────
def ensure_numeric(s: pd.Series):
    return pd.to_numeric(s, errors="coerce")

def as_text(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return obj.astype(str).str.strip()

def wrap_text(s: str, width: int = 28) -> str:
    words, line, out = str(s).split(), "", []
    for w in words:
        if len(line) + len(w) + 1 <= width:
            line = (line + " " + w).strip()
        else:
            out.append(line); line = w
    if line: out.append(line)
    return "<br>".join(out)

def pc_index(x):
    m = re.search(r"PC\s*(\d+)", str(x), re.I)
    return int(m.group(1)) if m else 1_000

def cluster_index(x):
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else 1_000

# ───────────────────────────
# Load non-PCA datasets
# ───────────────────────────
enr, _          = read_any_csv("enroll")
ass_course, _   = read_any_csv("ass_course")
ass_summed, _   = read_any_csv("ass_summed")
ass_improve, _  = read_any_csv("ass_improve")
seg_city_csv, _ = read_any_csv("seg_city_csv")
experiment, _   = read_any_csv("experiment")

# ───────────────────────────
# Load PCA workbook (your screenshots)
# ───────────────────────────
@st.cache_data(show_spinner=False)
def load_pca_workbook():
    xlsx = find_first(FILENAMES["pca_workbook"])
    combo = {"loadings": None, "explained": None, "centers": None, "city_pct": None, "path": xlsx}
    if not xlsx:
        return combo

    # Loadings
    try:
        loadings = pd.read_excel(xlsx, sheet_name="Loadings")
        # Your screenshot shows the first column (PC1/PC2/PC3) has no header.
        # If so, rename the first column to "Response".
        if "Response" not in loadings.columns:
            loadings = loadings.rename(columns={loadings.columns[0]: "Response"})
        combo["loadings"] = loadings
    except Exception:
        pass

    # ExplainedVariance — strip % and coerce numeric
    def read_explained(sheet_name):
        try:
            ev = pd.read_excel(xlsx, sheet_name=sheet_name)
            # Accept your header text exactly as in the screenshot.
            pc_col  = next((c for c in ev.columns if "principal" in str(c).lower()), ev.columns[0])
            var_col = next((c for c in ev.columns if "variance"  in str(c).lower()), ev.columns[1])
            ev = ev[[pc_col, var_col]].rename(columns={pc_col: "Principal Component", var_col: "Explained Variance (%)"})
            ev["Explained Variance (%)"] = (
                ev["Explained Variance (%)"]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            ev["Explained Variance (%)"] = pd.to_numeric(ev["Explained Variance (%)"], errors="coerce")
            ev = ev.dropna(subset=["Explained Variance (%)"])
            return ev if not ev.empty else None
        except Exception:
            return None

    for cand in ["ExplainedVariance", "Explained Variance", "EV", "Variance"]:
        ev = read_explained(cand)
        if ev is not None:
            combo["explained"] = ev
            break

    # ClusterCenters (if you have this sheet; optional)
    try:
        centers = pd.read_excel(xlsx, sheet_name="ClusterCenters")
        if "Cluster" not in centers.columns:
            centers = centers.rename(columns={centers.columns[0]: "Cluster"})
        combo["centers"] = centers
    except Exception:
        pass

    # City/Cluster/Percentage (your 3rd screenshot) — if provided as a sheet named like below
    try:
        city_pct = pd.read_excel(xlsx, sheet_name="CityClusterDistribution")
        # normalize
        ren = {city_pct.columns[0]: "City", city_pct.columns[1]: "Cluster", city_pct.columns[2]: "Percentage"}
        city_pct = city_pct.rename(columns=ren)
        city_pct["Percentage"] = (
            city_pct["Percentage"].astype(str).str.replace("%", "", regex=False).str.strip()
        )
        city_pct["Percentage"] = pd.to_numeric(city_pct["Percentage"], errors="coerce") / 100.0
        combo["city_pct"] = city_pct
    except Exception:
        pass

    return combo

pca_combo = load_pca_workbook()

# If city_pct not in workbook, fall back to CSV (your repo already has this)
if pca_combo["city_pct"] is None and seg_city_csv is not None and not seg_city_csv.empty:
    sc = seg_city_csv.copy()
    # expect columns like: City_y | 0 | 1 | 2 | 3  → melt to long form
    city_col = "City_y" if "City_y" in sc.columns else sc.columns[0]
    cluster_cols = [c for c in sc.columns if str(c).strip().isdigit()]
    long_df = sc.melt(id_vars=[city_col], value_vars=cluster_cols, var_name="Cluster", value_name="Employees")
    long_df["Cluster"] = long_df["Cluster"].apply(lambda x: f"Cluster {x}")
    pca_combo["city_pct"] = long_df  # will use for “Segments by City” (counts)

# ───────────────────────────
# KPI row (clean + relevant)
# ───────────────────────────
kpi = {}
if enr is not None and not enr.empty:
    c_country = "Country_Regional_Center" if "Country_Regional_Center" in enr.columns else enr.columns[0]
    c_enroll  = "Total_Enrollments"       if "Total_Enrollments"       in enr.columns else enr.columns[1]
    enr[c_enroll] = ensure_numeric(enr[c_enroll])
    kpi["Total Enrollments"]   = f"{int(enr[c_enroll].sum(skipna=True)):,}"
    kpi["Countries Represented"] = as_text(enr, c_country).nunique()

if ass_course is not None and not ass_course.empty and "Course_Title" in ass_course.columns:
    kpi["Courses Analyzed"] = as_text(ass_course, "Course_Title").nunique()

ev = pca_combo.get("explained")
if ev is not None and not ev.empty:
    total_var = float(ensure_numeric(ev["Explained Variance (%)"]).sum())
    kpi["Variance Explained (PC1–PC3)"] = f"{total_var:.1f}%"

if kpi:
    cols = st.columns(min(4, len(kpi)))
    for (label, value), c in zip(kpi.items(), cols):
        c.metric(label, value)

st.markdown("---")

# ───────────────────────────
# Tabs
# ───────────────────────────
tab1, tab2, tab3 = st.tabs(["📍 Enrollments", "🎯 Training Outcomes", "🧩 PCA & K-Means Segmentation"])

# ── TAB 1: Enrollments
with tab1:
    st.subheader("Enrollments by Country")
    if enr is None or enr.empty:
        st.info("Add `country_enrollment_summary.csv`.")
    else:
        c_country = "Country_Regional_Center" if "Country_Regional_Center" in enr.columns else enr.columns[0]
        c_enroll  = "Total_Enrollments"       if "Total_Enrollments"       in enr.columns else enr.columns[1]
        enr[c_enroll] = ensure_numeric(enr[c_enroll])
        view = enr.dropna(subset=[c_enroll]).copy()
        view["_country"] = as_text(view, c_country)

        # default top 10
        top10 = view.sort_values(c_enroll, ascending=False).head(10)["_country"].tolist()
        picks = st.multiselect("Countries (default: Top 10 by enrollments)", options=sorted(view["_country"].unique()), default=top10)
        if picks:
            view = view[view["_country"].isin(picks)]

        order = st.radio("Sort by", ["Enrollments (desc)", "Country (A–Z)"], horizontal=True)
        if order.startswith("Enrollments"):
            view = view.sort_values(c_enroll, ascending=False)
        else:
            view = view.sort_values("_country")

        if view.empty:
            st.info("No countries selected.")
        else:
            fig = px.bar(view, x="_country", y=c_enroll, height=420,
                         labels={"_country": "Country", c_enroll: "Enrollments"},
                         title="Enrollments for Selected Countries")
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_title_standoff=12)
            st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: Training Outcomes
with tab2:
    st.subheader("Training Outcomes by Course and Delivery Mode")
    st.caption("Proficiency = knowledge/skill on the assessment; Application = on-the-job use after training.")

    if ass_course is None or ass_course.empty or "Course_Title" not in ass_course.columns:
        st.info("Add `course_assessment_by_course.csv`.")
    else:
        df = ass_course.copy()
        # normalize metrics
        df["Delivery_Mode"] = df["Course_Title"].apply(lambda t: "Virtual" if isinstance(t, str) and "virtual" in t.lower() else "In-Person")
        df["Average Proficiency Change (Post − Pre)"] = ensure_numeric(df["Outcome_Proficiency_Score"]) - ensure_numeric(df["Intake_Proficiency_Score"])
        df["Average Application Change (Post − Pre)"] = ensure_numeric(df["Outcome_Applications_Score"]) - ensure_numeric(df["Intake_Applications_Score"])
        df["Score Increase Ratio"]                   = ensure_numeric(df.get("Score_Increase"))
        df["Avg Outcome Proficiency Score (Post)"]   = ensure_numeric(df["Outcome_Proficiency_Score"])
        df["Avg Outcome Application Score (Post)"]   = ensure_numeric(df["Outcome_Applications_Score"])

        metric_options = [
            "Average Proficiency Change (Post − Pre)",
            "Average Application Change (Post − Pre)",
            "Score Increase Ratio",
            "Avg Outcome Proficiency Score (Post)",
            "Avg Outcome Application Score (Post)",
        ]
        metric_label = st.selectbox("Metric", metric_options, index=1, help="Aggregated per course; “Change” = post − pre.")
        course_picks = st.multiselect("Courses (optional)", options=sorted(df["Course_Title"].dropna().unique()), default=[])

        df_plot = df if not course_picks else df[df["Course_Title"].isin(course_picks)]
        df_plot = df_plot.dropna(subset=[metric_label])

        if df_plot.empty:
            st.info("No rows with numeric values for the selected metric/courses.")
        else:
            left, right = st.columns([1.1, 1])
            with left:
                mean_mode = df_plot.groupby("Delivery_Mode", as_index=False)[metric_label].mean()
                fig = px.bar(mean_mode, x="Delivery_Mode", y=metric_label, height=400,
                             labels={"Delivery_Mode": "Delivery Mode", metric_label: metric_label},
                             title=f"{metric_label} by Delivery Mode")
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_title_standoff=14)
                st.plotly_chart(fig, use_container_width=True)

            with right:
                top = (df_plot.groupby("Course_Title", as_index=False)[metric_label]
                       .mean()
                       .sort_values(metric_label, ascending=False)
                       .head(15))
                top["_Course_Wrapped"] = top["Course_Title"].apply(lambda s: wrap_text(s, 28))
                fig2 = px.bar(top, y="_Course_Wrapped", x=metric_label, orientation="h", height=520,
                              labels={"_Course_Wrapped": "Course", metric_label: metric_label},
                              title=f"{metric_label} — Top 15 Courses")
                fig2.update_traces(text=top[metric_label].round(2), textposition="outside", cliponaxis=False)
                fig2.update_layout(margin=dict(l=10, r=30, t=60, b=10),
                                   yaxis={"categoryorder": "total ascending"},
                                   xaxis_title_standoff=14)
                st.plotly_chart(fig2, use_container_width=True)

# ── TAB 3: PCA & Segmentation
with tab3:
    st.subheader("K-Means Segmentation (k=4) and PCA Summary")

    # Segments by city (either from sheet or CSV)
    if isinstance(pca_combo["city_pct"], pd.DataFrame):
        st.markdown("#### Segments by City")
        city_df = pca_combo["city_pct"].copy()
        if {"City", "Cluster", "Employees"}.issubset(set(city_df.columns)):
            # already counts form
            view = city_df
        else:
            # percentage form (your sheet) → show stacked bar by percentage
            city_df["Percentage"] = ensure_numeric(city_df["Percentage"])
            fig_loc = px.bar(city_df, x="City", y="Percentage", color="Cluster", height=380,
                             labels={"Percentage": "Share of Employees", "City": "City", "Cluster": "Segment"},
                             title="Segment Share by City (Percentage)")
            fig_loc.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_title_standoff=12)
            st.plotly_chart(fig_loc, use_container_width=True)
            view = None

        if view is not None:
            fig_loc = px.bar(view, x="City", y="Employees", color="Cluster", height=380,
                             labels={"Employees": "Employees", "City": "City", "Cluster": "Segment"},
                             title="Segment Counts by City")
            fig_loc.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_title_standoff=12)
            st.plotly_chart(fig_loc, use_container_width=True)
    else:
        st.info("Add `CityClusterDistribution` sheet to the PCA workbook or `city_cluster_distribution.csv` to show segments by city.")

    st.markdown("#### PCA — Explained Variance")
    ev = pca_combo.get("explained")
    if ev is not None and not ev.empty:
        ev = ev.copy()
        ev["__order"] = ev["Principal Component"].apply(pc_index)
        ev = ev.sort_values("__order").drop(columns="__order")
        fig_ev = px.bar(ev, x="Principal Component", y="Explained Variance (%)", height=320,
                        labels={"Principal Component": "Principal Component", "Explained Variance (%)": "Explained Variance (%)"},
                        title="Explained Variance by Component")
        fig_ev.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_title_standoff=12)
        st.plotly_chart(fig_ev, use_container_width=True)
    else:
        st.warning("PCA explained variance not detected — ensure the workbook has a sheet named **ExplainedVariance** (or **Explained Variance**) with two columns: *Principal Component* and *Explained Variance* (values like `31.90%` are fine).")

    st.markdown("#### PCA — Top Contributing Questions")
    loadings = pca_combo.get("loadings")
    if loadings is not None and not loadings.empty:
        if "Response" not in loadings.columns:
            loadings = loadings.rename(columns={loadings.columns[0]: "Response"})
        question_cols = [c for c in loadings.columns if re.match(r"^Q\d+", str(c), re.I)]
        pcs = as_text(loadings, "Response")
        pc_pick = st.selectbox("Component", pcs.tolist(), index=0,
                               help="Highest absolute loadings indicate strongest contributing questions.")
        row = loadings[pcs == pc_pick].iloc[0]
        contrib = sorted(((q, float(row[q])) for q in question_cols), key=lambda x: abs(x[1]), reverse=True)[:8]
        disp = pd.DataFrame({"Survey Question (Q#)": [q for q, _ in contrib],
                             "Loading (± strength)": [v for _, v in contrib]})
        st.dataframe(disp, use_container_width=True)

    centers = pca_combo.get("centers")
    if centers is not None and not centers.empty:
        st.markdown("#### K-Means Cluster Centers in PCA Space")
        centers_sorted = centers.sort_values(by="Cluster", key=lambda s: s.map(cluster_index))
        st.dataframe(centers_sorted, use_container_width=True)
