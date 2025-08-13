# streamlit_app/app.py
from pathlib import Path
import re
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# Page — professional
# =========================
st.set_page_config(page_title="Workforce Analytics Dashboard", page_icon="📊", layout="wide")
st.title("Workforce Analytics Dashboard")
st.caption("Enrollments, Training Outcomes, Segmentation, and Program Improvement Analysis")

# =========================
# Paths & file discovery
# =========================
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
SEARCH_DIRS = [
    ROOT / "data" / "analysis-outputs",
    ROOT / "data" / "processed",
    ROOT / "data",  # fallback
]

FILENAMES = {
    "enroll": ["country_enrollment_summary.csv", "Country-wise_Enrollment_Summary.csv"],
    "ass_course": ["course_assessment_by_course.csv"],
    "ass_summed": ["course_assessment_summed.csv"],
    "ass_improve": ["assessment_improvement.csv", "AssessmentImprovement.csv"],
    "seg_city": ["city_cluster_distribution.csv", "City_Cluster_Distribution.csv"],
    "survey_loc": ["emp_survey_with_locations.csv", "surveyed_employees_with_full_locations.csv"],
    "experiment": ["experiment_curriculum_cleaned.csv", "nls_experiment_cleaned.csv"],
    "pca_components": ["pca_components.csv"],
    "pca_kmeans": ["pca_kmeans_results.csv", "pca_kmeans_results.xlsx"],
    "combined": ["combined_data.csv"],
}

@st.cache_data(show_spinner=False)
def find_first(candidates):
    """Find the first matching file across SEARCH_DIRS (case-insensitive fallback)."""
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

# --- robust CSV reader (fixes UnicodeDecodeError & mis-saved Excel-as-CSV) ---
def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read CSV with multiple fallbacks (encodings/engines); detect Excel-zip mis-saves."""
    try:
        with open(path, "rb") as fh:
            sig = fh.read(4)
        if sig.startswith(b"PK\x03\x04"):
            return pd.read_excel(path)
    except Exception:
        pass
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            try:
                return pd.read_csv(path, low_memory=False, encoding=enc, engine="python")
            except Exception:
                continue
    return pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8",
        on_bad_lines="skip",
        engine="python",
        dtype=str,
    )

@st.cache_data(show_spinner=False)
def read_any(kind):
    """Read CSV/XLSX by kind with robust fallbacks."""
    p = find_first(FILENAMES[kind])
    if p is None:
        return None, None
    suf = p.suffix.lower()
    if suf == ".csv":
        try:
            df = _read_csv_robust(p)
            return df, p
        except Exception:
            try:
                return pd.read_excel(p), p
            except Exception:
                return None, p
    if suf in (".xlsx", ".xls"):
        try:
            return pd.read_excel(p), p
        except Exception:
            return None, p
    return None, p

# =========================
# Helpers
# =========================
TEXTY_HINTS = ("name", "title", "country", "city", "office", "region", "location", "segment", "cluster", "label")

def as_text(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return obj.astype(str).str.strip()

def ensure_numeric(s: pd.Series):
    return pd.to_numeric(s, errors="coerce")

def top_n(series: pd.Series, n=10):
    vc = series.value_counts(dropna=True)
    return vc.index[:n].tolist()

def human_locations(values):
    out = []
    for v in pd.Series(values).dropna().unique():
        s = str(v).strip()
        if re.match(r"^\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*$", s):  # lat,lon
            continue
        if re.match(r"^\s*-?\d+(\.\d+)?\s*$", s):  # pure number
            continue
        out.append(s)
    return sorted(out)

def delivery_from_title(title: str) -> str:
    if isinstance(title, str) and "virtual" in title.lower():
        return "Virtual"
    return "In-Person"

def extract_explained_variance(df: pd.DataFrame):
    """Try to extract rows like 'PC1 ...' with a '%', regardless of column names."""
    out = []
    sdf = df.astype(str)
    for _, row in sdf.iterrows():
        pc = None
        pct = None
        for cell in row:
            if re.search(r"^\s*PC\d+", cell, re.I):
                pc = re.search(r"(PC\d+)", cell, re.I).group(1).upper()
            if re.search(r"\d+(\.\d+)?\s*%", cell):
                pct = re.search(r"(\d+(\.\d+)?)\s*%", cell).group(1)
        if pc and pct:
            out.append((pc, float(pct)))
    if out:
        return pd.DataFrame(out, columns=["Principal Component", "Explained Variance (%)"])
    return pd.DataFrame()

# =========================
# Load datasets
# =========================
enr, _ = read_any("enroll")
ass_course, _ = read_any("ass_course")
ass_summed, _ = read_any("ass_summed")  # optional reference table
ass_improve, _ = read_any("ass_improve")  # not displayed by default (kept for future)
seg_city, _ = read_any("seg_city")
survey_loc, _ = read_any("survey_loc")
exp, _ = read_any("experiment")
pca_components, _ = read_any("pca_components")
pca_kmeans, _ = read_any("pca_kmeans")
combined, _ = read_any("combined")

# =========================
# KPI row — simple & relevant
# =========================
kpis = {}

if enr is not None and not enr.empty:
    c_country = "Country_Regional_Center" if "Country_Regional_Center" in enr.columns else enr.columns[0]
    c_enroll = "Total_Enrollments" if "Total_Enrollments" in enr.columns else enr.columns[1]
    enr[c_enroll] = ensure_numeric(enr[c_enroll])
    kpis["Total Enrollments"] = f"{int(enr[c_enroll].sum(skipna=True)):,}"
    kpis["Countries Represented"] = as_text(enr, c_country).nunique()

if ass_course is not None and not ass_course.empty and "Course_Title" in ass_course.columns:
    kpis["Courses Analyzed"] = as_text(ass_course, "Course_Title").nunique()

# Segments KPI (count clusters)
if seg_city is not None and not seg_city.empty:
    cluster_cols = [c for c in seg_city.columns if c.strip().isdigit()]
    if cluster_cols:
        kpis["Employee Segments"] = len(cluster_cols)
elif pca_kmeans is not None and not pca_kmeans.empty:
    pcs = [c for c in pca_kmeans.columns if c.upper().startswith("PC")]
    if pcs:
        kpis["Employee Segments"] = len(pca_kmeans)

if kpis:
    cols = st.columns(min(4, len(kpis)))
    for (label, value), c in zip(kpis.items(), cols):
        c.metric(label, value)

st.markdown("---")

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["📍 Enrollments", "🎯 Training Outcomes", "🧩 Segmentation"])

# --------------------------------------------------------------------
# TAB 1 — Enrollments
# --------------------------------------------------------------------
with tab1:
    st.subheader("Enrollments by Country")
    if enr is None or enr.empty:
        st.info("Add `country_enrollment_summary.csv`.")
    else:
        c_country = "Country_Regional_Center" if "Country_Regional_Center" in enr.columns else enr.columns[0]
        c_enroll = "Total_Enrollments" if "Total_Enrollments" in enr.columns else enr.columns[1]
        enr[c_enroll] = ensure_numeric(enr[c_enroll])
        view = enr.dropna(subset=[c_enroll]).copy()
        country_s = as_text(view, c_country)
        default_countries = top_n(country_s, 10)
        picks = st.multiselect(
            "Countries (default: top 10 by enrollments)",
            options=sorted(country_s.unique()),
            default=default_countries
        )
        order = st.radio("Sort by", ["Enrollments (desc)", "Country (A–Z)"], horizontal=True)
        view["_country"] = country_s
        if picks:
            view = view[view["_country"].isin(picks)]
        view = view.sort_values(c_enroll if order.startswith("Enrollments") else "_country",
                                ascending=not order.startswith("Enrollments"))
        if view.empty:
            st.info("No countries selected.")
        else:
            fig = px.bar(view, x="_country", y=c_enroll, height=420,
                         labels={"_country": "Country", c_enroll: "Enrollments"},
                         title="Enrollments for Selected Countries")
            st.plotly_chart(fig, use_container_width=True, key="enroll_bar")

# --------------------------------------------------------------------
# TAB 2 — Training Outcomes (Aggregated; all courses by default)
# --------------------------------------------------------------------
with tab2:
    st.subheader("Training Outcomes by Course and Delivery Mode")
    if ass_course is None or ass_course.empty or "Course_Title" not in ass_course.columns:
        st.info("Add `course_assessment_by_course.csv`.")
    else:
        df = ass_course.copy()
        df["Delivery_Mode"] = df["Course_Title"].apply(delivery_from_title)
        # Compute clear, human-understandable metrics
        df["Proficiency_Change"] = ensure_numeric(df["Outcome_Proficiency_Score"]) - ensure_numeric(df["Intake_Proficiency_Score"])
        df["Applications_Change"] = ensure_numeric(df["Outcome_Applications_Score"]) - ensure_numeric(df["Intake_Applications_Score"])

        metric_map = {
            "Average proficiency change (Outcome − Intake)": "Proficiency_Change",
            "Average applications change (Outcome − Intake)": "Applications_Change",
            "Score increase ratio": "Score_Increase",
            "Average outcome proficiency score": "Outcome_Proficiency_Score",
            "Average outcome applications score": "Outcome_Applications_Score",
        }
        metric_label = st.selectbox("Metric", list(metric_map.keys()), index=0)
        metric_col = metric_map[metric_label]
        st.caption({
            "Average proficiency change (Outcome − Intake)": "Mean of outcome minus intake proficiency scores across courses.",
            "Average applications change (Outcome − Intake)": "Mean of outcome minus intake application scores across courses.",
            "Score increase ratio": "Precomputed improvement ratio from your dataset.",
            "Average outcome proficiency score": "Mean final proficiency scores (post-training).",
            "Average outcome applications score": "Mean final application scores (post-training).",
        }[metric_label])

        # Optional course filter, blank = all
        course_opts = sorted(df["Course_Title"].dropna().unique().tolist())
        picks = st.multiselect("Filter courses (blank = all)", options=course_opts, default=[])

        df["_y"] = ensure_numeric(df[metric_col])
        df = df.dropna(subset=["_y"])
        if picks:
            df = df[df["Course_Title"].isin(picks)]

        if df.empty:
            st.info("No rows with numeric values for the chosen selection.")
        else:
            left, right = st.columns([1.15, 1])
            with left:
                # by Delivery Mode (default, recruiter-friendly)
                mean_mode = df.groupby("Delivery_Mode", as_index=False)["_y"].mean()
                fig = px.bar(
                    mean_mode, x="Delivery_Mode", y="_y", height=400,
                    labels={"Delivery_Mode": "Delivery Mode", "_y": metric_label},
                    title=f"{metric_label} by Delivery Mode"
                )
                st.plotly_chart(fig, use_container_width=True, key="outcomes_mode")
            with right:
                # by Course (top 15 for readability)
                mean_course = (df.groupby("Course_Title", as_index=False)["_y"].mean()
                                 .sort_values("_y", ascending=False)
                                 .head(15))
                fig2 = px.bar(
                    mean_course, x="Course_Title", y="_y", height=400,
                    labels={"Course_Title": "Course", "_y": metric_label},
                    title=f"{metric_label} — Top 15 Courses"
                )
                st.plotly_chart(fig2, use_container_width=True, key="outcomes_course_top")

# --------------------------------------------------------------------
# TAB 3 — Segmentation
# --------------------------------------------------------------------
with tab3:
    st.subheader("Employee Segmentation")
    if seg_city is None or seg_city.empty:
        st.info("Add `city_cluster_distribution.csv` for segment counts by city.")
    else:
        df = seg_city.copy()
        city_col = "City_y" if "City_y" in df.columns else df.columns[0]
        cluster_cols = [c for c in df.columns if c.strip().isdigit()]

        # Segment sizes (sum across cities), ordered by cluster index 0,1,2,3...
        totals = df[cluster_cols].sum(numeric_only=True)
        ordered_clusters = sorted(cluster_cols, key=lambda x: int(x))
        sizes = pd.DataFrame({
            "Segment": [f"Cluster {c}" for c in ordered_clusters],
            "Employees": [int(totals[c]) for c in ordered_clusters]
        })

        c1, c2 = st.columns([1, 1.35])
        with c1:
            fig_sizes = px.bar(
                sizes, x="Segment", y="Employees", height=380,
                labels={"Segment": "Segment", "Employees": "Employees"},
                title="Segment Size"
            )
            st.plotly_chart(fig_sizes, use_container_width=True, key="seg_sizes")

        # Segments by city (stacked), preserve cluster order
        long_df = df.melt(id_vars=[city_col], value_vars=ordered_clusters, var_name="Cluster", value_name="Employees")
        long_df["Cluster"] = long_df["Cluster"].apply(lambda x: f"Cluster {x}")
        long_df["Employees"] = ensure_numeric(long_df["Employees"])
        long_df["Cluster"] = pd.Categorical(long_df["Cluster"], categories=[f"Cluster {c}" for c in ordered_clusters], ordered=True)

        city_opts = human_locations(long_df[city_col].unique().tolist())
        if not city_opts:
            city_opts = sorted(long_df[city_col].unique().tolist())
        default_cities = top_n(long_df[city_col], 12)
        picks = st.multiselect("Cities", options=city_opts, default=[c for c in default_cities if c in city_opts])

        view = long_df if not picks else long_df[long_df[city_col].isin(picks)]
        with c2:
            if view.empty:
                st.info("No data for the selected cities.")
            else:
                fig_loc = px.bar(
                    view, x=city_col, y="Employees", color="Cluster", height=380,
                    labels={city_col: "City", "Employees": "Employees", "Cluster": "Segment"},
                    title="Segments by City"
                )
                st.plotly_chart(fig_loc, use_container_width=True, key="seg_by_city")

    # Optional: PCA explained variance (only if detected)
    if pca_components is not None and not pca_components.empty:
        ev = extract_explained_variance(pca_components)
        if not ev.empty:
            with st.expander("PCA Explained Variance"):
                st.dataframe(ev, use_container_width=True)
