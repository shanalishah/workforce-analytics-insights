# streamlit_app/app.py
from pathlib import Path
import re
import io
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# Page — professional
# =========================
st.set_page_config(page_title="Workforce Analytics Dashboard", page_icon="📊", layout="wide")
st.title("Workforce Analytics Dashboard")
st.caption("Enrollments · Training Outcomes · PCA (Dimensionality Reduction) · K-Means Clustering (Segmentation)")

# =========================
# Paths & file discovery
# =========================
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
SEARCH_DIRS = [
    ROOT / "data" / "analysis-outputs",
    ROOT / "data" / "processed",
    ROOT / "data",          # fallback
    ROOT / "data" / "raw",  # for survey_questions.xlsx
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
    "survey_qs": ["survey_questions.xlsx", "survey_questions.csv"],
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
    """Read CSV/XLSX by kind with robust fallbacks and return (df, path)."""
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
# UX helpers & definitions
# =========================
DEFNS = {
    "proficiency": "Proficiency = knowledge or skill level demonstrated on the training assessment.",
    "application": "Application = how much that skill is used on the job (practical use after training).",
}

def wrap_text(s: str, width: int = 28) -> str:
    words, line, out = str(s).split(), "", []
    for w in words:
        if len(line) + len(w) + 1 <= width:
            line = (line + " " + w).strip()
        else:
            out.append(line); line = w
    if line: out.append(line)
    return "<br>".join(out)

def as_text(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.DataFrame): obj = obj.iloc[:, 0]
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

def pc_index(x):
    m = re.search(r"PC\s*(\d+)", str(x), re.I)
    return int(m.group(1)) if m else 1_000

def cluster_index(x):
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else 1_000

# ---- Parse your combo-style PCA CSV into pieces (loadings / explained / centers / city pct)
def parse_pca_combo_file(path: Path):
    if path is None or not path.exists():
        return {"loadings": None, "explained": None, "centers": None, "city_pct": None}

    # tolerant text read
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        try:
            text = path.read_text(encoding="cp1252", errors="replace")
        except Exception:
            text = path.read_bytes().decode("latin1", errors="replace")

    blocks = [b.strip() for b in re.split(r"\n\s*\n\s*\n*", text) if b.strip()]

    def try_read_csv_block(block: str):
        candidate = block.replace("\t", ",")
        candidate = re.sub(r",\s*,", ",,", candidate)
        try:
            return pd.read_csv(io.StringIO(candidate))
        except Exception:
            try:
                return pd.read_csv(io.StringIO(candidate), engine="python")
            except Exception:
                return None

    loadings = explained = centers = city_pct = None
    for b in blocks:
        dfb = try_read_csv_block(b)
        if dfb is None or dfb.empty:
            continue
        cols = [str(c).strip() for c in dfb.columns]
        cset = set(c.lower() for c in cols)

        # Loadings table
        if "response" in cset and ("q1" in cset or any(c.startswith("q") for c in cset)):
            for c in dfb.columns:
                if c.lower() not in ("response",):
                    dfb[c] = pd.to_numeric(dfb[c], errors="coerce")
            loadings = dfb
            continue

        # Explained variance table
        if "principal component" in cset and any("variance" in c for c in cset):
            var_col = next((c for c in dfb.columns if "variance" in c.lower()), None)
            if var_col is not None:
                ev = dfb.copy()
                ev[var_col] = ev[var_col].astype(str).str.replace("%", "", regex=False).str.strip()
                ev[var_col] = pd.to_numeric(ev[var_col], errors="coerce")
                ev = ev.rename(columns={var_col: "Explained Variance (%)"})
                explained = ev[["Principal Component", "Explained Variance (%)"]]
            continue

        # KMeans centers
        if "cluster" in cset and any(str(c).lower().startswith("pc") for c in cols):
            for c in dfb.columns:
                if str(c).lower() != "cluster":
                    dfb[c] = pd.to_numeric(dfb[c], errors="coerce")
            centers = dfb
            continue

        # City/cluster percentages
        if "city" in cset and "cluster" in cset and any("percent" in c for c in cset):
            pct_col = next((c for c in dfb.columns if "percent" in c.lower()), None)
            city_pct = dfb.copy()
            if pct_col:
                city_pct[pct_col] = city_pct[pct_col].astype(str).str.replace("%", "", regex=False).str.strip()
                city_pct[pct_col] = pd.to_numeric(city_pct[pct_col], errors="coerce") / 100.0
            city_col = next((c for c in city_pct.columns if c.lower() == "city"), "City")
            cl_col   = next((c for c in city_pct.columns if c.lower() == "cluster"), "Cluster")
            if pct_col:
                city_pct = city_pct.rename(columns={city_col: "City", cl_col: "Cluster", pct_col: "Percentage"})
            else:
                city_pct = city_pct.rename(columns={city_col: "City", cl_col: "Cluster"})
            continue

    return {"loadings": loadings, "explained": explained, "centers": centers, "city_pct": city_pct}

# =========================
# Load datasets
# =========================
enr, _ = read_any("enroll")
ass_course, _ = read_any("ass_course")
ass_summed, _ = read_any("ass_summed")  # optional
ass_improve, _ = read_any("ass_improve")  # optional
seg_city, _ = read_any("seg_city")
survey_loc, _ = read_any("survey_loc")
exp, _ = read_any("experiment")
pca_components_df, pca_components_path = read_any("pca_components")
pca_kmeans_df, _ = read_any("pca_kmeans")
survey_qs_df, _ = read_any("survey_qs")
combined_df, _ = read_any("combined")

# Parse combo PCA file (handles your stacked CSV)
pca_combo = parse_pca_combo_file(pca_components_path) if pca_components_path else {"explained": None, "loadings": None, "centers": None, "city_pct": None}

# Build Q→question mapping if available
q_to_text = {}
if survey_qs_df is not None and not survey_qs_df.empty:
    # Works for either xlsx/csv with columns QID, Question Text
    qid_col = next((c for c in survey_qs_df.columns if str(c).strip().lower() in ("qid", "question id")), None)
    qtext_col = next((c for c in survey_qs_df.columns if "question" in str(c).lower()), None)
    if qid_col and qtext_col:
        for _, r in survey_qs_df[[qid_col, qtext_col]].dropna().iterrows():
            q_to_text[str(r[qid_col]).strip().upper()] = str(r[qtext_col]).strip()

def pretty_question(label: str) -> str:
    key = str(label).strip().upper()
    return q_to_text.get(key, label)

# =========================
# KPI row — concise & relevant
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
    cluster_cols = [c for c in seg_city.columns if str(c).strip().isdigit()]
    if cluster_cols:
        kpis["Employee Segments (K-Means)"] = len(cluster_cols)
elif pca_combo.get("centers") is not None and not pca_combo["centers"].empty:
    kpis["Employee Segments (K-Means)"] = len(pca_combo["centers"])

# PCA KPI (variance covered by PCs if present)
pca_ev = pca_combo.get("explained")
if pca_ev is not None and not pca_ev.empty:
    total_var = ensure_numeric(pca_ev["Explained Variance (%)"]).sum()
    if pd.notna(total_var):
        kpis["Variance Explained (PC1–PC3)"] = f"{total_var:.1f}%"

if kpis:
    cols = st.columns(min(4, len(kpis)))
    for (label, value), c in zip(kpis.items(), cols):
        c.metric(label, value)

st.markdown("---")

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["📍 Enrollments", "🎯 Training Outcomes", "🧩 PCA & K-Means Segmentation"])

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
            default=default_countries,
            key="enroll_countries",
        )
        order = st.radio("Sort by", ["Enrollments (desc)", "Country (A–Z)"], horizontal=True, key="enroll_sort")
        view["_country"] = country_s
        if picks:
            view = view[view["_country"].isin(picks)]
        view = view.sort_values(c_enroll if order.startswith("Enrollments") else "_country",
                                ascending=not order.startswith("Enrollments"))
        if view.empty:
            st.info("No countries selected.")
        else:
            fig = px.bar(
                view, x="_country", y=c_enroll, height=420,
                labels={"_country": "Country", c_enroll: "Enrollments"},
                title="Enrollments for Selected Countries",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_title_standoff=12)
            st.plotly_chart(fig, use_container_width=True, key="enroll_bar")

# --------------------------------------------------------------------
# TAB 2 — Training Outcomes (Aggregated; All Courses default)
# --------------------------------------------------------------------
with tab2:
    st.subheader("Training Outcomes by Course and Delivery Mode")
    st.caption("“Proficiency” and “Application” are defined below each selector for clarity.")

    if ass_course is None or ass_course.empty or "Course_Title" not in ass_course.columns:
        st.info("Add `course_assessment_by_course.csv`.")
    else:
        df = ass_course.copy()
        df["Delivery_Mode"] = df["Course_Title"].apply(delivery_from_title)

        # Clear, recruiter-friendly metrics
        df["Average Proficiency Change (Post − Pre)"] = ensure_numeric(df["Outcome_Proficiency_Score"]) - ensure_numeric(df["Intake_Proficiency_Score"])
        df["Average Application Change (Post − Pre)"] = ensure_numeric(df["Outcome_Applications_Score"]) - ensure_numeric(df["Intake_Applications_Score"])
        df["Score Increase Ratio"] = ensure_numeric(df["Score_Increase"])
        df["Avg Outcome Proficiency Score (Post)"] = ensure_numeric(df["Outcome_Proficiency_Score"])
        df["Avg Outcome Application Score (Post)"] = ensure_numeric(df["Outcome_Applications_Score"])

        metric_options = [
            "Average Proficiency Change (Post − Pre)",
            "Average Application Change (Post − Pre)",
            "Score Increase Ratio",
            "Avg Outcome Proficiency Score (Post)",
            "Avg Outcome Application Score (Post)",
        ]

        metric_label = st.selectbox(
            "Metric",
            metric_options,
            index=0,
            key="outcomes_metric",
            help="All measures are aggregated per course. “Change” = post-training minus pre-training."
        )
        # Micro-copy definitions (plain English)
        with st.caption(unsafe_allow_html=True):
            st.write(f"**Proficiency** — {DEFNS['proficiency']}")
            st.write(f"**Application** — {DEFNS['application']}")

        # Optional course narrowing (default = all)
        course_opts = sorted(df["Course_Title"].dropna().unique().tolist())
        picks = st.multiselect("Courses (optional)", options=course_opts, default=[], key="outcomes_courses")

        df_plot = df if not picks else df[df["Course_Title"].isin(picks)]
        df_plot = df_plot.dropna(subset=[metric_label])

        if df_plot.empty:
            st.info("No rows with numeric values for the chosen selection.")
        else:
            left, right = st.columns([1.15, 1])

            # LEFT: delivery mode comparison
            with st.container():
                with left:
                    mean_mode = df_plot.groupby("Delivery_Mode", as_index=False)[metric_label].mean()
                    fig = px.bar(
                        mean_mode, x="Delivery_Mode", y=metric_label, height=400,
                        labels={"Delivery_Mode": "Delivery Mode", metric_label: metric_label},
                        title=f"{metric_label} by Delivery Mode",
                    )
                    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_title_standoff=14)
                    st.plotly_chart(fig, use_container_width=True, key="outcomes_mode_bar")

            # RIGHT: top courses — horizontal, wrapped labels, values on bars
            with st.container():
                with right:
                    top = (df_plot.groupby("Course_Title", as_index=False)[metric_label]
                           .mean()
                           .sort_values(metric_label, ascending=False)
                           .head(15))
                    top["_Course_Wrapped"] = top["Course_Title"].apply(lambda s: wrap_text(s, 28))

                    fig2 = px.bar(
                        top, y="_Course_Wrapped", x=metric_label, orientation="h", height=520,
                        labels={"_Course_Wrapped": "Course", metric_label: metric_label},
                        title=f"{metric_label} — Top 15 Courses",
                    )
                    fig2.update_traces(text=top[metric_label].round(2), textposition="outside", cliponaxis=False)
                    fig2.update_layout(
                        margin=dict(l=10, r=30, t=60, b=10),
                        yaxis={"categoryorder": "total ascending"},
                        xaxis_title_standoff=14
                    )
                    st.plotly_chart(fig2, use_container_width=True, key="outcomes_course_hbar")

# --------------------------------------------------------------------
# TAB 3 — PCA & K-Means Segmentation
# --------------------------------------------------------------------
with tab3:
    st.subheader("K-Means Clustering — 4 Employee Segments")
    st.caption("Segments generated via K-Means (k=4) on PCA-transformed features, grouping employees with similar skill profiles.")

    # 3A) Segment Size and Segments by City
    if seg_city is None or seg_city.empty:
        st.info("Add `city_cluster_distribution.csv` for segment counts by city.")
    else:
        df = seg_city.copy()
        city_col = "City_y" if "City_y" in df.columns else df.columns[0]
        cluster_cols = [c for c in df.columns if str(c).strip().isdigit()]
        ordered_clusters = sorted(cluster_cols, key=lambda x: int(x))
        totals = df[ordered_clusters].sum(numeric_only=True)

        sizes = pd.DataFrame({
            "Segment": [f"Cluster {c}" for c in ordered_clusters],
            "Employees": [int(totals[c]) for c in ordered_clusters]
        })

        c1, c2 = st.columns([1, 1.35])
        with c1:
            fig_sizes = px.bar(
                sizes, x="Segment", y="Employees", height=380,
                labels={"Segment": "Segment", "Employees": "Employees"},
                title="K-Means Segment Size (Counts)",
            )
            fig_sizes.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_title_standoff=12)
            st.plotly_chart(fig_sizes, use_container_width=True, key="seg_sizes")

        long_df = df.melt(id_vars=[city_col], value_vars=ordered_clusters, var_name="Cluster", value_name="Employees")
        long_df["Cluster"] = long_df["Cluster"].apply(lambda x: f"Cluster {x}")
        long_df["Employees"] = ensure_numeric(long_df["Employees"])
        long_df["Cluster"] = pd.Categorical(long_df["Cluster"], categories=[f"Cluster {c}" for c in ordered_clusters], ordered=True)

        city_opts = human_locations(long_df[city_col].unique().tolist())
        if not city_opts:
            city_opts = sorted(long_df[city_col].unique().tolist())
        default_cities = top_n(long_df[city_col], 12)
        picks = st.multiselect("Cities (optional)", options=city_opts, default=[c for c in default_cities if c in city_opts], key="seg_cities")

        view = long_df if not picks else long_df[long_df[city_col].isin(picks)]
        with c2:
            if view.empty:
                st.info("No data for the selected cities.")
            else:
                fig_loc = px.bar(
                    view, x=city_col, y="Employees", color="Cluster", height=380,
                    labels={city_col: "City", "Employees": "Employees", "Cluster": "Segment"},
                    title="Segments by City (K-Means)",
                )
                fig_loc.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_title_standoff=12)
                st.plotly_chart(fig_loc, use_container_width=True, key="seg_by_city")

    st.markdown("### PCA — Dimensionality Reduction")
    st.caption("PCA reduces many survey questions into a few underlying components (e.g., Skill Development, Operational Focus, Career Advancement).")

    # 3B) PCA Explained Variance (from parsed combo or tidy file)
    ev = None
    if pca_combo.get("explained") is not None and not pca_combo["explained"].empty:
        ev = pca_combo["explained"].copy()
        ev["__order"] = ev["Principal Component"].apply(pc_index)
        ev = ev.sort_values("__order").drop(columns="__order")
    elif pca_components_df is not None and not pca_components_df.empty:
        var_col = next((c for c in pca_components_df.columns if "variance" in str(c).lower()), None)
        pc_col = next((c for c in pca_components_df.columns if "principal" in str(c).lower()), None)
        if var_col and pc_col:
            ev = pca_components_df[[pc_col, var_col]].rename(
                columns={pc_col: "Principal Component", var_col: "Explained Variance (%)"}
            ).copy()
            ev["Explained Variance (%)"] = pd.to_numeric(ev["Explained Variance (%)"], errors="coerce")

    if ev is not None and not ev.empty:
        fig_ev = px.bar(
            ev, x="Principal Component", y="Explained Variance (%)", height=320,
            labels={"Principal Component": "Principal Component", "Explained Variance (%)": "Explained Variance (%)"},
            title="Explained Variance by Component",
        )
        fig_ev.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_title_standoff=12)
        st.plotly_chart(fig_ev, use_container_width=True, key="pca_explained_variance")
    else:
        st.info("PCA explained variance not detected — ensure `data/analysis-outputs/pca_components.csv` is present (combo format supported).")

    # 3C) Top contributing survey questions per component (from loadings)
    loadings = pca_combo.get("loadings")
    if loadings is not None and not loadings.empty and "Response" in loadings.columns:
        question_cols = [c for c in loadings.columns if re.match(r"^Q\\d+", str(c), re.I)]
        pcs = as_text(loadings, "Response")
        pc_pick = st.selectbox("Component", pcs.tolist(), index=0, key="pca_component_pick",
                               help="Top contributing survey questions (highest absolute loadings) for the selected component.")
        row = loadings[pcs == pc_pick].iloc[0]
        contrib = sorted(((q, float(row[q])) for q in question_cols), key=lambda x: abs(x[1]), reverse=True)[:8]
        disp = pd.DataFrame({
            "Survey Question": [pretty_question(q) for q, _ in contrib],
            "Loading (strength & direction)": [v for _, v in contrib],
        })
        st.dataframe(disp, use_container_width=True)

    # 3D) K-Means Cluster Centers in PCA Space (optional table if present)
    centers = pca_combo.get("centers")
    if centers is not None and not centers.empty:
        st.markdown("#### K-Means Cluster Centers (PCA Space)")
        centers_sorted = centers.sort_values(by="Cluster", key=lambda s: s.map(cluster_index))
        st.dataframe(centers_sorted, use_container_width=True)
