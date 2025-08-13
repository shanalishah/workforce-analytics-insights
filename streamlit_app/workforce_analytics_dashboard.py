# streamlit_app/workforce_analytics_dashboard.py
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Workforce Analytics — Interactive Insights",
    page_icon="📊",
    layout="wide",
    menu_items={"About": "Workforce Analytics — Enrollments, Segments (PCA+KMeans), and Program Improvements"},
)

# Subtle CSS polish
st.markdown("""
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
.section-box { background: rgba(0,0,0,0.03); padding: .6rem .8rem; border-radius: .5rem; }
</style>
""", unsafe_allow_html=True)

# ---------------- Paths & robust loaders ----------------
ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [
    ROOT / "data" / "analysis-outputs",
    ROOT / "data" / "processed",
    ROOT / "data" / "raw",   # used only for optional mappings (e.g., survey questions), if present
]

@st.cache_data(show_spinner=False)
def find_path(name: str):
    for base in SEARCH_DIRS:
        p = base / name
        if p.exists():
            return p
    return None

@st.cache_data(show_spinner=False)
def read_any(name: str):
    p = find_path(name)
    if p is None:
        return None, None
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, low_memory=False), p
    elif p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p), p
    else:
        return None, None

def guess_col(df: pd.DataFrame, candidates, *, prefer_text=True, exclude_regex=None):
    """
    Guess a sensible column from a list of candidate names.
    If prefer_text, prioritizes object dtype columns.
    Can exclude bad matches via regex.
    """
    cols = list(df.columns)
    name_map = {c.lower(): c for c in cols}
    # First by exact candidate name (case-insensitive)
    for cand in candidates:
        c = name_map.get(str(cand).lower())
        if c is not None:
            if exclude_regex and re.search(exclude_regex, c, re.I):
                continue
            if prefer_text and not pd.api.types.is_object_dtype(df[c]):
                # skip numeric if we're preferring text
                continue
            return c
    # Fallback: first object/text column not excluded
    if prefer_text:
        for c in cols:
            if exclude_regex and re.search(exclude_regex, c, re.I):
                continue
            if pd.api.types.is_object_dtype(df[c]):
                return c
    # Last resort: first column not excluded
    for c in cols:
        if exclude_regex and re.search(exclude_regex, c, re.I):
            continue
        return c
    return cols[0] if cols else None

def is_textual_location_series(s: pd.Series) -> bool:
    """Heuristic: location-like series should be mostly text with letters, not numeric/IDs."""
    s = s.dropna().astype(str)
    if s.empty:
        return False
    # At least 70% should contain a letter (avoid coordinates and pure IDs)
    pct_letters = (s.str.contains(r"[A-Za-z]", regex=True)).mean()
    # Limit extreme cardinality
    unique_ratio = s.nunique() / max(1, len(s))
    return pct_letters >= 0.7 and unique_ratio <= 0.8

def best_location_column(df: pd.DataFrame):
    # Try common names and ensure textual
    for cand in ["location", "city", "office", "region", "country"]:
        if cand in (c.lower() for c in df.columns):
            c = [x for x in df.columns if x.lower() == cand][0]
            if is_textual_location_series(df[c]):
                return c
    # Else pick the first textual column that looks location-like
    for c in df.columns:
        if is_textual_location_series(df[c]):
            return c
    # Fallback to first object column
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            return c
    return df.columns[0]

def best_course_column(df: pd.DataFrame):
    # Prefer human-friendly names
    for cand in ["course_title", "course_name"]:
        if cand in (c.lower() for c in df.columns):
            return [x for x in df.columns if x.lower() == cand][0]
    # Fallbacks
    for cand in ["course", "course_id"]:
        if cand in (c.lower() for c in df.columns):
            return [x for x in df.columns if x.lower() == cand][0]
    return df.columns[0]

@st.cache_data(show_spinner=False)
def load_survey_mapping():
    """
    Try to find a mapping like Q1->full question text.
    Looks for files named nls_survey_questions.xlsx or survey_questions.xlsx in SEARCH_DIRS.
    Expects first sheet with two columns: code (Q1...) and question text.
    """
    for cand in ["nls_survey_questions.xlsx", "survey_questions.xlsx"]:
        p = find_path(cand)
        if p is not None and p.suffix.lower() in (".xlsx", ".xls"):
            try:
                df = pd.read_excel(p)
                # try to auto-detect columns
                cols = [c.lower() for c in df.columns]
                if "code" in cols and "question" in cols:
                    code_col = df.columns[cols.index("code")]
                    text_col = df.columns[cols.index("question")]
                else:
                    # guess first two columns
                    code_col, text_col = df.columns[:2]
                mp = dict(zip(df[code_col].astype(str).str.strip(), df[text_col].astype(str).str.strip()))
                return mp
            except Exception:
                pass
    return {}

def apply_question_mapping(series: pd.Series, mapping: dict) -> pd.Series:
    """Replace Q1/Q2/etc with full question text when mapping is available."""
    if not mapping:
        return series
    def repl(x):
        sx = str(x).strip()
        if sx in mapping:
            return mapping[sx]
        # also handle like 'Q1 - something'
        m = re.match(r"^(Q\d+)", sx, flags=re.I)
        if m and m.group(1).upper() in mapping:
            return mapping[m.group(1).upper()]
        return sx
    return series.map(repl)

# ---------------- Header ----------------
st.title("Workforce Analytics — Interactive Insights")
st.caption("Explore training enrollments, employee segments (PCA + KMeans), and curriculum experiment outcomes.")

# ---------------- KPIs (useful metrics) ----------------
# We compute what we can from available files; show up to four relevant metrics.
kpi_vals = {}
# Enrollments
enr_df, _ = read_any("country_enrollment_summary.csv")
if enr_df is None:
    enr_df, _ = read_any("Country-wise_Enrollment_Summary.csv")
if enr_df is not None and not enr_df.empty:
    c_country = guess_col(enr_df, ["country", "country_name", "nation"])
    c_enroll  = guess_col(enr_df, ["enrollments", "enrollment", "total_enrollments", "count"], prefer_text=False)
    enr_df[c_enroll] = pd.to_numeric(enr_df[c_enroll], errors="coerce")
    enr_df = enr_df.dropna(subset=[c_enroll])
    if not enr_df.empty:
        kpi_vals["Total enrollments"] = f"{int(enr_df[c_enroll].sum()):,}"
        kpi_vals["Countries represented"] = enr_df[c_country].nunique()

# Courses
ass_df, _ = read_any("course_assessment_by_course.csv")
if ass_df is None:
    ass_df, _ = read_any("Course_wise_assessment.csv")
if ass_df is not None and not ass_df.empty:
    c_course = best_course_column(ass_df)
    kpi_vals["Courses analyzed"] = ass_df[c_course].astype(str).nunique()

# Segments
seg_df, _ = read_any("pca_kmeans_results.xlsx")
if seg_df is None:
    seg_df, _ = read_any("pca_kmeans_results.csv")
if seg_df is not None and not seg_df.empty:
    c_seg = guess_col(seg_df, ["segment", "cluster", "group", "label"])
    if c_seg:
        kpi_vals["Employee segments discovered"] = seg_df[c_seg].astype(str).nunique()

# Experiment improvements
exp_df, _ = read_any("experiment_curriculum_cleaned.csv")
if exp_df is None:
    exp_df, _ = read_any("nls_experiment_cleaned.csv")
if exp_df is not None and not exp_df.empty:
    # try to compute median improvement in proficiency
    pre_p  = guess_col(exp_df, ["pre_proficiency", "proficiency_pre", "pre_prof"], prefer_text=False)
    post_p = guess_col(exp_df, ["post_proficiency", "proficiency_post", "post_prof"], prefer_text=False)
    if pre_p and post_p:
        imp = pd.to_numeric(exp_df[post_p], errors="coerce") - pd.to_numeric(exp_df[pre_p], errors="coerce")
        if not imp.dropna().empty:
            kpi_vals["Median proficiency improvement"] = f"{imp.median():.2f}"

# Render up to 4 KPIs
if kpi_vals:
    cols = st.columns(min(4, len(kpi_vals)))
    for (label, value), col in zip(kpi_vals.items(), cols):
        col.metric(label, value)

st.markdown("---")

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs([
    "📍 Enrollments by Country",
    "🧩 Employee Segments (PCA + KMeans)",
    "🧪 Program Improvements (Experiment)",
])

# ==================== TAB 1: ENROLLMENTS BY COUNTRY ====================
with tab1:
    st.markdown("### Enrollments by Country")
    df_u, path_u = read_any("country_enrollment_summary.csv")
    if df_u is None:
        df_u, path_u = read_any("Country-wise_Enrollment_Summary.csv")

    if df_u is not None and not df_u.empty:
        col_country = guess_col(df_u, ["country", "country_name", "nation"])
        col_enroll  = guess_col(df_u, ["enrollments", "enrollment", "total_enrollments", "count"], prefer_text=False)
        df_u[col_enroll] = pd.to_numeric(df_u[col_enroll], errors="coerce")
        df_u = df_u.dropna(subset=[col_enroll])

        # ---- Controls (clear and meaningful) ----
        # Default selection = top 10 countries by enrollments
        top10 = df_u.sort_values(col_enroll, ascending=False)[col_country].astype(str).head(10).tolist()
        countries = st.multiselect(
            "Countries to include (default: top 10 by enrollments)",
            options=sorted(df_u[col_country].astype(str).unique()),
            default=top10,
        )
        sort_by = st.radio("Sort bars by", ["Enrollments (descending)", "Country (A–Z)"], horizontal=True)
        filtered = df_u[df_u[col_country].astype(str).isin(countries)] if countries else df_u.copy()
        if sort_by.startswith("Enrollments"):
            filtered = filtered.sort_values(col_enroll, ascending=False)
        else:
            filtered = filtered.sort_values(col_country, ascending=True)

        # ---- Chart ----
        if filtered.empty:
            st.info("No countries selected.")
        else:
            fig = px.bar(
                filtered,
                x=col_country,
                y=col_enroll,
                height=430,
                labels={col_country: "Country", col_enroll: "Enrollments"},
                title="Enrollments for selected countries",
            )
            st.plotly_chart(fig, use_container_width=True, key="enrollments_by_country")
        st.caption(f"Data source: {path_u.relative_to(ROOT) if path_u else 'N/A'}")
    else:
        st.info("Add `country_enrollment_summary.csv` to `data/analysis-outputs/`.")

    st.markdown("---")
    st.markdown("### Assessment Outcomes by Delivery Mode")
    df_a, path_a = read_any("course_assessment_by_course.csv")
    if df_a is None:
        df_a, path_a = read_any("Course_wise_assessment.csv")

    if df_a is not None and not df_a.empty:
        # Prefer course names
        col_course = best_course_column(df_a)
        # Build a human display list (dedup + sorted)
        course_options = sorted(df_a[col_course].dropna().astype(str).unique())
        course_sel = st.selectbox("Course", course_options)
        col_delivery = guess_col(df_a, ["delivery", "mode", "format", "delivery_mode"])
        # Pick outcome metric in clear language
        metric_pick = st.radio("Outcome", ["Change in Proficiency", "Change in Applications"], horizontal=True)

        # Numeric columns (robust)
        col_dprof = guess_col(df_a, ["delta_proficiency", "prof_delta", "proficiency_delta"], prefer_text=False) or df_a.columns[-2]
        col_dapps = guess_col(df_a, ["delta_applications", "apps_delta", "applications_delta"], prefer_text=False) or df_a.columns[-1]
        for c in [col_dprof, col_dapps]:
            df_a[c] = pd.to_numeric(df_a[c], errors="coerce")

        sub = df_a[df_a[col_course].astype(str) == str(course_sel)]
        ycol = col_dprof if metric_pick == "Change in Proficiency" else col_dapps
        fig_box = px.box(
            sub, x=col_delivery, y=ycol, points="all", height=400,
            labels={col_delivery: "Delivery mode", ycol: metric_pick},
            title=f"{metric_pick} by delivery mode — {course_sel}",
        )
        st.plotly_chart(fig_box, use_container_width=True, key="assessment_delivery")
        st.caption(f"Data source: {path_a.relative_to(ROOT) if path_a else 'N/A'}")
    else:
        st.info("Add `course_assessment_by_course.csv` to `data/analysis-outputs/`.")

# ==================== TAB 2: EMPLOYEE SEGMENTS (PCA + KMEANS) ====================
with tab2:
    st.markdown("### Top Features for PCA Components")
    st.caption("Which features contribute the most to each component. Useful to understand what drives segmentation.")
    comp, path_comp = read_any("pca_components.xlsx")
    if comp is None:
        comp, path_comp = read_any("pca_components.csv")

    if comp is not None and not comp.empty:
        # Prepare tidy loadings
        first = comp.columns[0]
        if first.lower() not in {"feature", "variable"}:
            comp = comp.rename(columns={first: "feature"})
        if "component" not in [c.lower() for c in comp.columns]:
            tidy = comp.melt(id_vars="feature", var_name="component", value_name="loading")
        else:
            cname = [c for c in comp.columns if c.lower() == "component"][0]
            tidy = comp.rename(columns={cname: "component"})
            if "loading" not in tidy.columns:
                other = [c for c in tidy.columns if c not in {"feature", "component"}]
                if other:
                    tidy = tidy.rename(columns={other[-1]: "loading"})

        # Map Q-codes to full questions if mapping exists
        qmap = load_survey_mapping()
        tidy["feature"] = apply_question_mapping(tidy["feature"], qmap)

        # Choose component and show top-N by absolute loading
        comp_sel = st.selectbox("Component", sorted(tidy["component"].astype(str).unique()))
        topN = st.slider("How many top features to show", 5, 30, 15)
        sub = tidy[tidy["component"].astype(str) == str(comp_sel)].copy()
        sub["abs_loading"] = sub["loading"].abs()
        sub = sub.sort_values("abs_loading", ascending=False).head(topN)

        fig_pca = px.bar(
            sub, x="feature", y="loading", height=430,
            labels={"feature": "Feature (question/metric)", "loading": "Importance (loading)"},
            title=f"Top {topN} features for component {comp_sel}",
        )
        st.plotly_chart(fig_pca, use_container_width=True, key="pca_top_features")
        st.caption(f"Data source: {path_comp.relative_to(ROOT) if path_comp else 'N/A'}")
    else:
        st.info("Add `pca_components.xlsx` (or .csv) to `data/analysis-outputs/`.")

    st.markdown("---")
    st.markdown("### Segment Distribution by Location")
    seg, path_seg = read_any("pca_kmeans_results.xlsx")
    if seg is None:
        seg, path_seg = read_any("pca_kmeans_results.csv")

    if seg is not None and not seg.empty:
        col_seg = guess_col(seg, ["segment", "cluster", "group", "label"])
        loc_col = best_location_column(seg)

        # Let users optionally narrow to specific locations
        # Default: show all; if selection made, filter to chosen ones
        all_locs = sorted(seg[loc_col].dropna().astype(str).unique())
        pick_locs = st.multiselect("Locations (optional)", options=all_locs, default=[])
        view = seg if not pick_locs else seg[seg[loc_col].astype(str).isin(pick_locs)]

        fig_seg = px.histogram(
            view, x=col_seg, color=col_seg, height=400,
            labels={col_seg: "Segment/cluster"},
            title="Distribution of employee segments",
        )
        st.plotly_chart(fig_seg, use_container_width=True, key="segment_distribution")
        st.caption(f"Data source: {path_seg.relative_to(ROOT) if path_seg else 'N/A'}")
    else:
        st.info("Add `pca_kmeans_results.xlsx` (or .csv) to `data/analysis-outputs/`.")

# ==================== TAB 3: PROGRAM IMPROVEMENTS (EXPERIMENT) ====================
with tab3:
    st.markdown("### Pre → Post Program Improvements")
    exp, path_exp = read_any("experiment_curriculum_cleaned.csv")
    if exp is None:
        exp, path_exp = read_any("nls_experiment_cleaned.csv")

    if exp is not None and not exp.empty:
        col_prog  = guess_col(exp, ["program", "curriculum", "group"])
        pre_p     = guess_col(exp, ["pre_proficiency", "proficiency_pre", "pre_prof"], prefer_text=False)
        post_p    = guess_col(exp, ["post_proficiency", "proficiency_post", "post_prof"], prefer_text=False)
        pre_a     = guess_col(exp, ["pre_applications", "applications_pre", "pre_apps"], prefer_text=False)
        post_a    = guess_col(exp, ["post_applications", "applications_post", "post_apps"], prefer_text=False)

        # Make deltas
        exp["Δ proficiency"] = pd.to_numeric(exp[post_p], errors="coerce") - pd.to_numeric(exp[pre_p], errors="coerce")
        exp["Δ applications"] = pd.to_numeric(exp[post_a], errors="coerce") - pd.to_numeric(exp[pre_a], errors="coerce")

        metric = st.radio("Outcome", ["Δ proficiency", "Δ applications"], horizontal=True)
        # Optional program filter (multiselect); default all
        progs = sorted(exp[col_prog].dropna().astype(str).unique())
        pick_progs = st.multiselect("Programs to include (optional)", options=progs, default=[])

        view = exp if not pick_progs else exp[exp[col_prog].astype(str).isin(pick_progs)]
        fig_exp = px.box(
            view, x=col_prog, y=metric, color=col_prog, points="all", height=430,
            labels={col_prog: "Program", metric: metric},
            title=f"{metric} — by program",
        )
        st.plotly_chart(fig_exp, use_container_width=True, key="experiment_deltas")
        st.caption(f"Data source: {path_exp.relative_to(ROOT) if path_exp else 'N/A'}")
    else:
        st.info("Add `experiment_curriculum_cleaned.csv` to `data/analysis-outputs/`.")
