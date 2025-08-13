# streamlit_app/workforce_analytics_dashboard.py
from pathlib import Path
import re
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

# Subtle CSS
st.markdown("""
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
.section-gap { margin-top: .75rem; margin-bottom: .75rem; }
</style>
""", unsafe_allow_html=True)

# ---------------- Survey QID -> text mapping (embedded) ----------------
Q_MAPPING = {
    "Q1":  "I prefer training that helps me enhance my job performance, without the need for group interaction.",
    "Q2":  "I’m motivated to learn new things that expand my abilities and knowledge beyond my day-to-day responsibilities.",
    "Q3":  "I am more interested in training that enhances my individual skills than in activities focused on team-building.",
    "Q4":  "I’m not interested in using training for networking or career changes; I value it more for improving my day-to-day work.",
    "Q5":  "I enjoy training that broadens my knowledge and helps me grow, even if it’s not directly related to my current role.",
    "Q6":  "I am motivated by training that not only helps me perform better in my current role but also enhances my overall skills and knowledge.",
    "Q7":  "I value training that makes me more effective in my job while also helping me grow professionally.",
    "Q8":  "I find value in training that supports my personal development and inspires me to grow in new directions.",
    "Q9":  "I look for training opportunities that allow me to develop my role-specific skills and learn new concepts that can help me in the future.",
    "Q10": "I’m more interested in improving my current role than in preparing for a new position or career change.",
    "Q11": "I find the most value in training that directly impacts my role, rather than in sessions involving group discussions.",
    "Q12": "I don’t see training as a way to transition into a new job; I prefer to use it to build on what I’m already doing.",
}

# ---------------- Paths & robust loaders ----------------
ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [
    ROOT / "data" / "analysis-outputs",
    ROOT / "data" / "processed",
    ROOT / "data" / "raw",   # optional for extra maps, if present
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

def guess_col(df: pd.DataFrame, candidates, *, prefer_text=True):
    """Return first matching candidate (case-insensitive). Fallback to first text col, else first col."""
    name_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = name_map.get(str(cand).lower())
        if c is not None:
            if prefer_text and not pd.api.types.is_object_dtype(df[c]):
                continue
            return c
    if prefer_text:
        for c in df.columns:
            if pd.api.types.is_object_dtype(df[c]):
                return c
    return df.columns[0] if len(df.columns) else None

def best_course_column(df: pd.DataFrame):
    for cand in ["course_title", "course_name"]:
        for col in df.columns:
            if col.lower() == cand:
                return col
    for cand in ["course", "course_id"]:
        for col in df.columns:
            if col.lower() == cand:
                return col
    return df.columns[0]

def looks_textual_location(s: pd.Series) -> bool:
    s = s.dropna().astype(str)
    if s.empty: return False
    pct_letters = (s.str.contains(r"[A-Za-z]", regex=True)).mean()
    unique_ratio = s.nunique() / max(1, len(s))
    return pct_letters >= 0.7 and unique_ratio <= 0.9

def best_location_column(df: pd.DataFrame):
    for cand in ["location", "city", "office", "region", "country"]:
        for col in df.columns:
            if col.lower() == cand and looks_textual_location(df[col]):
                return col
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) and looks_textual_location(df[col]):
            return col
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            return col
    return df.columns[0]

def apply_q_mapping(series: pd.Series) -> pd.Series:
    """Replace Qn with full text using embedded Q_MAPPING. Falls back to 'Question n' if unseen."""
    def repl(x):
        s = str(x).strip()
        key = s.upper()
        if key in Q_MAPPING:
            return Q_MAPPING[key]
        m = re.match(r"^(Q\d+)", key)
        if m and m.group(1) in Q_MAPPING:
            return Q_MAPPING[m.group(1)]
        m2 = re.match(r"^Q(\d+)", key)
        if m2:
            return f"Question {m2.group(1)}"
        return s
    return series.map(repl)

# ---------------- Header ----------------
st.title("Workforce Analytics — Interactive Insights")
st.caption("Explore training enrollments, employee segments (PCA + KMeans), and curriculum experiment outcomes.")

# ---------------- KPIs (useful) ----------------
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
# Experiment
exp_df, _ = read_any("experiment_curriculum_cleaned.csv")
if exp_df is None:
    exp_df, _ = read_any("nls_experiment_cleaned.csv")
if exp_df is not None and not exp_df.empty:
    pre_p  = guess_col(exp_df, ["pre_proficiency", "proficiency_pre", "pre_prof"], prefer_text=False)
    post_p = guess_col(exp_df, ["post_proficiency", "proficiency_post", "post_prof"], prefer_text=False)
    if pre_p and post_p:
        imp = pd.to_numeric(exp_df[post_p], errors="coerce") - pd.to_numeric(exp_df[pre_p], errors="coerce")
        if not imp.dropna().empty:
            kpi_vals["Median proficiency improvement"] = f"{imp.median():.2f}"

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

        # Multiselect of country names (default top 10 by enrollments)
        top10 = df_u.sort_values(col_enroll, ascending=False)[col_country].astype(str).head(10).tolist()
        countries = st.multiselect(
            "Countries to include (default: top 10 by enrollments)",
            options=sorted(df_u[col_country].astype(str).unique()),
            default=top10,
        )
        sort_by = st.radio("Sort bars by", ["Enrollments (descending)", "Country (A–Z)"], horizontal=True)
        filtered = df_u[df_u[col_country].astype(str).isin(countries)] if countries else df_u.copy()
        filtered = filtered.sort_values(col_enroll if sort_by.startswith("Enrollments") else col_country,
                                        ascending=not sort_by.startswith("Enrollments"))

        if filtered.empty:
            st.info("No countries selected.")
        else:
            fig = px.bar(
                filtered,
                x=col_country, y=col_enroll,
                height=430,
                labels={col_country: "Country", col_enroll: "Enrollments"},
                title="Enrollments for selected countries",
            )
            st.plotly_chart(fig, use_container_width=True, key="enrollments_by_country")
        st.caption(f"Data source: {path_u.relative_to(ROOT) if path_u else 'N/A'}")
    else:
        st.info("Add `country_enrollment_summary.csv` to `data/analysis-outputs/`.")

    st.markdown("### Assessment Outcomes by Delivery Mode")
    df_a, path_a = read_any("course_assessment_by_course.csv")
    if df_a is None:
        df_a, path_a = read_any("Course_wise_assessment.csv")

    if df_a is not None and not df_a.empty:
        # Prefer course names; normalize
        col_course = best_course_column(df_a)
        df_a[col_course] = df_a[col_course].astype(str).str.strip()

        course_options = sorted(df_a[col_course].dropna().unique())
        course_sel = st.selectbox("Course", course_options)

        col_delivery = guess_col(df_a, ["delivery", "mode", "format", "delivery_mode"])
        metric_pick = st.radio("Outcome", ["Change in Proficiency", "Change in Applications"], horizontal=True)

        # Robust numeric metrics
        col_dprof = guess_col(df_a, ["delta_proficiency", "prof_delta", "proficiency_delta"], prefer_text=False)
        col_dapps = guess_col(df_a, ["delta_applications", "apps_delta", "applications_delta"], prefer_text=False)
        for c in [col_dprof, col_dapps]:
            df_a[c] = pd.to_numeric(df_a[c], errors="coerce")

        # Filter safely
        mask = df_a[col_course].str.casefold() == str(course_sel).casefold()
        sub = df_a[mask].copy()

        if sub.empty:
            st.warning("No data for the selected course after filtering. Try a different course.")
        else:
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
    st.markdown("### Top Questions/Features by Component Influence")
    st.caption("Shows which questions or metrics contribute most to each component (higher |loading| ⇒ stronger influence).")

    comp, path_comp = read_any("pca_components.xlsx")
    if comp is None:
        comp, path_comp = read_any("pca_components.csv")

    if comp is not None and not comp.empty:
        first = comp.columns[0]
        if first.lower() not in {"feature", "variable"}:
            comp = comp.rename(columns={first: "feature"})

        # Wide→long; ensure numeric loading
        if "component" not in [c.lower() for c in comp.columns]:
            tidy = comp.melt(id_vars="feature", var_name="component", value_name="loading")
        else:
            cname = [c for c in comp.columns if c.lower() == "component"][0]
            tidy = comp.rename(columns={cname: "component"})
        tidy["loading"] = pd.to_numeric(tidy["loading"], errors="coerce")
        tidy = tidy.dropna(subset=["loading"])

        # Replace Q-codes with full text
        tidy["feature"] = apply_q_mapping(tidy["feature"])

        comps = sorted(tidy["component"].astype(str).unique())
        comp_sel = st.selectbox("Component", comps)

        topN = st.slider("Number of top questions/features", 5, 30, 15)
        sub = tidy[tidy["component"].astype(str) == str(comp_sel)].copy()
        if sub.empty:
            st.info("No numeric loadings for this component.")
        else:
            sub["abs_loading"] = sub["loading"].abs()
            sub = sub.sort_values("abs_loading", ascending=False).head(topN)
            fig_pca = px.bar(
                sub, x="feature", y="loading", height=430,
                labels={"feature": "Question/Feature", "loading": "Loading"},
                title=f"Top {topN} questions/features — component {comp_sel}",
            )
            st.plotly_chart(fig_pca, use_container_width=True, key="pca_top_features")
        st.caption(f"Data source: {path_comp.relative_to(ROOT) if path_comp else 'N/A'}")
    else:
        st.info("Add `pca_components.xlsx` (or .csv) to `data/analysis-outputs/`.")

    st.markdown("### Segment Distribution by Location")
    seg, path_seg = read_any("pca_kmeans_results.xlsx")
    if seg is None:
        seg, path_seg = read_any("pca_kmeans_results.csv")

    if seg is not None and not seg.empty:
        col_seg = guess_col(seg, ["segment", "cluster", "group", "label"])
        loc_col = best_location_column(seg)

        all_locs = sorted(seg[loc_col].dropna().astype(str).unique())
        pick_locs = st.multiselect("Locations (optional)", options=all_locs, default=[])
        view = seg if not pick_locs else seg[seg[loc_col].astype(str).isin(pick_locs)]

        if view.empty:
            st.info("No rows match the selected locations.")
        else:
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
        st.caption("Choose the correct columns if the defaults look off.")
        # Let user confirm program & metric fields (pre/post) with safe defaults
        prog_guess  = guess_col(exp, ['program','curriculum','group'])
        pre_p_guess = guess_col(exp, ['pre_proficiency','proficiency_pre','pre_prof'], prefer_text=False)
        post_p_guess= guess_col(exp, ['post_proficiency','proficiency_post','post_prof'], prefer_text=False)
        pre_a_guess = guess_col(exp, ['pre_applications','applications_pre','pre_apps'], prefer_text=False)
        post_a_guess= guess_col(exp, ['post_applications','applications_post','post_apps'], prefer_text=False)

        col_prog  = st.selectbox("Program field", list(exp.columns),
                                 index=list(exp.columns).index(prog_guess) if prog_guess in exp.columns else 0)
        col_pre_p = st.selectbox("Pre proficiency", list(exp.columns),
                                 index=list(exp.columns).index(pre_p_guess) if pre_p_guess in exp.columns else 0)
        col_post_p= st.selectbox("Post proficiency", list(exp.columns),
                                 index=list(exp.columns).index(post_p_guess) if post_p_guess in exp.columns else 0)
        col_pre_a = st.selectbox("Pre applications", list(exp.columns),
                                 index=list(exp.columns).index(pre_a_guess) if pre_a_guess in exp.columns else 0)
        col_post_a= st.selectbox("Post applications", list(exp.columns),
                                 index=list(exp.columns).index(post_a_guess) if post_a_guess in exp.columns else 0)

        # Compute deltas robustly
        exp = exp.copy()
        exp["Δ proficiency"]  = pd.to_numeric(exp[col_post_p], errors="coerce") - pd.to_numeric(exp[col_pre_p], errors="coerce")
        exp["Δ applications"] = pd.to_numeric(exp[col_post_a], errors="coerce") - pd.to_numeric(exp[col_pre_a], errors="coerce")

        metric = st.radio("Outcome", ["Δ proficiency", "Δ applications"], horizontal=True)
        progs = sorted(exp[col_prog].dropna().astype(str).unique())
        pick = st.multiselect("Programs to include (optional)", options=progs, default=[])

        view = exp if not pick else exp[exp[col_prog].astype(str).isin(pick)]
        view_numeric = view.dropna(subset=[metric])
        if view_numeric.empty:
            st.warning("No numeric results to plot for the chosen fields. Verify the selected columns.")
        else:
            fig_exp = px.box(
                view_numeric, x=col_prog, y=metric, color=col_prog, points="all", height=430,
                labels={col_prog: "Program", metric: metric},
                title=f"{metric} — by program",
            )
            st.plotly_chart(fig_exp, use_container_width=True, key="experiment_deltas")
        st.caption(f"Data source: {path_exp.relative_to(ROOT) if path_exp else 'N/A'}")
    else:
        st.info("Add `experiment_curriculum_cleaned.csv` to `data/analysis-outputs/`.")
