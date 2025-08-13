# streamlit_app/workforce_analytics_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Workforce Analytics — Interactive Insights",
    page_icon="📊",
    layout="wide",
    menu_items={"About": "Workforce Analytics — EDA · PCA · KMeans · Experiment"},
)

# Small CSS polish (spacing + subtle control boxes)
st.markdown("""
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
.section-box { background: rgba(0,0,0,0.03); padding: .6rem .8rem; border-radius: .5rem; }
</style>
""", unsafe_allow_html=True)

# ---------------- Paths & robust loaders ----------------
ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [ROOT / "data" / "analysis-outputs", ROOT / "data" / "processed"]

@st.cache_data(show_spinner=False)
def find_and_read(name: str):
    """
    Search known folders and read CSV/XLSX.
    Returns (df, path) or (None, None) if not found.
    """
    for base in SEARCH_DIRS:
        p = base / name
        if p.exists():
            if p.suffix.lower() == ".csv":
                return pd.read_csv(p, low_memory=False), p
            if p.suffix.lower() in (".xlsx", ".xls"):
                return pd.read_excel(p), p
    return None, None

def guess_col(df: pd.DataFrame, candidates):
    m = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in m:
            return m[cand.lower()]
    return None

def choose_col(df: pd.DataFrame, label: str, candidates, help_text=""):
    g = guess_col(df, candidates) or df.columns[0]
    return st.selectbox(label, list(df.columns), index=list(df.columns).index(g), help=help_text)

def safe_topn_slider(n_items: int, label="How many countries to display"):
    """Avoids Streamlit slider errors when dataset is small."""
    n_items = int(max(0, n_items))
    if n_items <= 1:
        st.info("Not enough rows to render a ranking.")
        return 1
    min_v, max_v = 1, min(25, n_items)
    value = min(10, max_v)
    return st.slider(label, min_v, max_v, value)

# ---------------- Header ----------------
st.title("Workforce Analytics — Interactive Insights")
st.caption("Explore training **enrollments**, employee **segments** (PCA + KMeans), and **curriculum experiment** outcomes.")

# ---------------- KPI row (explicit & friendly) ----------------
kpi_df, _ = find_and_read("country_enrollment_summary.csv")
if kpi_df is None:
    kpi_df, _ = find_and_read("Country-wise_Enrollment_Summary.csv")

c1, c2, c3 = st.columns(3)
if kpi_df is not None and not kpi_df.empty:
    col_country = guess_col(kpi_df, ["country", "country_name", "nation"]) or kpi_df.columns[0]
    col_enroll  = guess_col(kpi_df, ["enrollments", "enrollment", "total_enrollments", "count"]) or kpi_df.columns[1]
    kpi_df[col_enroll] = pd.to_numeric(kpi_df[col_enroll], errors="coerce")
    kpi_df = kpi_df.dropna(subset=[col_enroll])
    if not kpi_df.empty:
        total_enroll = int(kpi_df[col_enroll].sum())
        top_idx = kpi_df[col_enroll].idxmax()
        top_country = str(kpi_df.loc[top_idx, col_country])
        top_value   = int(kpi_df.loc[top_idx, col_enroll])
        c1.metric("Total enrollments (all countries)", f"{total_enroll:,}")
        c2.metric("Country with highest enrollments", top_country)
        c3.metric("Highest enrollments for one country", f"{top_value:,}")

st.markdown("---")

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs([
    "📍 Enrollments by Country",
    "🧩 Segments (PCA + KMeans)",
    "🧪 Curriculum Experiment",
])

# ==================== TAB 1: ENROLLMENTS BY COUNTRY ====================
with tab1:
    st.markdown("### Enrollments by Country")
    df_u, path_u = find_and_read("country_enrollment_summary.csv")
    if df_u is None:
        df_u, path_u = find_and_read("Country-wise_Enrollment_Summary.csv")

    if df_u is not None and not df_u.empty:
        # Controls tied to this chart
        with st.container():
            st.markdown('<div class="section-box"><b>Filters for this chart</b></div>', unsafe_allow_html=True)
            colA, colB = st.columns([2, 1.2])
            with colA:
                col_country = choose_col(df_u, "Country column", ["country", "country_name", "nation"])
            with colB:
                col_enroll  = choose_col(
                    df_u, "Enrollments column",
                    ["enrollments", "enrollment", "total_enrollments", "count"],
                    "Numeric enrollments per country"
                )
        # Prepare data
        df_u[col_enroll] = pd.to_numeric(df_u[col_enroll], errors="coerce")
        df_u = df_u.dropna(subset=[col_enroll])
        topn = safe_topn_slider(len(df_u), label="How many countries to display (by enrollments)")

        # Chart
        view = df_u.sort_values(col_enroll, ascending=False).head(topn)
        fig = px.bar(view, x=col_country, y=col_enroll, height=430,
                     labels={col_country: "Country", col_enroll: "Enrollments"},
                     title=f"Top {topn} countries by enrollments")
        st.plotly_chart(fig, use_container_width=True, key="chart_enrollments_by_country")
        st.caption(f"Data source: {path_u.relative_to(ROOT) if path_u else 'N/A'}")
    else:
        st.info("Add `country_enrollment_summary.csv` to `data/analysis-outputs/`.")

    st.markdown("---")
    st.markdown("### Assessment Outcomes by Delivery Mode")
    df_a, path_a = find_and_read("course_assessment_by_course.csv")
    if df_a is None:
        df_a, path_a = find_and_read("Course_wise_assessment.csv")

    if df_a is not None and not df_a.empty:
        # Controls for this chart
        with st.container():
            st.markdown('<div class="section-box"><b>Filters for this chart</b></div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1.6, 1.2, 1.2])
            with c1:
                col_course = choose_col(df_a, "Course name column",
                                        ["course_title", "course_name", "course", "course_id"])
                # Prefer names over IDs where available
                course_values = df_a[col_course].dropna().astype(str).unique()
                course_values.sort()
                course_pick = st.selectbox("Show results for course", course_values)
            with c2:
                col_delivery = choose_col(df_a, "Delivery column",
                                          ["delivery", "mode", "format", "delivery_mode"])
            with c3:
                metric_pick = st.radio(
                    "Outcome metric",
                    ["Change in Proficiency", "Change in Applications"],  # plain language
                    horizontal=False,
                    index=0
                )
        # Numeric columns (robust guessing)
        col_dprof = guess_col(df_a, ["delta_proficiency", "prof_delta", "proficiency_delta"]) or df_a.columns[-2]
        col_dapps = guess_col(df_a, ["delta_applications", "apps_delta", "applications_delta"]) or df_a.columns[-1]
        for c in [col_dprof, col_dapps]:
            df_a[c] = pd.to_numeric(df_a[c], errors="coerce")

        sub = df_a[df_a[col_course].astype(str) == str(course_pick)]
        ycol = col_dprof if metric_pick.startswith("Change in Proficiency") else col_dapps
        fig_box = px.box(sub, x=col_delivery, y=ycol, points="all", height=400,
                         labels={col_delivery: "Delivery mode", ycol: metric_pick})
        st.plotly_chart(fig_box, use_container_width=True, key="chart_assessment_outcomes")
        st.caption(f"Data source: {path_a.relative_to(ROOT) if path_a else 'N/A'}")
    else:
        st.info("Add `course_assessment_by_course.csv` to `data/analysis-outputs/`.")

# ==================== TAB 2: SEGMENTS (PCA + KMEANS) ====================
with tab2:
    st.markdown("### Principal Component Loadings")
    comp, path_comp = find_and_read("pca_components.xlsx")
    if comp is None:
        comp, path_comp = find_and_read("pca_components.csv")

    if comp is not None and not comp.empty:
        # Controls for this chart
        with st.container():
            st.markdown('<div class="section-box"><b>Filters for this chart</b></div>', unsafe_allow_html=True)
            first = comp.columns[0]
            if first.lower() not in {"feature", "variable"}:
                comp = comp.rename(columns={first: "feature"})
            # Wide -> tidy if needed
            if "component" not in [c.lower() for c in comp.columns]:
                tidy = comp.melt(id_vars="feature", var_name="component", value_name="loading")
            else:
                cname = [c for c in comp.columns if c.lower() == "component"][0]
                tidy = comp.rename(columns={cname: "component"})
                if "loading" not in tidy.columns:
                    other = [c for c in tidy.columns if c not in {"feature", "component"}]
                    if other:
                        tidy = tidy.rename(columns={other[-1]: "loading"})
            comp_sel = st.selectbox("Component to inspect", sorted(tidy["component"].astype(str).unique()))
        # Chart
        fig_pca = px.bar(
            tidy[tidy["component"].astype(str) == str(comp_sel)],
            x="feature", y="loading", height=430,
            labels={"feature": "Feature", "loading": "Loading"},
            title=f"Top loadings for component {comp_sel}"
        )
        st.plotly_chart(fig_pca, use_container_width=True, key="chart_pca_loadings")
        st.caption(f"Data source: {path_comp.relative_to(ROOT) if path_comp else 'N/A'}")
    else:
        st.info("Add `pca_components.xlsx` (or .csv) to `data/analysis-outputs/`.")

    st.markdown("---")
    st.markdown("### Cluster Distribution by Location")
    seg, path_seg = find_and_read("pca_kmeans_results.xlsx")
    if seg is None:
        seg, path_seg = find_and_read("pca_kmeans_results.csv")

    if seg is not None and not seg.empty:
        # Controls for this chart
        with st.container():
            st.markdown('<div class="section-box"><b>Filters for this chart</b></div>', unsafe_allow_html=True)
            col_loc = choose_col(seg, "Location column", ["location", "office", "city", "region"])
            col_seg = choose_col(seg, "Segment/cluster column", ["segment", "cluster", "group", "label"])
            loc_pick = st.selectbox("Filter by location", ["All"] + sorted(seg[col_loc].dropna().astype(str).unique()))
        # Chart
        view = seg if loc_pick == "All" else seg[seg[col_loc].astype(str) == str(loc_pick)]
        fig_seg = px.histogram(view, x=col_seg, color=col_seg, height=400,
                               labels={col_seg: "Segment/cluster"},
                               title="Segments distribution")
        st.plotly_chart(fig_seg, use_container_width=True, key="chart_segment_distribution")
        st.caption(f"Data source: {path_seg.relative_to(ROOT) if path_seg else 'N/A'}")
    else:
        st.info("Add `pca_kmeans_results.xlsx` (or .csv) to `data/analysis-outputs/`.")

# ==================== TAB 3: CURRICULUM EXPERIMENT ====================
with tab3:
    st.markdown("### Program Improvements (A/B vs Current)")
    exp, path_exp = find_and_read("experiment_curriculum_cleaned.csv")
    if exp is None:
        exp, path_exp = find_and_read("nls_experiment_cleaned.csv")  # legacy name

    if exp is not None and not exp.empty:
        # Controls for this chart
        with st.container():
            st.markdown('<div class="section-box"><b>Filters for this chart</b></div>', unsafe_allow_html=True)
            col_prog  = choose_col(exp, "Program column", ["program", "curriculum", "group"])
            col_pre_p = choose_col(exp, "Pre proficiency",  ["pre_proficiency", "proficiency_pre", "pre_prof"])
            col_post_p= choose_col(exp, "Post proficiency", ["post_proficiency", "proficiency_post", "post_prof"])
            col_pre_a = choose_col(exp, "Pre applications", ["pre_applications", "applications_pre", "pre_apps"])
            col_post_a= choose_col(exp, "Post applications",["post_applications", "applications_post", "post_apps"])
            metric_sel = st.radio("Outcome to compare", ["Change in Proficiency", "Change in Applications"],
                                  horizontal=True, index=0)

        # Compute deltas robustly
        exp["delta_proficiency"] = pd.to_numeric(exp[col_post_p], errors="coerce") - pd.to_numeric(exp[col_pre_p], errors="coerce")
        exp["delta_applications"] = pd.to_numeric(exp[col_post_a], errors="coerce") - pd.to_numeric(exp[col_pre_a], errors="coerce")
        ycol = "delta_proficiency" if metric_sel.startswith("Change in Proficiency") else "delta_applications"

        # Chart
        fig_exp = px.box(exp, x=col_prog, y=ycol, points="all", color=col_prog, height=430,
                         labels={col_prog: "Program", ycol: metric_sel},
                         title="Pre → Post improvement by program")
        st.plotly_chart(fig_exp, use_container_width=True, key="chart_experiment_deltas")
        st.caption(f"Data source: {path_exp.relative_to(ROOT) if path_exp else 'N/A'}")
    else:
        st.info("Add `experiment_curriculum_cleaned.csv` to `data/analysis-outputs/`.")
