# streamlit_app/workforce_analytics_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------- App config ---------
st.set_page_config(
    page_title="Workforce Analytics — Interactive Insights",
    page_icon="📊",
    layout="wide",
    menu_items={"About": "Workforce Analytics — EDA · PCA · KMeans · Experiment"},
)

# Gentle CSS for spacing
st.markdown("""
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
.section-controls { background: rgba(0,0,0,0.02); padding: .6rem .8rem; border-radius: .5rem; }
.section-title { margin-bottom: .4rem; }
</style>
""", unsafe_allow_html=True)

# --------- Paths & readers ---------
ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [ROOT / "data" / "analysis-outputs", ROOT / "data" / "processed"]

@st.cache_data(show_spinner=False)
def find_and_read(name: str):
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

def safe_slider(label: str, n_items: int, default_top: int = 10, hard_cap: int = 25):
    n_items = int(max(0, n_items))
    if n_items <= 1:
        st.info("Not enough rows to show a Top-N chart.")
        return 1
    s_min = 1
    s_max = min(hard_cap, n_items)
    s_val = min(default_top, s_max)
    return st.slider(label, s_min, s_max, s_val)

# --------- Header ---------
st.title("Workforce Analytics — Interactive Insights")
st.caption("Explore training **enrollments**, employee **segments** (PCA + KMeans), and **curriculum experiment** outcomes.")

# --------- KPIs (clearly labeled) ---------
util_df, _ = find_and_read("country_enrollment_summary.csv")
if util_df is None:
    util_df, _ = find_and_read("Country-wise_Enrollment_Summary.csv")

k1, k2, k3 = st.columns(3)
if util_df is not None and not util_df.empty:
    c_country = guess_col(util_df, ["country", "country_name", "nation"]) or util_df.columns[0]
    c_enroll  = guess_col(util_df, ["enrollments", "enrollment", "total_enrollments", "count"]) or util_df.columns[1]
    util_df[c_enroll] = pd.to_numeric(util_df[c_enroll], errors="coerce")
    util_df = util_df.dropna(subset=[c_enroll])
    if not util_df.empty:
        total_enroll = int(util_df[c_enroll].sum())
        top_idx = util_df[c_enroll].idxmax()
        top_country = util_df.loc[top_idx, c_country]
        top_value   = int(util_df.loc[top_idx, c_enroll])
        k1.metric("Total enrollments (all countries)", f"{total_enroll:,}")
        k2.metric("Top country by enrollments", str(top_country))
        k3.metric("Max enrollments for a country", f"{top_value:,}")

st.markdown("---")

# --------- Tabs ---------
tab1, tab2, tab3 = st.tabs(["📍 Enrollments by Country", "🧩 Segments (PCA + KMeans)", "🧪 Curriculum Experiment"])

# ==================== TAB 1: ENROLLMENTS ====================
with tab1:
    st.markdown("### Enrollments by Country", help="Shows training enrollments aggregated per country.")
    df_u, p_u = find_and_read("country_enrollment_summary.csv")
    if df_u is None:
        df_u, p_u = find_and_read("Country-wise_Enrollment_Summary.csv")

    if df_u is not None and not df_u.empty:
        # --- Controls directly ABOVE the chart it affects ---
        with st.container():
            st.markdown('<div class="section-controls">**Controls**</div>', unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1.2])
            with col1:
                col_country = choose_col(df_u, "Country column", ["country", "country_name", "nation"])
            with col2:
                col_enroll  = choose_col(df_u, "Enrollments column",
                                         ["enrollments", "enrollment", "total_enrollments", "count"],
                                         "Numeric enrollments per country")
        df_u[col_enroll] = pd.to_numeric(df_u[col_enroll], errors="coerce")
        df_u = df_u.dropna(subset=[col_enroll])

        # Safe Top-N (avoids slider crash)
        topn = safe_slider("Top N countries", len(df_u))

        # --- Chart ---
        view = df_u.sort_values(col_enroll, ascending=False).head(topn)
        fig = px.bar(view, x=col_country, y=col_enroll, height=430)
        st.plotly_chart(fig, use_container_width=True, key="util_bar")

        st.caption(f"Data source: {p_u.relative_to(ROOT) if p_u else 'N/A'}")
    else:
        st.info("Add `country_enrollment_summary.csv` to `data/analysis-outputs/`.")

    st.markdown("---")
    st.markdown("### Assessment Gains by Delivery Mode", help="Compare Δ proficiency/applications by delivery for a selected course.")
    df_a, p_a = find_and_read("course_assessment_by_course.csv")
    if df_a is None:
        df_a, p_a = find_and_read("Course_wise_assessment.csv")

    if df_a is not None and not df_a.empty:
        # --- Controls for this chart ---
        with st.container():
            st.markdown('<div class="section-controls">**Controls**</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1.4, 1.0, 1.0])
            with c1:
                col_course   = choose_col(df_a, "Course",   ["course_title","course","course_name","course_id"])
                course_sel = st.selectbox("Select course", sorted(df_a[col_course].dropna().astype(str).unique()))
            with c2:
                col_delivery = choose_col(df_a, "Delivery", ["delivery","mode","format","delivery_mode"])
            with c3:
                metric_pick = st.radio("Metric", ["Δ Proficiency", "Δ Applications"], horizontal=False)
        col_dprof = guess_col(df_a, ["delta_proficiency","prof_delta","proficiency_delta"]) or df_a.columns[-2]
        col_dapp  = guess_col(df_a, ["delta_applications","apps_delta","applications_delta"]) or df_a.columns[-1]
        for c in [col_dprof, col_dapp]:
            df_a[c] = pd.to_numeric(df_a[c], errors="coerce")

        sub = df_a[df_a[col_course].astype(str) == str(course_sel)]
        ycol = col_dprof if metric_pick == "Δ Proficiency" else col_dapp
        fig_box = px.box(sub, x=col_delivery, y=ycol, points="all", height=400)
        st.plotly_chart(fig_box, use_container_width=True, key="assess_box")
        st.caption(f"Data source: {p_a.relative_to(ROOT) if p_a else 'N/A'}")
    else:
        st.info("Add `course_assessment_by_course.csv` to `data/analysis-outputs/`.")

# ==================== TAB 2: SEGMENTS ====================
with tab2:
    st.markdown("### PCA Component Loadings", help="Which features drive each principal component the most.")
    comp, pcomp = find_and_read("pca_components.xlsx")
    if comp is None:
        comp, pcomp = find_and_read("pca_components.csv")

    if comp is not None and not comp.empty:
        # Controls
        with st.container():
            st.markdown('<div class="section-controls">**Controls**</div>', unsafe_allow_html=True)
            first = comp.columns[0]
            if first.lower() not in {"feature", "variable"}:
                comp = comp.rename(columns={first: "feature"})
            if "component" not in [c.lower() for c in comp.columns]:
                tidy = comp.melt(id_vars="feature", var_name="component", value_name="loading")
            else:
                cname = [c for c in comp.columns if c.lower() == "component"][0]
                tidy = comp.rename(columns={cname: "component"})
                if "loading" not in tidy.columns:
                    other = [c for c in tidy.columns if c not in {"feature","component"}]
                    if other:
                        tidy = tidy.rename(columns={other[-1]: "loading"})
            comp_sel = st.selectbox("Component", sorted(tidy["component"].astype(str).unique()))
        # Chart
        fig = px.bar(tidy[tidy["component"].astype(str) == str(comp_sel)], x="feature", y="loading", height=430)
        st.plotly_chart(fig, use_container_width=True, key="pca_bar")
        st.caption(f"Data source: {pcomp.relative_to(ROOT) if pcomp else 'N/A'}")
    else:
        st.info("Add `pca_components.xlsx` (or .csv) to `data/analysis-outputs/`.")

    st.markdown("---")
    st.markdown("### Cluster Distribution by Location", help="How segments are distributed across locations.")
    seg, pseg = find_and_read("pca_kmeans_results.xlsx")
    if seg is None:
        seg, pseg = find_and_read("pca_kmeans_results.csv")

    if seg is not None and not seg.empty:
        # Controls
        with st.container():
            st.markdown('<div class="section-controls">**Controls**</div>', unsafe_allow_html=True)
            col_loc = choose_col(seg, "Location column", ["location","office","city","region"])
            col_seg = choose_col(seg, "Segment/Cluster column", ["segment","cluster","group","label"])
            loc = st.selectbox("Location filter", ["All"] + sorted(seg[col_loc].dropna().astype(str).unique()))
        # Chart
        view = seg if loc == "All" else seg[seg[col_loc].astype(str) == str(loc)]
        fig = px.histogram(view, x=col_seg, color=col_seg, height=400)
        st.plotly_chart(fig, use_container_width=True, key="seg_hist")
        st.caption(f"Data source: {pseg.relative_to(ROOT) if pseg else 'N/A'}")
    else:
        st.info("Add `pca_kmeans_results.xlsx` (or .csv) to `data/analysis-outputs/`.")

# ==================== TAB 3: EXPERIMENT ====================
with tab3:
    st.markdown("### Program Improvements (A/B vs Current)", help="Compares pre→post changes in proficiency/applications.")
    exp, pexp = find_and_read("experiment_curriculum_cleaned.csv")
    if exp is None:
        exp, pexp = find_and_read("nls_experiment_cleaned.csv")  # legacy

    if exp is not None and not exp.empty:
        # Controls
        with st.container():
            st.markdown('<div class="section-controls">**Controls**</div>', unsafe_allow_html=True)
            col_prog  = choose_col(exp, "Program column", ["program","curriculum","group"])
            col_pre_p = choose_col(exp, "Pre proficiency",  ["pre_proficiency","proficiency_pre","pre_prof"])
            col_post_p= choose_col(exp, "Post proficiency", ["post_proficiency","proficiency_post","post_prof"])
            col_pre_a = choose_col(exp, "Pre applications", ["pre_applications","applications_pre","pre_apps"])
            col_post_a= choose_col(exp, "Post applications",["post_applications","applications_post","post_apps"])
            metric = st.radio("Metric", ["proficiency", "applications"], horizontal=True, index=0)

        # Chart
        exp["delta_proficiency"] = pd.to_numeric(exp[col_post_p], errors="coerce") - pd.to_numeric(exp[col_pre_p], errors="coerce")
        exp["delta_applications"] = pd.to_numeric(exp[col_post_a], errors="coerce") - pd.to_numeric(exp[col_pre_a], errors="coerce")
        y = f"delta_{metric}"
        fig = px.box(exp, x=col_prog, y=y, points="all", color=col_prog, height=430)
        st.plotly_chart(fig, use_container_width=True, key="exp_box")
        st.caption(f"Data source: {pexp.relative_to(ROOT) if pexp else 'N/A'}")
    else:
        st.info("Add `experiment_curriculum_cleaned.csv` to `data/analysis-outputs/`.")
