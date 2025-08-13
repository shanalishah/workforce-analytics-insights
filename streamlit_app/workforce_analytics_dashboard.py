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

# Optional: gentle CSS tweaks (reduce padding a bit, standardize chart heights)
st.markdown("""
<style>
/* tighten top/bottom spacing just a touch */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
/* make expander label slightly smaller */
.streamlit-expanderHeader { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# --------- Paths & readers ---------
ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [ROOT / "data" / "analysis-outputs", ROOT / "data" / "processed"]

@st.cache_data(show_spinner=False)
def find_and_read(name: str):
    """Search known folders and read CSV/XLSX."""
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
    """
    Streamlit slider safe wrapper for Top-N selection.
    Ensures (min <= max) even when n_items is very small.
    """
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
st.caption("Explore utilization, segmentation (PCA + KMeans), and curriculum experiment outcomes.")

# --------- KPI row (lightweight) ---------
k1, k2, k3 = st.columns(3)
util_df, _ = find_and_read("country_enrollment_summary.csv")
if util_df is None:
    util_df, _ = find_and_read("Country-wise_Enrollment_Summary.csv")
if util_df is not None:
    c_country = guess_col(util_df, ["country", "country_name", "nation"]) or util_df.columns[0]
    c_enroll  = guess_col(util_df, ["enrollments", "enrollment", "total_enrollments", "count"]) or util_df.columns[1]
    util_df[c_enroll] = pd.to_numeric(util_df[c_enroll], errors="coerce")
    util_df = util_df.dropna(subset=[c_enroll])
    if not util_df.empty:
        total_enroll = int(util_df[c_enroll].sum())
        top_row = util_df.loc[util_df[c_enroll].idxmax()]
        k1.metric("Total enrollments", f"{total_enroll:,}")
        k2.metric("Top country", str(top_row[c_country]))
        k3.metric("Max enrollments (country)", f"{int(top_row[c_enroll]):,}")

st.markdown("---")

# --------- Tabs ---------
tab1, tab2, tab3 = st.tabs(["📍 Utilization", "🧩 Segmentation (PCA + KMeans)", "🧪 Curriculum Experiment"])

# ==================== UTILIZATION ====================
with tab1:
    controls, viz = st.columns([1.0, 3.0], gap="large")

    # ---- Controls (left rail) ----
    with controls:
        st.subheader("Filters")
        df_u, p_u = find_and_read("country_enrollment_summary.csv")
        if df_u is None:
            df_u, p_u = find_and_read("Country-wise_Enrollment_Summary.csv")

        if df_u is not None:
            col_country = choose_col(df_u, "Country", ["country", "country_name", "nation"])
            col_enroll  = choose_col(df_u, "Enrollments",
                                     ["enrollments", "enrollment", "total_enrollments", "count"],
                                     help_text="Numeric enrollments per country")
            # top-N safely
            topn = safe_slider("Top N countries", len(df_u))
        else:
            st.info("Upload country_enrollment_summary.csv")

        st.markdown("---")
        df_a, p_a = find_and_read("course_assessment_by_course.csv")
        if df_a is None:
            df_a, p_a = find_and_read("Course_wise_assessment.csv")

        if df_a is not None:
            st.subheader("Assessment")
            col_course   = choose_col(df_a, "Course",   ["course_title","course","course_name","course_id"])
            col_delivery = choose_col(df_a, "Delivery", ["delivery","mode","format","delivery_mode"])
            col_dprof    = choose_col(df_a, "Δ Proficiency", ["delta_proficiency","prof_delta","proficiency_delta"])
            col_dapp     = choose_col(df_a, "Δ Applications", ["delta_applications","apps_delta","applications_delta"])
            # dynamic course choose lives here on left
            course_sel = st.selectbox("Filter course", sorted(df_a[col_course].dropna().astype(str).unique()))
        else:
            st.info("Upload course_assessment_by_course.csv")

    # ---- Visuals (right) ----
    with viz:
        st.subheader("Training utilization by country")
        if df_u is not None:
            df_u[col_enroll] = pd.to_numeric(df_u[col_enroll], errors="coerce")
            df_u = df_u.dropna(subset=[col_enroll])
            if df_u.empty:
                st.warning("No numeric enrollments found.")
            else:
                view = df_u.sort_values(col_enroll, ascending=False).head(topn)
                fig = px.bar(view, x=col_country, y=col_enroll, height=420)
                st.plotly_chart(fig, use_container_width=True, key="util_bar")

        st.markdown("---")
        st.subheader("Assessment gains by delivery")
        if df_a is not None:
            # coerce numeric
            for c in [col_dprof, col_dapp]:
                df_a[c] = pd.to_numeric(df_a[c], errors="coerce")
            sub = df_a[df_a[col_course].astype(str) == str(course_sel)]
            c1, c2 = st.columns(2)
            with c1:
                fig1 = px.box(sub, x=col_delivery, y=col_dprof, points="all", height=380)
                st.plotly_chart(fig1, use_container_width=True, key="assess_prof")
            with c2:
                fig2 = px.box(sub, x=col_delivery, y=col_dapp, points="all", height=380)
                st.plotly_chart(fig2, use_container_width=True, key="assess_apps")

# ==================== SEGMENTATION ====================
with tab2:
    controls2, viz2 = st.columns([1.0, 3.0], gap="large")

    with controls2:
        st.subheader("PCA")
        comp, pcomp = find_and_read("pca_components.xlsx")
        if comp is None:
            comp, pcomp = find_and_read("pca_components.csv")

        comp_sel = None
        if comp is not None:
            first = comp.columns[0]
            if first.lower() not in {"feature", "variable"}:
                comp = comp.rename(columns={first: "feature"})
            # reshape wide→long if needed
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
        else:
            st.info("Upload pca_components.xlsx (or .csv)")

        st.markdown("---")
        st.subheader("Clusters")
        seg, pseg = find_and_read("pca_kmeans_results.xlsx")
        if seg is None:
            seg, pseg = find_and_read("pca_kmeans_results.csv")

        loc = col_loc = col_seg = None
        if seg is not None:
            col_loc = choose_col(seg, "Location", ["location","office","city","region"])
            col_seg = choose_col(seg, "Segment",  ["segment","cluster","group","label"])
            loc = st.selectbox("Location filter", ["All"] + sorted(seg[col_loc].dropna().astype(str).unique()))
        else:
            st.info("Upload pca_kmeans_results.xlsx (or .csv)")

    with viz2:
        st.subheader("PCA components (top loadings)")
        if comp is not None and comp_sel is not None:
            # tidy was defined above; recompute tidy to keep scope clear
            first = comp.columns[0]
            if first.lower() not in {"feature", "variable"}:
                comp = comp.rename(columns={first: "feature"})
            if "component" not in [c.lower() for c in comp.columns]:
                tidy2 = comp.melt(id_vars="feature", var_name="component", value_name="loading")
            else:
                cname = [c for c in comp.columns if c.lower() == "component"][0]
                tidy2 = comp.rename(columns={cname: "component"})
                if "loading" not in tidy2.columns:
                    other = [c for c in tidy2.columns if c not in {"feature","component"}]
                    if other:
                        tidy2 = tidy2.rename(columns={other[-1]: "loading"})
            fig = px.bar(tidy2[tidy2["component"].astype(str) == str(comp_sel)], x="feature", y="loading", height=420)
            st.plotly_chart(fig, use_container_width=True, key="pca_bar")

        st.markdown("---")
        st.subheader("Cluster distribution by location")
        if seg is not None and col_loc is not None and col_seg is not None:
            view = seg if loc == "All" else seg[seg[col_loc].astype(str) == str(loc)]
            fig = px.histogram(view, x=col_seg, color=col_seg, height=380)
            st.plotly_chart(fig, use_container_width=True, key="seg_hist")

# ==================== EXPERIMENT ====================
with tab3:
    controls3, viz3 = st.columns([1.0, 3.0], gap="large")

    with controls3:
        st.subheader("Experiment")
        exp, pexp = find_and_read("experiment_curriculum_cleaned.csv")
        if exp is None:
            exp, pexp = find_and_read("nls_experiment_cleaned.csv")  # legacy
        metric = None
        if exp is not None:
            col_prog  = choose_col(exp, "Program", ["program","curriculum","group"])
            col_pre_p = choose_col(exp, "Pre proficiency",  ["pre_proficiency","proficiency_pre","pre_prof"])
            col_post_p= choose_col(exp, "Post proficiency", ["post_proficiency","proficiency_post","post_prof"])
            col_pre_a = choose_col(exp, "Pre applications", ["pre_applications","applications_pre","pre_apps"])
            col_post_a= choose_col(exp, "Post applications",["post_applications","applications_post","post_apps"])
            metric = st.radio("Metric", ["proficiency", "applications"], horizontal=False, index=0)
        else:
            st.info("Upload experiment_curriculum_cleaned.csv")

    with viz3:
        st.subheader("Program improvements (A/B vs Current)")
        if exp is not None:
            exp["delta_proficiency"] = pd.to_numeric(exp[col_post_p], errors="coerce") - pd.to_numeric(exp[col_pre_p], errors="coerce")
            exp["delta_applications"] = pd.to_numeric(exp[col_post_a], errors="coerce") - pd.to_numeric(exp[col_pre_a], errors="coerce")
            y = f"delta_{metric or 'proficiency'}"
            fig = px.box(exp, x=col_prog, y=y, points="all", color=col_prog, height=420)
            st.plotly_chart(fig, use_container_width=True, key="exp_box")
