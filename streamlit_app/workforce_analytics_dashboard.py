import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------- App config ---------
st.set_page_config(
    page_title="Workforce Analytics — Interactive Insights",
    layout="wide",
    page_icon="📊",
    menu_items={"About": "Workforce Analytics — EDA · PCA · KMeans · Experiment"}
)

# --------- Paths & loaders ---------
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
    left, right = st.columns([2.2, 1.0])
    with left:
        st.subheader("Training utilization by country")
        df, p = find_and_read("country_enrollment_summary.csv")
        if df is None:
            df, p = find_and_read("Country-wise_Enrollment_Summary.csv")
        if df is not None:
            col_country = choose_col(df, "Country column", ["country", "country_name", "nation"])
            col_enroll  = choose_col(df, "Enrollments column",
                                     ["enrollments", "enrollment", "total_enrollments", "count"],
                                     help_text="Numeric enrollments per country")
            df[col_enroll] = pd.to_numeric(df[col_enroll], errors="coerce")
            df = df.dropna(subset=[col_enroll])
            topn = st.slider("Top N countries", 5, min(25, len(df)), min(10, len(df)))
            view = df.sort_values(col_enroll, ascending=False).head(topn)
            fig = px.bar(view, x=col_country, y=col_enroll)
            st.plotly_chart(fig, use_container_width=True, key="util_bar")
        else:
            st.info("Upload country_enrollment_summary.csv to data/analysis-outputs/")

    with right:
        st.subheader("Assessment gains by delivery")
        df2, p2 = find_and_read("course_assessment_by_course.csv")
        if df2 is None:
            df2, p2 = find_and_read("Course_wise_assessment.csv")
        if df2 is not None:
            col_course   = choose_col(df2, "Course column", ["course_title","course","course_name","course_id"])
            col_delivery = choose_col(df2, "Delivery column", ["delivery","mode","format","delivery_mode"])
            col_dprof    = choose_col(df2, "Δ Proficiency", ["delta_proficiency","prof_delta","proficiency_delta"])
            col_dapp     = choose_col(df2, "Δ Applications", ["delta_applications","apps_delta","applications_delta"])
            for c in [col_dprof, col_dapp]:
                df2[c] = pd.to_numeric(df2[c], errors="coerce")
            course_sel = st.selectbox("Filter course", sorted(df2[col_course].dropna().astype(str).unique()))
            sub = df2[df2[col_course].astype(str) == str(course_sel)]
            fig1 = px.box(sub, x=col_delivery, y=col_dprof, points="all")
            st.plotly_chart(fig1, use_container_width=True, key="assess_prof")
            fig2 = px.box(sub, x=col_delivery, y=col_dapp, points="all")
            st.plotly_chart(fig2, use_container_width=True, key="assess_apps")
        else:
            st.info("Upload course_assessment_by_course.csv to data/analysis-outputs/")

# ==================== SEGMENTATION ====================
with tab2:
    st.subheader("PCA components (top loadings)")
    comp, pcomp = find_and_read("pca_components.xlsx")
    if comp is None:
        comp, pcomp = find_and_read("pca_components.csv")
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
        fig = px.bar(tidy[tidy["component"].astype(str) == str(comp_sel)], x="feature", y="loading")
        st.plotly_chart(fig, use_container_width=True, key="pca_bar")
    else:
        st.info("Upload pca_components.xlsx (or .csv) to data/analysis-outputs/")

    st.markdown("---")
    st.subheader("Cluster distribution by location")
    seg, pseg = find_and_read("pca_kmeans_results.xlsx")
    if seg is None:
        seg, pseg = find_and_read("pca_kmeans_results.csv")
    if seg is not None:
        col_loc = choose_col(seg, "Location column", ["location","office","city","region"])
        col_seg = choose_col(seg, "Segment column", ["segment","cluster","group","label"])
        loc = st.selectbox("Location filter", ["All"] + sorted(seg[col_loc].dropna().astype(str).unique()))
        view = seg if loc == "All" else seg[seg[col_loc].astype(str) == str(loc)]
        fig = px.histogram(view, x=col_seg, color=col_seg)
        st.plotly_chart(fig, use_container_width=True, key="seg_hist")
    else:
        st.info("Upload pca_kmeans_results.xlsx (or .csv) to data/analysis-outputs/")

# ==================== EXPERIMENT ====================
with tab3:
    st.subheader("Program improvements (A/B vs Current)")
    exp, pexp = find_and_read("experiment_curriculum_cleaned.csv")
    if exp is None:
        exp, pexp = find_and_read("nls_experiment_cleaned.csv")  # legacy
    if exp is not None:
        col_prog  = choose_col(exp, "Program column", ["program","curriculum","group"])
        col_pre_p = choose_col(exp, "Pre proficiency",  ["pre_proficiency","proficiency_pre","pre_prof"])
        col_post_p= choose_col(exp, "Post proficiency", ["post_proficiency","proficiency_post","post_prof"])
        col_pre_a = choose_col(exp, "Pre applications", ["pre_applications","applications_pre","pre_apps"])
        col_post_a= choose_col(exp, "Post applications",["post_applications","applications_post","post_apps"])

        exp["delta_proficiency"] = pd.to_numeric(exp[col_post_p], errors="coerce") - pd.to_numeric(exp[col_pre_p], errors="coerce")
        exp["delta_applications"] = pd.to_numeric(exp[col_post_a], errors="coerce") - pd.to_numeric(exp[col_pre_a], errors="coerce")

        metric = st.radio("Metric", ["proficiency", "applications"], horizontal=True)
        y = f"delta_{metric}"
        fig = px.box(exp, x=col_prog, y=y, points="all", color=col_prog)
        st.plotly_chart(fig, use_container_width=True, key="exp_box")
    else:
        st.info("Upload experiment_curriculum_cleaned.csv to data/analysis-outputs/")
