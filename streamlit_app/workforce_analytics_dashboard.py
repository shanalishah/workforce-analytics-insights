import streamlit as st, pandas as pd
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="Workforce Analytics Insights", page_icon="📊", layout="wide")
ROOT = Path(__file__).resolve().parents[1]

st.title("Workforce Analytics Insights")
st.caption("EDA · PCA · KMeans clustering · Experiment analysis")

tab1, tab2, tab3 = st.tabs(["Utilization", "Segmentation (PCA+KMeans)", "Curriculum Experiment"])

with tab1:
    st.subheader("Training Utilization by Country")
    try:
        df = pd.read_csv(ROOT / "data/processed/country_enrollment_summary.csv")
        topn = st.slider("Top N countries", 5, min(25, len(df)), 10)
        view = df.sort_values("enrollments", ascending=False).head(topn)
        st.plotly_chart(px.bar(view, x="country", y="enrollments"), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load country_enrollment_summary.csv — {e}")

    st.markdown("—")
    st.subheader("Assessment Gains by Delivery")
    try:
        df2 = pd.read_csv(ROOT / "data/processed/course_assessment_by_course.csv")
        course = st.selectbox("Course", sorted(df2["course_title"].dropna().unique()))
        sub = df2[df2["course_title"] == course]
        c1, c2 = st.columns(2)
        with c1:
            st.write("Δ Proficiency")
            st.plotly_chart(px.box(sub, x="delivery", y="delta_proficiency", points="all"), use_container_width=True)
        with c2:
            st.write("Δ Applications")
            st.plotly_chart(px.box(sub, x="delivery", y="delta_applications", points="all"), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load course_assessment_by_course.csv — {e}")

with tab2:
    st.subheader("PCA Components (Top Loadings)")
    try:
        comp = pd.read_excel(ROOT / "data/processed/pca_components.xlsx")
        comp = comp.rename(columns={comp.columns[0]:"feature"})
        tidy = comp.melt(id_vars="feature", var_name="component", value_name="loading")
        comp_sel = st.selectbox("Component", sorted(tidy["component"].unique()))
        st.plotly_chart(px.bar(tidy[tidy["component"]==comp_sel], x="feature", y="loading"), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load pca_components.xlsx — {e}")

    st.markdown("—")
    st.subheader("Cluster Distribution by Location")
    try:
        seg = pd.read_excel(ROOT / "data/processed/pca_kmeans_results.xlsx")
        # expect columns like: employee_id, location, segment
        loc = st.selectbox("Location", ["All"] + sorted(seg["location"].dropna().unique()))
        view = seg if loc == "All" else seg[seg["location"]==loc]
        st.plotly_chart(px.histogram(view, x="segment", color="segment"), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load pca_kmeans_results.xlsx — {e}")

with tab3:
    st.subheader("Program Improvements (A/B vs Current)")
    try:
        exp = pd.read_csv(ROOT / "data/processed/experiment_curriculum_cleaned.csv")
        # expect columns: program, pre_proficiency, post_proficiency, pre_applications, post_applications
        metric = st.radio("Metric", ["proficiency", "applications"], horizontal=True)
        exp[f"delta_{metric}"] = exp[f"post_{metric}"] - exp[f"pre_{metric}"]
        st.plotly_chart(px.box(exp, x="program", y=f"delta_{metric}", points="all", color="program"), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load experiment_curriculum_cleaned.csv — {e}")
