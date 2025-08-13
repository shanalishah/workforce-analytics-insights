# streamlit_app/workforce_analytics_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Workforce Analytics Insights",
    page_icon="📊",
    layout="wide",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": "Workforce Analytics — EDA · PCA · KMeans · Experiment"
    },
)

# ---------- PATHS & LOADERS ----------
ROOT = Path(__file__).resolve().parents[1]
DIRS = [
    ROOT / "data" / "analysis-outputs",  # your real data
    ROOT / "data" / "processed",         # fallback
]

@st.cache_data(show_spinner=False)
def _find_and_read(name: str):
    # Auto-detect CSV vs Excel; search known folders
    for base in DIRS:
        p = base / name
        if p.exists():
            if p.suffix.lower() == ".csv":
                return pd.read_csv(p), p
            if p.suffix.lower() in (".xlsx", ".xls"):
                # Requires openpyxl in requirements.txt
                return pd.read_excel(p), p
    return None, None

def _guess_col(df: pd.DataFrame, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def _choose_col(df: pd.DataFrame, label: str, candidates, help_text=""):
    guess = _guess_col(df, candidates) or df.columns[0]
    return st.selectbox(label, options=list(df.columns),
                        index=list(df.columns).index(guess), help=help_text)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Settings")
    theme_hint = st.toggle("Compact mode", value=True, help="Reduces vertical spacing")
    st.caption("Data sources searched: `data/analysis-outputs/`, then `data/processed/`.")

# ---------- TITLE / INTRO ----------
st.title("Workforce Analytics Insights")
st.caption("EDA · PCA · KMeans clustering · Curriculum experiment")

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["📍 Utilization", "🧩 Segmentation (PCA+KMeans)", "🧪 Curriculum Experiment"])

# ==================== TAB 1: UTILIZATION ====================
with tab1:
    box = st.container()
    with box:
        # --- Top KPIs row (simple) ---
        kpi1, kpi2, kpi3 = st.columns(3)
        # country_enrollment_summary
        df_c, p_c = _find_and_read("country_enrollment_summary.csv")
        if df_c is None:
            # fallback names
            df_c, p_c = _find_and_read("Country-wise_Enrollment_Summary.csv")
        if df_c is not None:
            country_col = _choose_col(df_c, "Country column", ["country", "country_name", "nation"])
            enroll_col  = _choose_col(df_c, "Enrollments column",
                                      ["enrollments", "enrollment", "total_enrollments", "count"],
                                      help_text="Numeric enrollments per country")
            df_c[enroll_col] = pd.to_numeric(df_c[enroll_col], errors="coerce")
            df_c = df_c.dropna(subset=[enroll_col])

            total_enroll = int(df_c[enroll_col].sum())
            top_country = df_c.iloc[df_c[enroll_col].argmax()][country_col]
            top_value   = int(df_c[enroll_col].max())

            kpi1.metric("Total enrollments", f"{total_enroll:,}")
            kpi2.metric("Top country", str(top_country))
            kpi3.metric("Max enrollments (country)", f"{top_value:,}")

            st.markdown("---")
            st.subheader("Training utilization by country")
            topn = st.slider("Show Top N countries", 5, min(25, len(df_c)), min(10, len(df_c)))
            view = df_c.sort_values(enroll_col, ascending=False).head(topn)
            fig = px.bar(view, x=country_col, y=enroll_col)
            st.plotly_chart(fig, use_container_width=True, key="util_bar_chart")  # UNIQUE KEY
            st.caption(f"Source: {p_c.relative_to(ROOT) if p_c else 'N/A'}")

        else:
            st.warning("Could not find `country_enrollment_summary.csv` (or fallback).")

        st.markdown("---")
        # --- Assessment gains by delivery ---
        df_a, p_a = _find_and_read("course_assessment_by_course.csv")
        if df_a is None:
            df_a, p_a = _find_and_read("Course_wise_assessment.csv")
        if df_a is not None:
            st.subheader("Assessment gains by delivery mode")
            col_course  = _choose_col(df_a, "Course column",   ["course_title", "course_name", "course", "course_id"])
            col_delivery= _choose_col(df_a, "Delivery column", ["delivery", "mode", "format", "delivery_mode"])
            col_dprof   = _choose_col(df_a, "Δ Proficiency column",
                                      ["delta_proficiency", "prof_delta", "proficiency_delta"])
            col_dapps   = _choose_col(df_a, "Δ Applications column",
                                      ["delta_applications", "apps_delta", "applications_delta"])

            for c in [col_dprof, col_dapps]:
                df_a[c] = pd.to_numeric(df_a[c], errors="coerce")

            course_sel = st.selectbox("Filter — course", sorted(df_a[col_course].dropna().astype(str).unique()))
            sub = df_a[df_a[col_course].astype(str) == str(course_sel)]

            c1, c2 = st.columns(2)
            with c1:
                st.write("Δ Proficiency")
                fig1 = px.box(sub, x=col_delivery, y=col_dprof, points="all")
                st.plotly_chart(fig1, use_container_width=True, key="assess_prof_box")  # UNIQUE KEY
            with c2:
                st.write("Δ Applications")
                fig2 = px.box(sub, x=col_delivery, y=col_dapps, points="all")
                st.plotly_chart(fig2, use_container_width=True, key="assess_apps_box")  # UNIQUE KEY
            st.caption(f"Source: {p_a.relative_to(ROOT) if p_a else 'N/A'}")
        else:
            st.warning("Could not find `course_assessment_by_course.csv` (or fallback).")

# ==================== TAB 2: SEGMENTATION ====================
with tab2:
    # --- PCA loadings ---
    st.subheader("PCA components (top loadings)")
    comp, p_comp = _find_and_read("pca_components.xlsx")
    if comp is None:
        comp, p_comp = _find_and_read("pca_components.csv")
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
                # guess the last non-id column
                candidates = [c for c in tidy.columns if c not in {"feature", "component"}]
                if candidates:
                    tidy = tidy.rename(columns={candidates[-1]: "loading"})

        comp_sel = st.selectbox("Component", sorted(tidy["component"].astype(str).unique()))
        fig = px.bar(tidy[tidy["component"].astype(str) == str(comp_sel)], x="feature", y="loading")
        st.plotly_chart(fig, use_container_width=True, key="pca_bar_chart")  # UNIQUE KEY
        st.caption(f"Source: {p_comp.relative_to(ROOT) if p_comp else 'N/A'}")
    else:
        st.warning("Could not find `pca_components.xlsx` or `pca_components.csv`.")

    st.markdown("---")
    # --- Cluster distribution ---
    st.subheader("Cluster distribution by location")
    seg, p_seg = _find_and_read("pca_kmeans_results.xlsx")
    if seg is None:
        seg, p_seg = _find_and_read("pca_kmeans_results.csv")
    if seg is not None:
        col_loc = _choose_col(seg, "Location column", ["location", "office", "city", "region"])
        col_seg = _choose_col(seg, "Segment column", ["segment", "cluster", "group", "label"])
        locs = ["All"] + sorted(seg[col_loc].dropna().astype(str).unique())
        loc = st.selectbox("Filter — location", locs)
        view = seg if loc == "All" else seg[seg[col_loc].astype(str) == str(loc)]
        fig = px.histogram(view, x=col_seg, color=col_seg)
        st.plotly_chart(fig, use_container_width=True, key="seg_hist_chart")  # UNIQUE KEY
        st.caption(f"Source: {p_seg.relative_to(ROOT) if p_seg else 'N/A'}")
    else:
        st.warning("Could not find `pca_kmeans_results.xlsx` or `pca_kmeans_results.csv`.")

# ==================== TAB 3: EXPERIMENT ====================
with tab3:
    st.subheader("Program improvements (A/B vs Current)")
    exp, p_exp = _find_and_read("experiment_curriculum_cleaned.csv")
    if exp is None:
        exp, p_exp = _find_and_read("nls_experiment_cleaned.csv")  # legacy name
    if exp is not None:
        col_prog = _choose_col(exp, "Program column", ["program", "curriculum", "group"])
        col_pre_p = _choose_col(exp, "Pre proficiency",  ["pre_proficiency","proficiency_pre","pre_prof"])
        col_post_p= _choose_col(exp, "Post proficiency", ["post_proficiency","proficiency_post","post_prof"])
        col_pre_a = _choose_col(exp, "Pre applications", ["pre_applications","applications_pre","pre_apps"])
        col_post_a= _choose_col(exp, "Post applications",["post_applications","applications_post","post_apps"])

        # Compute deltas
        exp["delta_proficiency"] = pd.to_numeric(exp[col_post_p], errors="coerce") - pd.to_numeric(exp[col_pre_p], errors="coerce")
        exp["delta_applications"] = pd.to_numeric(exp[col_post_a], errors="coerce") - pd.to_numeric(exp[col_pre_a], errors="coerce")

        metric = st.radio("Metric", ["proficiency", "applications"], horizontal=True)
        ycol = f"delta_{metric}"
        fig = px.box(exp, x=col_prog, y=ycol, points="all", color=col_prog)
        st.plotly_chart(fig, use_container_width=True, key="experiment_box_chart")  # UNIQUE KEY
        st.caption(f"Source: {p_exp.relative_to(ROOT) if p_exp else 'N/A'}")
    else:
        st.warning("Could not find `experiment_curriculum_cleaned.csv`.")
