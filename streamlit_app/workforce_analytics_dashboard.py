import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Workforce Analytics Insights", page_icon="📊", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
DIRS = [
    ROOT / "data" / "analysis-outputs",  # your current folder
    ROOT / "data" / "processed",         # future-proof fallback
    ROOT / "data" / "samples",           # tiny public samples (optional)
]

def _find_path(fname: str):
    for base in DIRS:
        p = base / fname
        if p.exists():
            return p
    return None

def _read_table(name: str):
    p = _find_path(name)
    if p is None:
        raise FileNotFoundError(f"Not found in {DIRS}: {name}")
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    elif p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

def guess_col(df: pd.DataFrame, candidates, default=None):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return default

def choose_col_ui(df: pd.DataFrame, label: str, candidates, help_text=""):
    guess = guess_col(df, candidates)
    return st.selectbox(
        label,
        options=list(df.columns),
        index=(list(df.columns).index(guess) if guess in df.columns else 0),
        help=help_text
    )

st.title("Workforce Analytics Insights")
st.caption("EDA · PCA · KMeans clustering · Experiment analysis")

tab1, tab2, tab3 = st.tabs(["Utilization", "Segmentation (PCA+KMeans)", "Curriculum Experiment"])

# -------------------- UTILIZATION --------------------
with tab1:
    st.subheader("Training Utilization by Country")
    try:
        df = None
        # Try common names for the utilization file
        for name in [
            "country_enrollment_summary.csv",
            "country_enrollment_summary.xlsx",
            "Country-wise_Enrollment_Summary.csv",  # legacy
        ]:
            p = _find_path(name)
            if p is not None:
                df = _read_table(name)
                break
        if df is None:
            raise FileNotFoundError("country_enrollment_summary file not found")

        # Pick columns (auto-guess, then allow override)
        col_country = choose_col_ui(
            df,
            "Country column",
            candidates=["country", "nation", "country_name"]
        )
        col_enroll = choose_col_ui(
            df,
            "Enrollments column",
            candidates=["enrollments", "enrollment", "count", "total_enrollments"],
            help_text="Numeric column with enrollments per country"
        )

        # coerce to numeric safely
        df[col_enroll] = pd.to_numeric(df[col_enroll], errors="coerce")
        df = df.dropna(subset=[col_enroll])

        topn = st.slider("Top N countries", 5, min(25, len(df)), min(10, len(df)))
        view = df.sort_values(col_enroll, ascending=False).head(topn)
        fig = px.bar(view, x=col_country, y=col_enroll)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Utilization: {e}")

    st.markdown("—")
    st.subheader("Assessment Gains by Delivery")
    try:
        df2 = None
        for name in [
            "course_assessment_by_course.csv",
            "Course_wise_assessment.csv",       # legacy
            "course_assessment_by_course.xlsx",
        ]:
            p = _find_path(name)
            if p is not None:
                df2 = _read_table(name)
                break
        if df2 is None:
            raise FileNotFoundError("course_assessment_by_course file not found")

        # Let user pick columns for course, delivery, proficiency delta, and applications delta
        col_course = choose_col_ui(
            df2, "Course column",
            candidates=["course_title", "course", "course_name", "course_id"]
        )
        col_delivery = choose_col_ui(
            df2, "Delivery column",
            candidates=["delivery", "mode", "format", "delivery_mode"]
        )
        col_dprof = choose_col_ui(
            df2, "Δ Proficiency column",
            candidates=["delta_proficiency", "delta_prof", "prof_delta", "proficiency_delta"]
        )
        col_dapp = choose_col_ui(
            df2, "Δ Applications column",
            candidates=["delta_applications", "delta_apps", "applications_delta"]
        )

        # Coerce numeric
        for c in [col_dprof, col_dapp]:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

        course_selected = st.selectbox("Course", sorted(df2[col_course].dropna().unique()))
        sub = df2[df2[col_course] == course_selected]
        c1, c2 = st.columns(2)
        with c1:
            st.write("Δ Proficiency by Delivery")
            st.plotly_chart(px.box(sub, x=col_delivery, y=col_dprof, points="all"), use_container_width=True)
        with c2:
            st.write("Δ Applications by Delivery")
            st.plotly_chart(px.box(sub, x=col_delivery, y=col_dapp, points="all"), use_container_width=True)

    except Exception as e:
        st.warning(f"Assessment: {e}")

# -------------------- SEGMENTATION --------------------
with tab2:
    st.subheader("PCA Components (Top Loadings)")
    try:
        comp = None
        for name in ["pca_components.xlsx", "pca_components.csv", "PCA.xlsx"]:
            p = _find_path(name)
            if p is not None:
                comp = _read_table(name)
                break
        if comp is None:
            raise FileNotFoundError("pca_components file not found")

        # assume 1st col = feature if unlabeled
        first = comp.columns[0]
        if first.lower() not in {"feature", "variable"}:
            comp = comp.rename(columns={first: "feature"})
        # reshape if wide
        if "component" not in [c.lower() for c in comp.columns]:
            tidy = comp.melt(id_vars="feature", var_name="component", value_name="loading")
        else:
            # already tidy
            cname = [c for c in comp.columns if c.lower() == "component"][0]
            tidy = comp.rename(columns={cname: "component"})
            if "loading" not in tidy.columns:
                # try the last column as loading
                candidates = [c for c in tidy.columns if c not in {"feature", "component"}]
                if candidates:
                    tidy = tidy.rename(columns={candidates[-1]: "loading"})

        comp_sel = st.selectbox("Component", sorted(tidy["component"].astype(str).unique()))
        st.plotly_chart(px.bar(tidy[tidy["component"].astype(str) == str(comp_sel)], x="feature", y="loading"),
                        use_container_width=True)

    except Exception as e:
        st.warning(f"PCA: {e}")

    st.markdown("—")
    st.subheader("Cluster Distribution by Location")
    try:
        seg = None
        for name in ["pca_kmeans_results.xlsx", "pca_kmeans_results.csv", "PCA_and_KMeans_Results.xlsx"]:
            p = _find_path(name)
            if p is not None:
                seg = _read_table(name)
                break
        if seg is None:
            raise FileNotFoundError("pca_kmeans_results file not found")

        col_loc = choose_col_ui(seg, "Location column", candidates=["location", "office", "city", "region"])
        col_seg = choose_col_ui(seg, "Segment column", candidates=["segment", "cluster", "group"])

        locs = ["All"] + sorted(seg[col_loc].dropna().astype(str).unique())
        loc = st.selectbox("Location filter", locs)
        view = seg if loc == "All" else seg[seg[col_loc].astype(str) == str(loc)]
        st.plotly_chart(px.histogram(view, x=col_seg, color=col_seg), use_container_width=True)

    except Exception as e:
        st.warning(f"Segmentation: {e}")

# -------------------- EXPERIMENT --------------------
with tab3:
    st.subheader("Program Improvements (A/B vs Current)")
    try:
        exp = None
        for name in ["experiment_curriculum_cleaned.csv", "nls_experiment_cleaned.csv"]:
            p = _find_path(name)
            if p is not None:
                exp = _read_table(name)
                break
        if exp is None:
            raise FileNotFoundError("experiment_curriculum_cleaned file not found")

        col_prog = choose_col_ui(exp, "Program column", ["program", "curriculum", "group"])
        col_pre_p = choose_col_ui(exp, "Pre Proficiency column",
                                  ["pre_proficiency", "proficiency_pre", "pre_prof"])
        col_post_p = choose_col_ui(exp, "Post Proficiency column",
                                   ["post_proficiency", "proficiency_post", "post_prof"])
        col_pre_a = choose_col_ui(exp, "Pre Applications column",
                                  ["pre_applications", "applications_pre", "pre_apps"])
        col_post_a = choose_col_ui(exp, "Post Applications column",
                                   ["post_applications", "applications_post", "post_apps"])

        # compute deltas
        exp["delta_proficiency"] = pd.to_numeric(exp[col_post_p], errors="coerce") - pd.to_numeric(exp[col_pre_p], errors="coerce")
        exp["delta_applications"] = pd.to_numeric(exp[col_post_a], errors="coerce") - pd.to_numeric(exp[col_pre_a], errors="coerce")

        metric = st.radio("Metric", ["proficiency", "applications"], horizontal=True)
        ycol = f"delta_{metric}"
        st.plotly_chart(px.box(exp, x=col_prog, y=ycol, points="all", color=col_prog), use_container_width=True)

    except Exception as e:
        st.warning(f"Experiment: {e}")
