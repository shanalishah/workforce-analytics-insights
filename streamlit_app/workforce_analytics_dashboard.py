# streamlit_app/workforce_analytics_dashboard.py
from pathlib import Path
import re
import pandas as pd
import plotly.express as px
import streamlit as st

# =================== Page / Theme ===================
st.set_page_config(
    page_title="Workforce Analytics — Clear Insights",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
h1, h2, h3 { margin-bottom: .25rem; }
.caption { color: var(--text-color-secondary); }
hr { margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# =================== Paths & IO ===================
ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [ROOT / "data" / "analysis-outputs", ROOT / "data" / "processed"]

@st.cache_data(show_spinner=False)
def find(name: str):
    for base in SEARCH_DIRS:
        p = base / name
        if p.exists():
            return p
    return None

@st.cache_data(show_spinner=False)
def read_any(name: str):
    p = find(name)
    if p is None:
        return None, None
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, low_memory=False), p
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p), p
    return None, None

# =================== Helpers (robust) ===================
def as_text_series(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return obj.astype(str).str.strip()

def guess_col(df: pd.DataFrame, candidates, *, prefer_text=True):
    m = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = m.get(str(cand).lower())
        if c is not None:
            if prefer_text and not pd.api.types.is_object_dtype(df[c]):
                continue
            return c
    if prefer_text:
        for c in df.columns:
            if pd.api.types.is_object_dtype(df[c]):
                return c
    return df.columns[0] if len(df.columns) else None

def best_location_col(df: pd.DataFrame):
    # choose readable place names; avoid coordinates or numeric codes
    for cand in ["city", "office", "region", "country", "location"]:
        for c in df.columns:
            if c.lower() == cand:
                s = as_text_series(df, c)
                if _looks_human_location(s):
                    return c
    # fallback
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) and _looks_human_location(as_text_series(df, c)):
            return c
    return df.columns[0]

def _looks_human_location(s: pd.Series) -> bool:
    s = s.dropna().astype(str)
    if s.empty: return False
    coord_like = s.str.match(r"^\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*$", na=False).mean()
    numeric_only = s.str.match(r"^\s*-?\d+(\.\d+)?\s*$", na=False).mean()
    has_letters = s.str.contains(r"[A-Za-z]", regex=True).mean()
    return has_letters >= 0.7 and coord_like <= 0.02 and numeric_only <= 0.05

def best_course_col(df: pd.DataFrame):
    for cand in ["course_title", "course_name", "course"]:
        for c in df.columns:
            if c.lower() == cand:
                return c
    for c in df.columns:
        if c.lower() == "course_id":
            return c
    return df.columns[0]

def ensure_delta(df, delta_col, pre_cands, post_cands, tmp_name):
    ok = delta_col and (pd.to_numeric(df[delta_col], errors="coerce").notna().any())
    if ok:
        df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce")
        return delta_col
    pre = guess_col(df, pre_cands, prefer_text=False)
    post = guess_col(df, post_cands, prefer_text=False)
    if pre and post:
        df[tmp_name] = pd.to_numeric(df[post], errors="coerce") - pd.to_numeric(df[pre], errors="coerce")
        return tmp_name
    return None

def top_n(series: pd.Series, n=10):
    counts = series.value_counts()
    return [v for v in counts.index[:n].tolist()]

# =================== Header ===================
st.title("Workforce Analytics — Clear Insights")
st.caption("A focused view on enrollments, training outcomes, segments, and program improvements.")

# =================== Load once for all tabs ===================
enr, _ = read_any("country_enrollment_summary.csv")
if enr is None:
    enr, _ = read_any("Country-wise_Enrollment_Summary.csv")

ass, _ = read_any("course_assessment_by_course.csv")
if ass is None:
    ass, _ = read_any("Course_wise_assessment.csv")

seg, _ = read_any("pca_kmeans_results.xlsx")
if seg is None:
    seg, _ = read_any("pca_kmeans_results.csv")

exp, _ = read_any("experiment_curriculum_cleaned.csv")
if exp is None:
    exp, _ = read_any("nls_experiment_cleaned.csv")

# =================== KPIs ===================
kpi = {}
if enr is not None and not enr.empty:
    c_country = guess_col(enr, ["country", "country_name", "nation"])
    c_enroll  = guess_col(enr, ["enrollments", "enrollment", "total_enrollments", "count"], prefer_text=False)
    enr[c_enroll] = pd.to_numeric(enr[c_enroll], errors="coerce")
    kpi["Total enrollments"] = f"{int(enr[c_enroll].sum(skipna=True)):,}"
    kpi["Countries represented"] = as_text_series(enr, c_country).nunique()

if ass is not None and not ass.empty:
    kpi["Courses analyzed"] = as_text_series(ass, best_course_col(ass)).nunique()

if seg is not None and not seg.empty:
    seg_col_guess = guess_col(seg, ["segment", "cluster", "group", "label"])
    kpi["Employee segments"] = as_text_series(seg, seg_col_guess).nunique()

if exp is not None and not exp.empty:
    pre_p  = guess_col(exp, ["pre_proficiency","proficiency_pre","pre_prof"], prefer_text=False)
    post_p = guess_col(exp, ["post_proficiency","proficiency_post","post_prof"], prefer_text=False)
    if pre_p and post_p:
        delta = pd.to_numeric(exp[post_p], errors="coerce") - pd.to_numeric(exp[pre_p], errors="coerce")
        if delta.notna().any():
            kpi["Median proficiency Δ"] = f"{delta.median():.2f}"

if kpi:
    cols = st.columns(min(4, len(kpi)))
    for (k, v), c in zip(kpi.items(), cols):
        c.metric(k, v)

st.markdown("---")

# =================== Tabs ===================
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Enrollments",
    "🎯 Training Outcomes",
    "🧩 Segments",
    "✨ About & Custom GPT",
])

# =========================================================
# TAB 1 — Enrollments
# =========================================================
with tab1:
    st.subheader("Enrollments by Country")
    if enr is None or enr.empty:
        st.info("Add `country_enrollment_summary.csv` to `data/analysis-outputs/`.")
    else:
        c_country = guess_col(enr, ["country", "country_name", "nation"])
        c_enroll  = guess_col(enr, ["enrollments", "enrollment", "total_enrollments", "count"], prefer_text=False)
        enr[c_enroll] = pd.to_numeric(enr[c_enroll], errors="coerce")
        view = enr.dropna(subset=[c_enroll]).copy()

        country_s = as_text_series(view, c_country)
        default_countries = top_n(country_s, n=10)
        picked = st.multiselect(
            "Countries (default shows top 10 by enrollments)",
            options=sorted(country_s.unique()),
            default=default_countries
        )
        order = st.radio("Sort by", ["Enrollments (desc)", "Country (A–Z)"], horizontal=True)

        view["_country"] = country_s
        if picked:
            view = view[view["_country"].isin(picked)]

        view = view.sort_values(c_enroll if order.startswith("Enrollments") else "_country",
                                ascending=not order.startswith("Enrollments"))

        if view.empty:
            st.info("No countries selected.")
        else:
            fig = px.bar(view, x="_country", y=c_enroll, height=420,
                         labels={"_country": "Country", c_enroll: "Enrollments"},
                         title="Enrollments for selected countries")
            st.plotly_chart(fig, use_container_width=True, key="enrollments_by_country")

# =========================================================
# TAB 2 — Training Outcomes
# =========================================================
with tab2:
    st.subheader("Training Outcomes by Course & Delivery Mode")
    if ass is None or ass.empty:
        st.info("Add `course_assessment_by_course.csv` to `data/analysis-outputs/`.")
    else:
        col_course = best_course_col(ass)
        course_s = as_text_series(ass, col_course)
        col_delivery = guess_col(ass, ["delivery","mode","format","delivery_mode"])
        delivery_s = as_text_series(ass, col_delivery)

        # Ensure outcomes (compute deltas if needed)
        d_prof = guess_col(ass, ["delta_proficiency","prof_delta","proficiency_delta"], prefer_text=False)
        d_apps = guess_col(ass, ["delta_applications","apps_delta","applications_delta"], prefer_text=False)

        d_prof = ensure_delta(
            ass, d_prof,
            ["pre_proficiency","proficiency_pre","pre_prof"],
            ["post_proficiency","proficiency_post","post_prof"],
            "__delta_prof"
        )
        d_apps = ensure_delta(
            ass, d_apps,
            ["pre_applications","applications_pre","pre_apps"],
            ["post_applications","applications_post","post_apps"],
            "__delta_apps"
        )

        course_pick = st.selectbox("Course", sorted(course_s.dropna().unique()))
        outcome_pick = st.radio("Outcome", ["Change in Proficiency", "Change in Applications"], horizontal=True)
        ycol = d_prof if outcome_pick == "Change in Proficiency" else d_apps

        mask = course_s.str.casefold() == str(course_pick).casefold()
        sub = ass[mask].copy()
        sub["_delivery"] = delivery_s[mask].values

        if (ycol is None) or sub.empty or sub[ycol].dropna().empty:
            st.info("No numeric outcome available for this course/outcome. Try a different course or verify the columns.")
        else:
            # Mean + Distribution
            c1, c2 = st.columns([1.1, 1])
            with c1:
                mean_df = sub.groupby("_delivery", as_index=False)[ycol].mean()
                fig_bar = px.bar(mean_df, x="_delivery", y=ycol, height=380,
                                 labels={"_delivery": "Delivery mode", ycol: outcome_pick},
                                 title=f"Average {outcome_pick} — {course_pick}")
                st.plotly_chart(fig_bar, use_container_width=True, key="course_outcomes_mean")
            with c2:
                fig_box = px.box(sub, x="_delivery", y=ycol, points="all", height=380,
                                 labels={"_delivery": "Delivery mode", ycol: outcome_pick},
                                 title="Distribution")
                st.plotly_chart(fig_box, use_container_width=True, key="course_outcomes_dist")

# =========================================================
# TAB 3 — Segments
# =========================================================
with tab3:
    st.subheader("Employee Segments")
    if seg is None or seg.empty:
        st.info("Add `pca_kmeans_results.xlsx/csv` to `data/analysis-outputs/`.")
    else:
        seg_col = guess_col(seg, ["segment","cluster","group","label"])
        loc_col = best_location_col(seg)

        seg_s = as_text_series(seg, seg_col)
        loc_s = as_text_series(seg, loc_col)

        # A) Segment sizes
        sizes = pd.DataFrame({"segment": seg_s}).value_counts().reset_index(name="employees")
        sizes.columns = ["segment", "employees"]

        c1, c2 = st.columns([1, 1.3])
        with c1:
            fig_sizes = px.bar(sizes, x="segment", y="employees", height=380,
                               labels={"segment": "Segment", "employees": "Employees"},
                               title="Segment size")
            st.plotly_chart(fig_sizes, use_container_width=True, key="segment_sizes")

        # B) Segments by location (readable locations only)
        def readable(v: str) -> bool:
            if re.match(r"^\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*$", v):  # lat,lon
                return False
            if re.match(r"^\s*-?\d+(\.\d+)?\s*$", v):  # numeric only
                return False
            return True

        loc_options = [v for v in sorted(loc_s.dropna().unique()) if readable(v)]
        default_locs = top_n(pd.Series(loc_options), n=15)
        picks = st.multiselect("Locations (leave empty to show all)", options=loc_options, default=default_locs)

        view = pd.DataFrame({"segment": seg_s, "location": loc_s})
        if picks:
            view = view[view["location"].isin(picks)]

        with c2:
            if view.empty:
                st.info("No data for the selected locations.")
            else:
                counts = view.value_counts(["location","segment"]).reset_index(name="employees")
                fig_geo = px.bar(counts, x="location", y="employees", color="segment", height=380,
                                 labels={"location": "Location", "employees": "Employees", "segment": "Segment"},
                                 title="Segments by location")
                st.plotly_chart(fig_geo, use_container_width=True, key="segments_by_location")

# =========================================================
# TAB 4 — About & Custom GPT
# =========================================================
with tab4:
    st.subheader("About this Project")
    st.write(
        "This dashboard summarizes key insights from a workforce analytics study: enrollments across regions, "
        "training outcomes by delivery mode, employee segmentation (K-Means), and program improvements."
    )
    st.markdown("**Technologies:** pandas, scikit-learn, Plotly, Streamlit")

    st.markdown("---")
    st.subheader("Custom GPT for Targeted Employee Messaging")
    st.write(
        "I also built a Custom GPT to generate employee-segment-aware content (e.g., program flyers) that aligns with "
        "motivation themes such as career advancement, operational excellence, skill development, and re-engagement."
    )
    # 👉 Replace this with your actual Custom GPT link
    CUSTOM_GPT_URL = st.text_input(
        "Custom GPT Link (optional)",
        value="https://your-custom-gpt-link.example",
        help="Paste your live Custom GPT link here to make it easy for recruiters to try."
    )
    if CUSTOM_GPT_URL and CUSTOM_GPT_URL.startswith("http"):
        st.link_button("Open Custom GPT", CUSTOM_GPT_URL, type="primary")

    st.markdown("##### Segment-Aligned Messaging Examples")
    st.markdown("- **Career Advancement:** leadership training, growth pathways, high-visibility projects.")
    st.markdown("- **Operational Excellence:** productivity methods, structured execution, real-world applicability.")
    st.markdown("- **Skill Development:** expert-led sessions, specialization, future-proof capabilities.")
    st.markdown("- **Re-engagement:** personal growth, wellness, engaging formats to rekindle motivation.")

    st.caption("The messaging strategy and examples are based on my Custom GPT executive summary and flyers.")
