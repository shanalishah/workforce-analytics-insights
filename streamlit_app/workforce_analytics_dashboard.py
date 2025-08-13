# streamlit_app/workforce_analytics_dashboard.py
from pathlib import Path
import re
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- Page ----------
st.set_page_config(
    page_title="Workforce Analytics — Clear Insights",
    page_icon="📊",
    layout="wide",
)

st.title("Workforce Analytics — Clear Insights")
st.caption("A focused view on enrollments, training outcomes, segments, and program improvements.")

# ---------- Paths & loaders ----------
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

# ---------- Helpers (robust & simple) ----------
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

def looks_human_location(s: pd.Series) -> bool:
    s = s.dropna().astype(str)
    if s.empty: return False
    pct_letters = (s.str.contains(r"[A-Za-z]", regex=True)).mean()
    numeric_like = (s.str.match(r"^\s*-?\d+(\.\d+)?(,\s*-?\d+(\.\d+)?)*\s*$", na=False)).mean()
    return pct_letters >= 0.7 and numeric_like <= 0.05

def best_location_col(df: pd.DataFrame):
    for cand in ["city", "office", "region", "country", "location"]:
        for c in df.columns:
            if c.lower() == cand and looks_human_location(df[c]):
                return c
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) and looks_human_location(df[c]):
            return c
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            return c
    return df.columns[0]

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

# ---------- KPIs ----------
kpis = {}

enr, _ = read_any("country_enrollment_summary.csv")
if enr is None:
    enr, _ = read_any("Country-wise_Enrollment_Summary.csv")
if enr is not None and not enr.empty:
    c_country = guess_col(enr, ["country", "country_name", "nation"])
    c_enroll  = guess_col(enr, ["enrollments", "enrollment", "total_enrollments", "count"], prefer_text=False)
    enr[c_enroll] = pd.to_numeric(enr[c_enroll], errors="coerce")
    enr = enr.dropna(subset=[c_enroll])
    if not enr.empty:
        kpis["Total enrollments"] = f"{int(enr[c_enroll].sum()):,}"
        kpis["Countries represented"] = as_text_series(enr, c_country).nunique()

ass, _ = read_any("course_assessment_by_course.csv")
if ass is None:
    ass, _ = read_any("Course_wise_assessment.csv")
if ass is not None and not ass.empty:
    kpis["Courses analyzed"] = as_text_series(ass, best_course_col(ass)).nunique()

seg, _ = read_any("pca_kmeans_results.xlsx")
if seg is None:
    seg, _ = read_any("pca_kmeans_results.csv")
if seg is not None and not seg.empty:
    seg_col_guess = guess_col(seg, ["segment", "cluster", "group", "label"])
    kpis["Employee segments"] = as_text_series(seg, seg_col_guess).nunique()

exp, _ = read_any("experiment_curriculum_cleaned.csv")
if exp is None:
    exp, _ = read_any("nls_experiment_cleaned.csv")
if exp is not None and not exp.empty:
    pre_p  = guess_col(exp, ["pre_proficiency","proficiency_pre","pre_prof"], prefer_text=False)
    post_p = guess_col(exp, ["post_proficiency","proficiency_post","post_prof"], prefer_text=False)
    if pre_p and post_p:
        delta = pd.to_numeric(exp[post_p], errors="coerce") - pd.to_numeric(exp[pre_p], errors="coerce")
        if delta.notna().any():
            kpis["Median proficiency Δ"] = f"{delta.median():.2f}"

if kpis:
    cols = st.columns(min(4, len(kpis)))
    for (k, v), c in zip(kpis.items(), cols):
        c.metric(k, v)

st.markdown("---")

# =========================================================
# 1) Overview — Enrollments by Country
# =========================================================
st.subheader("Enrollments by Country")

if enr is not None and not enr.empty:
    c_country = guess_col(enr, ["country", "country_name", "nation"])
    c_enroll  = guess_col(enr, ["enrollments", "enrollment", "total_enrollments", "count"], prefer_text=False)
    enr[c_enroll] = pd.to_numeric(enr[c_enroll], errors="coerce")
    enr = enr.dropna(subset=[c_enroll])

    country_s = as_text_series(enr, c_country)
    top10 = enr.assign(_country=country_s).sort_values(c_enroll, ascending=False)["_country"].head(10).tolist()
    countries = st.multiselect(
        "Countries to include (default = top 10 by enrollments)",
        options=sorted(country_s.unique()),
        default=top10,
    )
    order = st.radio("Sort by", ["Enrollments (desc)", "Country (A–Z)"], horizontal=True)

    view = enr.assign(_country=country_s)
    if countries:
        view = view[view["_country"].isin(countries)]
    view = view.sort_values(c_enroll if order.startswith("Enrollments") else "_country",
                            ascending=not order.startswith("Enrollments"))

    if view.empty:
        st.info("No countries selected.")
    else:
        fig = px.bar(view, x="_country", y=c_enroll, height=420,
                     labels={"_country": "Country", c_enroll: "Enrollments"},
                     title="Enrollments for selected countries")
        st.plotly_chart(fig, use_container_width=True, key="enrollments_by_country")
else:
    st.info("Add `country_enrollment_summary.csv` to `data/analysis-outputs/`.")

st.markdown("---")

# =========================================================
# 2) Course Outcomes — by Delivery Mode
# =========================================================
st.subheader("Training Outcomes by Course & Delivery Mode")

if ass is not None and not ass.empty:
    col_course = best_course_col(ass)
    course_s = as_text_series(ass, col_course)
    col_delivery = guess_col(ass, ["delivery","mode","format","delivery_mode"])
    delivery_s = as_text_series(ass, col_delivery)

    # Ensure outcomes exist; compute if needed
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
        st.info("No numeric outcome available for this course/outcome. Try a different course or check the file headers.")
    else:
        # Show both a bar of means and a distribution boxplot side-by-side
        c1, c2 = st.columns([1.1, 1])
        with c1:
            mean_df = sub.groupby("_delivery", as_index=False)[ycol].mean()
            fig_bar = px.bar(mean_df, x="_delivery", y=ycol, height=380,
                             labels={"_delivery": "Delivery mode", ycol: outcome_pick},
                             title=f"Average {outcome_pick} by delivery — {course_pick}")
            st.plotly_chart(fig_bar, use_container_width=True, key="course_outcomes_mean")
        with c2:
            fig_box = px.box(sub, x="_delivery", y=ycol, points="all", height=380,
                             labels={"_delivery": "Delivery mode", ycol: outcome_pick},
                             title="Distribution")
            st.plotly_chart(fig_box, use_container_width=True, key="course_outcomes_dist")
else:
    st.info("Add `course_assessment_by_course.csv` to `data/analysis-outputs/`.")

st.markdown("---")

# =========================================================
# 3) Segments — Size & Geography
# =========================================================
st.subheader("Employee Segments")

if seg is not None and not seg.empty:
    seg_col = guess_col(seg, ["segment","cluster","group","label"])
    loc_col = best_location_col(seg)

    seg_s = as_text_series(seg, seg_col)
    loc_s = as_text_series(seg, loc_col)

    # Segment sizes
    seg_sizes = pd.DataFrame({"segment": seg_s}).value_counts().reset_index(name="count")
    seg_sizes.columns = ["segment", "count"]

    c1, c2 = st.columns([1, 1.2])
    with c1:
        fig_seg = px.bar(seg_sizes, x="segment", y="count", height=380,
                         labels={"segment": "Segment", "count": "Employees"},
                         title="Segment sizes")
        st.plotly_chart(fig_seg, use_container_width=True, key="segment_sizes")

    # Geography: filter locations (readable only; drop coordinate-like/numeric codes)
    def is_readable_loc(v: str) -> bool:
        if re.match(r"^\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*$", v):  # lat,lon
            return False
        if re.match(r"^\s*-?\d+(\.\d+)?\s*$", v):  # pure number
            return False
        return True

    loc_options = [v for v in sorted(loc_s.dropna().unique()) if is_readable_loc(v)]
    top_locs = pd.Series(loc_s).value_counts().index.tolist()[:15]
    top_locs = [v for v in top_locs if v in loc_options]

    picks = st.multiselect("Filter locations (leave empty to show all)", options=loc_options, default=top_locs)
    view = pd.DataFrame({"segment": seg_s, "location": loc_s})
    if picks:
        view = view[view["location"].isin(picks)]

    with c2:
        if view.empty:
            st.info("No data for the selected locations.")
        else:
            # stacked bar: employees by location, colored by segment
            counts = view.value_counts(["location","segment"]).reset_index(name="count")
            fig_geo = px.bar(counts, x="location", y="count", color="segment", height=380,
                             labels={"location": "Location", "count": "Employees", "segment": "Segment"},
                             title="Segments by location")
            st.plotly_chart(fig_geo, use_container_width=True, key="segments_by_location")
else:
    st.info("Add `pca_kmeans_results.xlsx/csv` to `data/analysis-outputs/`.")

st.markdown("---")

# =========================================================
# 4) Program Improvements — Pre → Post
# =========================================================
st.subheader("Program Improvements (Pre → Post)")

if exp is not None and not exp.empty:
    def idx_of(df, col):
        try:
            return list(df.columns).index(col) if col in df.columns else 0
        except Exception:
            return 0

    prog_guess  = guess_col(exp, ['program','curriculum','group'])
    pre_p_guess = guess_col(exp, ['pre_proficiency','proficiency_pre','pre_prof'], prefer_text=False)
    post_p_gs   = guess_col(exp, ['post_proficiency','proficiency_post','post_prof'], prefer_text=False)
    pre_a_guess = guess_col(exp, ['pre_applications','applications_pre','pre_apps'], prefer_text=False)
    post_a_gs   = guess_col(exp, ['post_applications','applications_post','post_apps'], prefer_text=False)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: col_prog = st.selectbox("Program field", list(exp.columns), index=idx_of(exp, prog_guess))
    with c2: col_pre_p = st.selectbox("Pre proficiency", list(exp.columns), index=idx_of(exp, pre_p_guess))
    with c3: col_post_p= st.selectbox("Post proficiency", list(exp.columns), index=idx_of(exp, post_p_gs))
    with c4: col_pre_a = st.selectbox("Pre applications", list(exp.columns), index=idx_of(exp, pre_a_guess))
    with c5: col_post_a= st.selectbox("Post applications", list(exp.columns), index=idx_of(exp, post_a_gs))

    dfv = exp.copy()
    dfv["Δ proficiency"]  = pd.to_numeric(dfv[col_post_p], errors="coerce") - pd.to_numeric(dfv[col_pre_p], errors="coerce")
    dfv["Δ applications"] = pd.to_numeric(dfv[col_post_a], errors="coerce") - pd.to_numeric(dfv[col_pre_a], errors="coerce")

    metric = st.radio("Outcome", ["Δ proficiency", "Δ applications"], horizontal=True)
    prog_s = as_text_series(dfv, col_prog)
    options = sorted(prog_s.dropna().unique())
    chosen = st.multiselect("Programs (leave empty to include all)", options=options, default=[])

    view = dfv.assign(_prog=prog_s)
    if chosen:
        view = view[view["_prog"].isin(chosen)]
    view = view.dropna(subset=[metric])

    if view.empty:
        st.info("No numeric results for the chosen fields.")
    else:
        fig_exp = px.box(view, x="_prog", y=metric, color="_prog", points="all", height=420,
                         labels={"_prog": "Program", metric: metric},
                         title=f"{metric} by program")
        st.plotly_chart(fig_exp, use_container_width=True, key="experiment_deltas")
else:
    st.info("Add `experiment_curriculum_cleaned.csv` (or `nls_experiment_cleaned.csv`) to `data/analysis-outputs/`.")
