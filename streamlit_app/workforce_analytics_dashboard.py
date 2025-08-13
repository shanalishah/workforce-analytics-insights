# streamlit_app/app.py
from pathlib import Path
import re
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# Page setup (professional)
# =========================
st.set_page_config(page_title="Workforce Analytics Dashboard", page_icon="📊", layout="wide")
st.title("Workforce Analytics Dashboard")
st.caption("Enrollments, Training Outcomes, Segmentation, and Program Improvement Analysis")

# =========================
# Paths & file loading
# =========================
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
SEARCH_DIRS = [ROOT / "data" / "analysis-outputs", ROOT / "data" / "processed"]

FILENAMES = {
    "enroll": ["country_enrollment_summary.csv"],
    "ass_course": ["course_assessment_by_course.csv"],
    "ass_summed": ["course_assessment_summed.csv"],
    "ass_improve": ["assessment_improvement.csv"],
    "seg_results": ["pca_kmeans_results.csv"],
    "seg_city": ["city_cluster_distribution.csv"],
    "survey_loc": ["emp_survey_with_locations.csv"],
    "experiment": ["experiment_curriculum_cleaned.csv"],
}

@st.cache_data(show_spinner=False)
def find_first(paths):
    for name in paths:
        for base in SEARCH_DIRS:
            p = base / name
            if p.exists():
                return p
    return None

@st.cache_data(show_spinner=False)
def read_csv_any(kind):
    p = find_first(FILENAMES[kind])
    if p is None:
        return None, None
    try:
        df = pd.read_csv(p, low_memory=False)
        return df, p
    except Exception:
        return None, p

# =========================
# Helpers (robust)
# =========================
def as_text_series(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return obj.astype(str).str.strip()

def pick_col(df: pd.DataFrame, candidates, *, prefer_text=True):
    # exact then case-insensitive
    for c in df.columns:
        if c in candidates:
            if prefer_text and not pd.api.types.is_object_dtype(df[c]):
                continue
            return c
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = lower.get(cand.lower())
        if c is not None:
            if prefer_text and not pd.api.types.is_object_dtype(df[c]):
                continue
            return c
    if prefer_text:
        for c in df.columns:
            if pd.api.types.is_object_dtype(df[c]):
                return c
    return df.columns[0] if len(df.columns) else None

def ensure_numeric(df: pd.DataFrame, col: str):
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def ensure_delta(df, delta_col, pre_cands, post_cands, tmp_name):
    """Return (usable_col, message_if_missing)."""
    if delta_col and pd.to_numeric(df[delta_col], errors="coerce").notna().any():
        df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce")
        return delta_col, None
    pre  = pick_col(df, pre_cands,  prefer_text=False)
    post = pick_col(df, post_cands, prefer_text=False)
    if pre and post:
        df[tmp_name] = pd.to_numeric(df[post], errors="coerce") - pd.to_numeric(df[pre], errors="coerce")
        return tmp_name, None
    return None, f"Need one of {pre_cands} and one of {post_cands}"

def human_locations(values):
    out = []
    for v in pd.Series(values).dropna().unique():
        s = str(v).strip()
        if re.match(r"^\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*$", s):  # lat,lon
            continue
        if re.match(r"^\s*-?\d+(\.\d+)?\s*$", s):  # pure number
            continue
        out.append(s)
    return sorted(out)

def top_n(series: pd.Series, n=10):
    vc = series.value_counts(dropna=True)
    return vc.index[:n].tolist()

# =========================
# Load all needed data
# =========================
df_enr, p_enr = read_csv_any("enroll")
df_ass_course, p_ass_course = read_csv_any("ass_course")
df_ass_summed, p_ass_summed = read_csv_any("ass_summed")
df_ass_improve, p_ass_improve = read_csv_any("ass_improve")
df_seg_results, p_seg_results = read_csv_any("seg_results")
df_seg_city, p_seg_city = read_csv_any("seg_city")
df_survey_loc, p_survey_loc = read_csv_any("survey_loc")
df_exp, p_exp = read_csv_any("experiment")

# =========================
# KPI row (only facts)
# =========================
kpis = {}

if df_enr is not None and not df_enr.empty:
    c_country = pick_col(df_enr, ["country", "Country", "country_name", "nation"])
    c_enroll  = pick_col(df_enr, ["enrollments","Enrollments","enrollment","total_enrollments","count"], prefer_text=False)
    ensure_numeric(df_enr, c_enroll)
    kpis["Total Enrollments"] = f"{int(df_enr[c_enroll].sum(skipna=True)):,}"
    kpis["Countries Represented"] = as_text_series(df_enr, c_country).nunique()

if df_ass_course is not None and not df_ass_course.empty:
    c_course = pick_col(df_ass_course, ["course_title","course_name","course","CourseTitle","Course"])
    kpis["Courses Analyzed"] = as_text_series(df_ass_course, c_course).nunique()

if df_seg_results is not None and not df_seg_results.empty:
    c_seg = pick_col(df_seg_results, ["segment","cluster","group","label","Cluster"])
    kpis["Employee Segments"] = as_text_series(df_seg_results, c_seg).nunique()

if df_exp is not None and not df_exp.empty:
    pre_p  = pick_col(df_exp, ["pre_proficiency","proficiency_pre","pre_prof","PreProficiency"], prefer_text=False)
    post_p = pick_col(df_exp, ["post_proficiency","proficiency_post","post_prof","PostProficiency"], prefer_text=False)
    if pre_p and post_p:
        d = pd.to_numeric(df_exp[post_p], errors="coerce") - pd.to_numeric(df_exp[pre_p], errors="coerce")
        if d.notna().any():
            kpis["Median Proficiency Δ"] = f"{d.median():.2f}"

if kpis:
    cols = st.columns(min(4, len(kpis)))
    for (label, value), c in zip(kpis.items(), cols):
        c.metric(label, value)

st.markdown("---")

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["📍 Enrollments", "🎯 Training Outcomes", "🧩 Segmentation"])

# --------------------------------------------------------------------
# TAB 1 — Enrollments
# --------------------------------------------------------------------
with tab1:
    st.subheader("Enrollments by Country")
    if df_enr is None or df_enr.empty:
        st.info("Add `country_enrollment_summary.csv` to `data/analysis-outputs/`.")
    else:
        c_country = pick_col(df_enr, ["country", "Country", "country_name", "nation"])
        c_enroll  = pick_col(df_enr, ["enrollments","Enrollments","enrollment","total_enrollments","count"], prefer_text=False)
        ensure_numeric(df_enr, c_enroll)
        view = df_enr.dropna(subset=[c_enroll]).copy()
        country_s = as_text_series(view, c_country)

        default_countries = top_n(country_s, 10)
        picks = st.multiselect(
            "Countries (default shows top 10 by enrollments)",
            options=sorted(country_s.unique()),
            default=default_countries
        )
        order = st.radio("Sort by", ["Enrollments (desc)", "Country (A–Z)"], horizontal=True)

        view["_country"] = country_s
        if picks:
            view = view[view["_country"].isin(picks)]
        view = view.sort_values(c_enroll if order.startswith("Enrollments") else "_country",
                                ascending=not order.startswith("Enrollments"))

        if view.empty:
            st.info("No countries selected.")
        else:
            fig = px.bar(view, x="_country", y=c_enroll, height=420,
                         labels={"_country": "Country", c_enroll: "Enrollments"},
                         title="Enrollments for Selected Countries")
            st.plotly_chart(fig, use_container_width=True, key="enroll_bar")

st.markdown("---")

# --------------------------------------------------------------------
# TAB 2 — Training Outcomes
# --------------------------------------------------------------------
with tab2:
    st.subheader("Training Outcomes by Course and Delivery Mode")
    if (df_ass_course is None or df_ass_course.empty) and (df_ass_summed is None or df_ass_summed.empty) and (df_ass_improve is None or df_ass_improve.empty):
        st.info("Add at least one of: `course_assessment_by_course.csv`, `course_assessment_summed.csv`, or `assessment_improvement.csv`.")
    else:
        # Prefer the detailed per-row file; fall back to summed if needed
        df_src = None
        src_name = None
        if df_ass_course is not None and not df_ass_course.empty:
            df_src, src_name = df_ass_course.copy(), "course_assessment_by_course.csv"
        elif df_ass_improve is not None and not df_ass_improve.empty:
            df_src, src_name = df_ass_improve.copy(), "assessment_improvement.csv"
        else:
            df_src, src_name = df_ass_summed.copy(), "course_assessment_summed.csv"

        # Columns
        col_course   = pick_col(df_src, ["course_title","course_name","course","CourseTitle","Course"])
        course_s     = as_text_series(df_src, col_course)
        col_delivery = pick_col(df_src, ["delivery","mode","format","delivery_mode","DeliveryMode"])
        delivery_s   = as_text_series(df_src, col_delivery)

        # Outcomes: delta proficiency / delta applications (compute if needed)
        d_prof_guess = pick_col(df_src, ["delta_proficiency","prof_delta","proficiency_delta","DeltaProficiency"], prefer_text=False)
        d_apps_guess = pick_col(df_src, ["delta_applications","apps_delta","applications_delta","DeltaApplications"], prefer_text=False)

        d_prof, prof_msg = ensure_delta(
            df_src, d_prof_guess,
            ["pre_proficiency","proficiency_pre","pre_prof","PreProficiency"],
            ["post_proficiency","proficiency_post","post_prof","PostProficiency"],
            "__delta_prof"
        )
        d_apps, apps_msg = ensure_delta(
            df_src, d_apps_guess,
            ["pre_applications","applications_pre","pre_apps","PreApplications"],
            ["post_applications","applications_post","post_apps","PostApplications"],
            "__delta_apps"
        )

        course_pick = st.selectbox("Course", sorted(course_s.dropna().unique()))
        metric_pick = st.radio("Outcome", ["Change in Proficiency", "Change in Applications"], horizontal=True)
        ycol = d_prof if metric_pick == "Change in Proficiency" else d_apps

        mask = course_s.str.casefold() == str(course_pick).casefold()
        sub = df_src[mask].copy()
        sub["_delivery"] = delivery_s[mask].values

        has_metric = bool(ycol) and (ycol in sub.columns) and pd.to_numeric(sub[ycol], errors="coerce").notna().any()

        if not has_metric:
            reasons = []
            if metric_pick == "Change in Proficiency" and prof_msg: reasons.append(prof_msg)
            if metric_pick == "Change in Applications" and apps_msg: reasons.append(apps_msg)
            extra = (" " + " | ".join(set(reasons))) if reasons else ""
            st.info(f"No numeric outcome available for this course/outcome from `{src_name}`.{extra}")
        else:
            sub["_y"] = pd.to_numeric(sub[ycol], errors="coerce")
            sub = sub.dropna(subset=["_y"])
            c1, c2 = st.columns([1.1, 1])
            with c1:
                mean_df = sub.groupby("_delivery", as_index=False)["_y"].mean()
                fig_bar = px.bar(
                    mean_df, x="_delivery", y="_y", height=380,
                    labels={"_delivery": "Delivery Mode", "_y": metric_pick},
                    title=f"Average {metric_pick} — {course_pick}"
                )
                st.plotly_chart(fig_bar, use_container_width=True, key="outcomes_mean")
            with c2:
                fig_box = px.box(
                    sub, x="_delivery", y="_y", points="all", height=380,
                    labels={"_delivery": "Delivery Mode", "_y": metric_pick},
                    title="Distribution"
                )
                st.plotly_chart(fig_box, use_container_width=True, key="outcomes_dist")

st.markdown("---")

# --------------------------------------------------------------------
# TAB 3 — Segmentation
# --------------------------------------------------------------------
with tab3:
    st.subheader("Employee Segmentation")
    # Segment sizes (from results)
    if df_seg_results is None or df_seg_results.empty:
        st.info("Add `pca_kmeans_results.csv` to `data/analysis-outputs/` for segmentation results.")
    else:
        seg_col = pick_col(df_seg_results, ["segment","cluster","group","label","Cluster"])
        seg_s = as_text_series(df_seg_results, seg_col).astype(str)
        sizes = seg_s.value_counts(dropna=False).rename_axis("Segment").reset_index(name="Employees")
        sizes["Employees"] = sizes["Employees"].astype(int)

        c1, c2 = st.columns([1, 1.3])
        with c1:
            fig_sizes = px.bar(
                sizes, x="Segment", y="Employees", height=380,
                labels={"Segment": "Segment", "Employees": "Employees"},
                title="Segment Size"
            )
            st.plotly_chart(fig_sizes, use_container_width=True, key="seg_sizes")

        # Segments by location:
        # Prefer pre-aggregated city/cluster file if present; otherwise derive from results
        with c2:
            if df_seg_city is not None and not df_seg_city.empty:
                # Expected columns: city/location, cluster/segment, count
                loc_col = pick_col(df_seg_city, ["city","City","location","Location","office","region","country"])
                seg_col2= pick_col(df_seg_city, ["segment","Segment","cluster","Cluster","group","label"])
                cnt_col = pick_col(df_seg_city, ["count","Count","employees","Employees"], prefer_text=False)
                ensure_numeric(df_seg_city, cnt_col)
                dv = df_seg_city.copy()
                dv["_location"] = as_text_series(dv, loc_col)
                dv["_segment"]  = as_text_series(dv, seg_col2)
                dv = dv.dropna(subset=[cnt_col])
                # Filter locations (human-readable)
                loc_opts = human_locations(dv["_location"])
                if not loc_opts:
                    loc_opts = sorted(dv["_location"].unique().tolist())
                default_locs = top_n(pd.Series(dv["_location"]), 15)
                picks = st.multiselect("Filter by Location", options=loc_opts, default=[l for l in default_locs if l in loc_opts])
                view = dv if not picks else dv[dv["_location"].isin(picks)]
                if view.empty:
                    st.info("No data for the selected locations.")
                else:
                    fig_loc = px.bar(
                        view, x="_location", y=cnt_col, color="_segment", height=380,
                        labels={"_location": "Location", cnt_col: "Employees", "_segment": "Segment"},
                        title="Segments by Location"
                    )
                    st.plotly_chart(fig_loc, use_container_width=True, key="seg_by_loc_preagg")
            else:
                # Build from results if no pre-agg file
                # Guess a location column using the separate survey file if available
                loc_series = None
                if df_survey_loc is not None and not df_survey_loc.empty:
                    # Try to join on a common identifier if present, else just use its location column standalone
                    loc_guess = pick_col(df_survey_loc, ["city","City","location","Location","office","region","country"])
                    loc_series = as_text_series(df_survey_loc, loc_guess)
                else:
                    # Try to use a location-like column from seg results directly
                    loc_guess = pick_col(df_seg_results, ["city","City","location","Location","office","region","country"])
                    if loc_guess:
                        loc_series = as_text_series(df_seg_results, loc_guess)

                if loc_series is None or loc_series.empty:
                    st.info("No location information found. Add `city_cluster_distribution.csv` or include a location column.")
                else:
                    loc_clean = pd.Series(human_locations(loc_series))
                    if loc_clean.empty:
                        # fall back to raw unique strings
                        loc_clean = pd.Series(sorted(loc_series.dropna().unique()))

                    # Create counts by (location, segment) from seg results
                    tmp = pd.DataFrame({
                        "Location": as_text_series(df_seg_results, loc_guess).astype(str),
                        "Segment": seg_s
                    })
                    counts = tmp.value_counts(["Location","Segment"]).rename("Employees").reset_index()
                    # Filter
                    loc_opts = sorted(counts["Location"].unique().tolist())
                    default_locs = top_n(counts["Location"], 15)
                    picks = st.multiselect("Filter by Location", options=loc_opts, default=default_locs)
                    view = counts if not picks else counts[counts["Location"].isin(picks)]
                    if view.empty:
                        st.info("No data for the selected locations.")
                    else:
                        fig_loc = px.bar(
                            view, x="Location", y="Employees", color="Segment", height=380,
                            labels={"Location": "Location", "Employees": "Employees", "Segment": "Segment"},
                            title="Segments by Location"
                        )
                        st.plotly_chart(fig_loc, use_container_width=True, key="seg_by_loc_built")
