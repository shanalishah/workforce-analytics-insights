# streamlit_app/app.py
from pathlib import Path
import re
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# Page — professional
# =========================
st.set_page_config(page_title="Workforce Analytics Dashboard", page_icon="📊", layout="wide")
st.title("Workforce Analytics Dashboard")
st.caption("Enrollments, Training Outcomes, Segmentation, and Program Improvement Analysis")

# =========================
# Paths & file loading
# =========================
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
SEARCH_DIRS = [
    ROOT / "data" / "analysis-outputs",
    ROOT / "data" / "processed",
    ROOT / "data",  # extra fallback
]

FILENAMES = {
    "enroll": ["country_enrollment_summary.csv", "Country-wise_Enrollment_Summary.csv"],
    "ass_course": ["course_assessment_by_course.csv"],
    "ass_summed": ["course_assessment_summed.csv", "Course_wise_assessment.csv"],
    "ass_improve": ["assessment_improvement.csv", "AssessmentImprovement.csv"],
    "seg_results": ["pca_kmeans_results.csv", "pca_kmeans_results.xlsx"],
    "seg_city": ["city_cluster_distribution.csv", "City_Cluster_Distribution.csv"],
    "survey_loc": ["emp_survey_with_locations.csv", "surveyed_employees_with_full_locations.csv"],
    "experiment": ["experiment_curriculum_cleaned.csv", "nls_experiment_cleaned.csv"],
}

@st.cache_data(show_spinner=False)
def find_first(candidates):
    # search by exact name, then case-insensitive scan of directory
    for base in SEARCH_DIRS:
        for name in candidates:
            p = base / name
            if p.exists():
                return p
        # case-insensitive fallback
        for p in base.glob("**/*"):
            if p.is_file() and p.suffix.lower() in (".csv", ".xlsx", ".xls"):
                for name in candidates:
                    if p.name.lower() == name.lower():
                        return p
    return None

@st.cache_data(show_spinner=False)
def read_any(kind):
    p = find_first(FILENAMES[kind])
    if p is None:
        return None, None
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, low_memory=False), p
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p), p
    return None, p

# =========================
# Helpers
# =========================
TEXTY_HINTS = ("name", "title", "country", "city", "office", "region", "location", "segment", "cluster", "label")

def as_text(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return obj.astype(str).str.strip()

def pick_col(df: pd.DataFrame, candidates, *, prefer_text=True):
    # exact first, then ci
    for c in df.columns:
        if c in candidates:
            if prefer_text and not pd.api.types.is_object_dtype(df[c]): continue
            return c
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = lower.get(str(cand).lower())
        if c is not None:
            if prefer_text and not pd.api.types.is_object_dtype(df[c]): continue
            return c
    if prefer_text:
        for c in df.columns:
            if pd.api.types.is_object_dtype(df[c]): return c
    return df.columns[0] if len(df.columns) else None

def best_course_col(df):
    return (
        pick_col(df, ["course_title","course_name","course","CourseTitle","Course"])
        or pick_col(df, ["course_id","CourseID"], prefer_text=False)
        or df.columns[0]
    )

def best_delivery_col(df):
    return pick_col(df, ["delivery","mode","format","delivery_mode","DeliveryMode"])

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

def numeric_metric_candidates(df: pd.DataFrame):
    # numeric columns that aren't obvious IDs/text
    nums = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            lc = c.lower()
            if any(h in lc for h in TEXTY_HINTS):  # skip mis-detected texty names
                continue
            nums.append(c)
    # add popular delta names even if dtype not yet numeric (we’ll coerce later)
    for c in df.columns:
        if re.search(r"(delta|change|pre|post)", c, re.I) and c not in nums:
            nums.append(c)
    return nums

def ensure_numeric(series: pd.Series):
    return pd.to_numeric(series, errors="coerce")

# =========================
# Load datasets
# =========================
enr, p_enr = read_any("enroll")
ass_course, p_ass_course = read_any("ass_course")
ass_summed, p_ass_summed = read_any("ass_summed")
ass_improve, p_ass_improve = read_any("ass_improve")
seg_results, p_seg_results = read_any("seg_results")
seg_city, p_seg_city = read_any("seg_city")
survey_loc, p_survey_loc = read_any("survey_loc")
exp, p_exp = read_any("experiment")

# =========================
# KPI row — simple, useful
# =========================
kpis = {}

if enr is not None and not enr.empty:
    c_country = pick_col(enr, ["country","Country","country_name","nation"])
    c_enroll  = pick_col(enr, ["enrollments","Enrollments","enrollment","total_enrollments","count"], prefer_text=False)
    enr[c_enroll] = ensure_numeric(enr[c_enroll])
    kpis["Total Enrollments"] = f"{int(enr[c_enroll].sum(skipna=True)):,}"
    kpis["Countries Represented"] = as_text(enr, c_country).nunique()

# Replace “Median Proficiency Δ” with more useful metrics if experiment data exists
if exp is not None and not exp.empty:
    pre_p  = pick_col(exp, ["pre_proficiency","proficiency_pre","pre_prof","PreProficiency"], prefer_text=False)
    post_p = pick_col(exp, ["post_proficiency","proficiency_post","post_prof","PostProficiency"], prefer_text=False)
    if pre_p and post_p:
        delta_prof = ensure_numeric(exp[post_p]) - ensure_numeric(exp[pre_p])
        if delta_prof.notna().any():
            kpis["Avg Proficiency Change"] = f"{delta_prof.mean():.2f}"
            kpis["% Positive Change"] = f"{(delta_prof > 0).mean()*100:.0f}%"

if ass_course is not None and not ass_course.empty:
    kpis["Courses Analyzed"] = as_text(ass_course, best_course_col(ass_course)).nunique()

if seg_results is not None and not seg_results.empty:
    kpis["Employee Segments"] = as_text(seg_results, pick_col(seg_results, ["segment","cluster","group","label","Cluster"])).nunique()

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
    if enr is None or enr.empty:
        st.info("Add `country_enrollment_summary.csv` to `data/analysis-outputs/`.")
    else:
        c_country = pick_col(enr, ["country","Country","country_name","nation"])
        c_enroll  = pick_col(enr, ["enrollments","Enrollments","enrollment","total_enrollments","count"], prefer_text=False)
        enr[c_enroll] = ensure_numeric(enr[c_enroll])
        view = enr.dropna(subset=[c_enroll]).copy()
        country_s = as_text(view, c_country)

        default_countries = top_n(country_s, 10)
        picks = st.multiselect(
            "Countries (default shows top 10 by enrollments)",
            options=sorted(country_s.unique()),
            default=default_countries
        )
        order = st.radio("Sort by", ["Enrollments (desc)", "Country (A–Z)"], horizontal=True)

        view["_country"] = country_s
        if picks: view = view[view["_country"].isin(picks)]
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

    # pick a source that actually exists (by-course preferred)
    src_df = None
    src_name = None
    for df, name in [(ass_course, "course_assessment_by_course.csv"),
                     (ass_improve, "assessment_improvement.csv"),
                     (ass_summed, "course_assessment_summed.csv")]:
        if df is not None and not df.empty:
            src_df, src_name = df.copy(), name
            break

    if src_df is None:
        st.info("Add one of the following: `course_assessment_by_course.csv`, `assessment_improvement.csv`, or `course_assessment_summed.csv`.")
    else:
        # identify essential fields
        col_course   = best_course_col(src_df)
        col_delivery = best_delivery_col(src_df)
        course_s     = as_text(src_df, col_course) if col_course else pd.Series(dtype=str)
        delivery_s   = as_text(src_df, col_delivery) if col_delivery else pd.Series(dtype=str)

        # numeric metrics available to plot (let user choose)
        # we’ll coerce each candidate to numeric later
        metric_candidates = numeric_metric_candidates(src_df)
        # prefer deltas if present
        preferred_order = [c for c in metric_candidates if re.search(r"(delta|change)", c, re.I)] + \
                          [c for c in metric_candidates if not re.search(r"(delta|change)", c, re.I)]
        metrics = preferred_order if preferred_order else metric_candidates

        if not metrics:
            st.info(f"No numeric metrics detected in `{src_name}` to visualize.")
        else:
            c1, c2, c3 = st.columns([1.1, 1, 1])
            with c1:
                # allow “All courses” to avoid dead-ends
                course_opts = ["(All courses)"] + sorted(course_s.dropna().unique().tolist()) if not course_s.empty else ["(All courses)"]
                course_pick = st.selectbox("Course", course_opts)
            with c2:
                metric_pick = st.selectbox("Metric", metrics)
            with c3:
                group_field = st.selectbox("Group by", [col_delivery] if col_delivery else [], index=0 if col_delivery else None)

            # filter by course (optional)
            dfv = src_df.copy()
            y = ensure_numeric(dfv[metric_pick])
            dfv["_y"] = y
            if course_pick != "(All courses)" and not course_s.empty:
                dfv = dfv[course_s.str.casefold() == course_pick.casefold()]

            # require some numeric values
            dfv = dfv.dropna(subset=["_y"])
            if dfv.empty:
                st.info("No rows with numeric values for the chosen metric/course.")
            else:
                if group_field:
                    dfv["_grp"] = as_text(src_df, group_field).reindex(dfv.index)
                    mean_df = dfv.groupby("_grp", as_index=False)["_y"].mean()
                    fig = px.bar(mean_df, x="_grp", y="_y", height=400,
                                 labels={"_grp": group_field, "_y": metric_pick},
                                 title=f"{metric_pick} by {group_field}" + ("" if course_pick=="(All courses)" else f" — {course_pick}"))
                    st.plotly_chart(fig, use_container_width=True, key="outcomes_by_group")
                    # distribution
                    fig2 = px.box(dfv, x="_grp", y="_y", points="all", height=400,
                                  labels={"_grp": group_field, "_y": metric_pick},
                                  title="Distribution")
                    st.plotly_chart(fig2, use_container_width=True, key="outcomes_dist")
                else:
                    # simple bar by course if no delivery column exists
                    if not course_s.empty:
                        dfv["_course"] = course_s.reindex(dfv.index)
                        mean_df = dfv.groupby("_course", as_index=False)["_y"].mean()
                        fig = px.bar(mean_df, x="_course", y="_y", height=400,
                                     labels={"_course": "Course", "_y": metric_pick},
                                     title=f"{metric_pick} by Course")
                        st.plotly_chart(fig, use_container_width=True, key="outcomes_by_course")
                    else:
                        st.dataframe(dfv[["_y"]].rename(columns={"_y": metric_pick}))

st.markdown("---")

# --------------------------------------------------------------------
# TAB 3 — Segmentation
# --------------------------------------------------------------------
with tab3:
    st.subheader("Employee Segmentation")

    # prefer full results file
    if seg_results is None or seg_results.empty:
        st.info("Add `pca_kmeans_results.csv` to `data/analysis-outputs/` or `data/processed/`.")
    else:
        seg_col = pick_col(seg_results, ["segment","cluster","group","label","Cluster"])
        seg_labels = as_text(seg_results, seg_col).astype(str)

        # A) Segment sizes (integer counts)
        sizes = seg_labels.value_counts(dropna=False).rename_axis("Segment").reset_index(name="Employees")
        sizes["Employees"] = sizes["Employees"].astype(int)

        c1, c2 = st.columns([1, 1.3])
        with c1:
            fig_sizes = px.bar(
                sizes, x="Segment", y="Employees", height=380,
                labels={"Segment": "Segment", "Employees": "Employees"},
                title="Segment Size"
            )
            st.plotly_chart(fig_sizes, use_container_width=True, key="seg_sizes")

        # B) Segments by location — pre-aggregated file preferred
        with c2:
            if seg_city is not None and not seg_city.empty:
                loc_col = pick_col(seg_city, ["city","City","location","Location","office","region","country"])
                seg_col2= pick_col(seg_city, ["segment","Segment","cluster","Cluster","group","label"])
                cnt_col = pick_col(seg_city, ["count","Count","employees","Employees"], prefer_text=False)
                dv = seg_city.copy()
                dv["_location"] = as_text(dv, loc_col)
                dv["_segment"]  = as_text(dv, seg_col2)
                dv["_n"] = ensure_numeric(dv[cnt_col])
                dv = dv.dropna(subset=["_n"])
                loc_opts = human_locations(dv["_location"])
                if not loc_opts:
                    loc_opts = sorted(dv["_location"].unique().tolist())
                default_locs = top_n(dv["_location"], 15)
                picks = st.multiselect("Filter by Location", options=loc_opts, default=[l for l in default_locs if l in loc_opts])
                view = dv if not picks else dv[dv["_location"].isin(picks)]
                if view.empty:
                    st.info("No data for the selected locations.")
                else:
                    fig_loc = px.bar(
                        view, x="_location", y="_n", color="_segment", height=380,
                        labels={"_location": "Location", "_n": "Employees", "_segment": "Segment"},
                        title="Segments by Location"
                    )
                    st.plotly_chart(fig_loc, use_container_width=True, key="seg_by_loc_preagg")
            else:
                # derive from seg_results; try to find a location from survey_loc or seg_results itself
                loc_guess = None
                if survey_loc is not None and not survey_loc.empty:
                    loc_guess = pick_col(survey_loc, ["city","City","location","Location","office","region","country"])
                    loc_series = as_text(survey_loc, loc_guess)
                else:
                    loc_guess = pick_col(seg_results, ["city","City","location","Location","office","region","country"])
                    loc_series = as_text(seg_results, loc_guess) if loc_guess else pd.Series(dtype=str)

                if loc_series.empty:
                    st.info("No location column found. Add `city_cluster_distribution.csv` or include a location field.")
                else:
                    tmp = pd.DataFrame({"Location": loc_series, "Segment": seg_labels})
                    counts = tmp.value_counts(["Location","Segment"]).rename("Employees").reset_index()
                    loc_opts = human_locations(counts["Location"])
                    if not loc_opts:
                        loc_opts = sorted(counts["Location"].unique().tolist())
                    default_locs = top_n(counts["Location"], 15)
                    picks = st.multiselect("Filter by Location", options=loc_opts, default=[l for l in default_locs if l in loc_opts])
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
