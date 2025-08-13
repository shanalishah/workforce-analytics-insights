# streamlit_app/workforce_analytics_dashboard.py
from pathlib import Path
import re
import pandas as pd
import plotly.express as px
import streamlit as st

# =================== Page ===================
st.set_page_config(
    page_title="Workforce Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Workforce Analytics Dashboard")
st.caption("Enrollments, Training Outcomes, Segmentation, and Program Improvement Analysis.")

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

# =================== Helpers ===================
def as_text_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a single clean string Series for a column (handles duplicate headers)."""
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return obj.astype(str).str.strip()

def guess_col(df: pd.DataFrame, candidates, *, prefer_text=True):
    """Pick first matching column from candidates (case-insensitive)."""
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

def best_course_col(df: pd.DataFrame):
    for cand in ["course_title", "course_name", "course"]:
        for c in df.columns:
            if c.lower() == cand:
                return c
    for c in df.columns:
        if c.lower() == "course_id":
            return c
    return df.columns[0]

def _looks_human_location_series(s: pd.Series) -> bool:
    """Heuristic: mostly letters, not coordinates or numeric codes."""
    s = s.dropna().astype(str)
    if s.empty: return False
    has_letters = s.str.contains(r"[A-Za-z]", regex=True).mean()
    coord_like = s.str.match(r"^\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*$", na=False).mean()
    numeric_only = s.str.match(r"^\s*-?\d+(\.\d+)?\s*$", na=False).mean()
    return has_letters >= 0.6 and coord_like <= 0.02 and numeric_only <= 0.05

def best_location_col(df: pd.DataFrame):
    for cand in ["city", "office", "region", "country", "location"]:
        for c in df.columns:
            if c.lower() == cand:
                if _looks_human_location_series(as_text_series(df, c)):
                    return c
    # fallback: any textual column that looks like human place names
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) and _looks_human_location_series(as_text_series(df, c)):
            return c
    # final fallback
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            return c
    return df.columns[0]

def ensure_delta(df, delta_col, pre_cands, post_cands, tmp_name):
    """Ensure a delta metric exists; compute if necessary."""
    ok = delta_col and (pd.to_numeric(df[delta_col], errors="coerce").notna().any())
    if ok:
        df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce")
        return delta_col, None
    pre = guess_col(df, pre_cands, prefer_text=False)
    post = guess_col(df, post_cands, prefer_text=False)
    if pre and post:
        df[tmp_name] = pd.to_numeric(df[post], errors="coerce") - pd.to_numeric(df[pre], errors="coerce")
        return tmp_name, None
    return None, f"Missing both a delta column and pre/post columns: need one of {pre_cands} and {post_cands}"

def top_n_values(series: pd.Series, n=10):
    vc = pd.Series(series).value_counts(dropna=True)
    return [v for v in vc.index[:n].tolist()]

def readable_locations(options):
    """Drop coordinates and numeric codes."""
    out = []
    for v in options:
        s = str(v).strip()
        if re.match(r"^\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*$", s):  # lat,lon
            continue
        if re.match(r"^\s*-?\d+(\.\d+)?\s*$", s):  # numeric only
            continue
        out.append(s)
    return out

# =================== Load data (once) ===================
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
    kpi["Total Enrollments"] = f"{int(enr[c_enroll].sum(skipna=True)):,}"
    kpi["Countries Represented"] = as_text_series(enr, c_country).nunique()

if ass is not None and not ass.empty:
    kpi["Courses Analyzed"] = as_text_series(ass, best_course_col(ass)).nunique()

if seg is not None and not seg.empty:
    seg_col_guess = guess_col(seg, ["segment", "cluster", "group", "label"])
    kpi["Employee Segments"] = as_text_series(seg, seg_col_guess).nunique()

if exp is not None and not exp.empty:
    pre_p  = guess_col(exp, ["pre_proficiency","proficiency_pre","pre_prof"], prefer_text=False)
    post_p = guess_col(exp, ["post_proficiency","proficiency_post","post_prof"], prefer_text=False)
    if pre_p and post_p:
        delta = pd.to_numeric(exp[post_p], errors="coerce") - pd.to_numeric(exp[pre_p], errors="coerce")
        if delta.notna().any():
            kpi["Median Proficiency Δ"] = f"{delta.median():.2f}"

if kpi:
    cols = st.columns(min(4, len(kpi)))
    for (k, v), c in zip(kpi.items(), cols):
        c.metric(k, v)

st.markdown("---")

# =================== Tabs ===================
tab1, tab2, tab3 = st.tabs([
    "📍 Enrollments",
    "🎯 Training Outcomes",
    "🧩 Segmentation",
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
        default_countries = top_n_values(country_s, n=10)
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
                         title="Enrollments for Selected Countries")
            st.plotly_chart(fig, use_container_width=True, key="enrollments_by_country")

st.markdown("---")

# =========================================================
# TAB 2 — Training Outcomes
# =========================================================
with tab2:
    st.subheader("Training Outcomes by Course and Delivery Mode")
    if ass is None or ass.empty:
        st.info("Add `course_assessment_by_course.csv` to `data/analysis-outputs/`.")
    else:
        # Columns
        col_course   = best_course_col(ass)
        course_s     = as_text_series(ass, col_course)
        col_delivery = guess_col(ass, ["delivery","mode","format","delivery_mode"])
        delivery_s   = as_text_series(ass, col_delivery)

        # Ensure deltas (or compute)
        d_prof_guess = guess_col(ass, ["delta_proficiency","prof_delta","proficiency_delta"], prefer_text=False)
        d_apps_guess = guess_col(ass, ["delta_applications","apps_delta","applications_delta"], prefer_text=False)

        d_prof, prof_msg = ensure_delta(
            ass, d_prof_guess,
            ["pre_proficiency","proficiency_pre","pre_prof"],
            ["post_proficiency","proficiency_post","post_prof"],
            "__delta_prof"
        )
        d_apps, apps_msg = ensure_delta(
            ass, d_apps_guess,
            ["pre_applications","applications_pre","pre_apps"],
            ["post_applications","applications_post","post_apps"],
            "__delta_apps"
        )

        # Controls
        course_pick = st.selectbox("Course", sorted(course_s.dropna().unique()))
        metric_pick = st.radio("Outcome", ["Change in Proficiency", "Change in Applications"], horizontal=True)
        ycol = d_prof if metric_pick == "Change in Proficiency" else d_apps

        # Filter & validate
        mask = course_s.str.casefold() == str(course_pick).casefold()
        sub = ass[mask].copy()
        sub["_delivery"] = delivery_s[mask].values

        has_metric = (ycol is not None) and (ycol in sub.columns) and sub[ycol].notna().any()

        if not has_metric:
            reasons = []
            if metric_pick == "Change in Proficiency" and prof_msg: reasons.append(prof_msg)
            if metric_pick == "Change in Applications" and apps_msg: reasons.append(apps_msg)
            extra = (" " + " | ".join(set(reasons))) if reasons else ""
            st.info(f"No numeric outcome available for this course/outcome.{extra}")
        else:
            # Show mean + distribution
            c1, c2 = st.columns([1.1, 1])
            with c1:
                mean_df = sub.groupby("_delivery", as_index=False)[ycol].mean(numeric_only=True)
                mean_df[ycol] = pd.to_numeric(mean_df[ycol], errors="coerce")
                fig_bar = px.bar(
                    mean_df, x="_delivery", y=ycol, height=380,
                    labels={"_delivery": "Delivery Mode", ycol: metric_pick},
                    title=f"Average {metric_pick} — {course_pick}"
                )
                st.plotly_chart(fig_bar, use_container_width=True, key="course_outcomes_mean")
            with c2:
                sub_y = pd.to_numeric(sub[ycol], errors="coerce")
                sub_plot = sub.assign(_y=sub_y).dropna(subset=["_y"])
                fig_box = px.box(
                    sub_plot, x="_delivery", y="_y", points="all", height=380,
                    labels={"_delivery": "Delivery Mode", "_y": metric_pick},
                    title="Distribution"
                )
                st.plotly_chart(fig_box, use_container_width=True, key="course_outcomes_dist")

st.markdown("---")

# =========================================================
# TAB 3 — Segmentation
# =========================================================
with tab3:
    st.subheader("Employee Segments")
    if seg is None or seg.empty:
        st.info("Add `pca_kmeans_results.xlsx/csv` to `data/analysis-outputs/`.")
    else:
        # Fields (let code pick sensible defaults)
        seg_col_guess = guess_col(seg, ["segment","cluster","group","label"])
        loc_col_guess = best_location_col(seg)

        # Clean labels as strings to avoid 0–1 axes
        seg_s = as_text_series(seg, seg_col_guess).astype(str)
        loc_s = as_text_series(seg, loc_col_guess).astype(str)

        # A) Segment sizes (true counts, integers)
        sizes = (
            pd.DataFrame({"segment": seg_s})
            .value_counts()
            .reset_index(name="Employees")
        )
        sizes.columns = ["Segment", "Employees"]
        sizes = sizes.sort_values("Employees", ascending=False)

        c1, c2 = st.columns([1, 1.25])
        with c1:
            fig_sizes = px.bar(
                sizes, x="Segment", y="Employees", height=380,
                labels={"Segment": "Segment", "Employees": "Employees"},
                title="Segment Size"
            )
            st.plotly_chart(fig_sizes, use_container_width=True, key="segment_sizes")

        # B) Segments by location (readable location names only)
        loc_options = readable_locations(sorted(pd.Series(loc_s).dropna().unique()))
        # Default to top 15 by frequency among readable locations
        vc_loc = pd.Series(loc_s[loc_s.isin(loc_options)]).value_counts()
        default_locs = vc_loc.index[:15].tolist()

        picks = st.multiselect("Filter by Location (optional)", options=loc_options, default=default_locs)
        view = pd.DataFrame({"Segment": seg_s, "Location": loc_s})
        if picks:
            view = view[view["Location"].isin(picks)]

        with c2:
            if view.empty:
                st.info("No data for the selected locations.")
            else:
                counts = view.value_counts(["Location","Segment"]).reset_index(name="Employees")
                counts = counts.sort_values(["Location","Segment"])
                fig_geo = px.bar(
                    counts, x="Location", y="Employees", color="Segment", height=380,
                    labels={"Location": "Location", "Employees": "Employees", "Segment": "Segment"},
                    title="Segments by Location"
                )
                st.plotly_chart(fig_geo, use_container_width=True, key="segments_by_location")
