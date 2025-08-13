# streamlit_app/workforce_analytics_dashboard.py
from pathlib import Path
import re
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Workforce Analytics — Interactive Insights",
    page_icon="📊",
    layout="wide",
    menu_items={"About": "Workforce Analytics — Enrollments, Segments (PCA+KMeans), and Program Improvements"},
)

# Subtle CSS
st.markdown("""
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
.section-gap { margin-top: .75rem; margin-bottom: .75rem; }
</style>
""", unsafe_allow_html=True)

# ---------------- QID -> text mapping (embedded) ----------------
Q_MAPPING = {
    "Q1":  "I prefer training that helps me enhance my job performance, without the need for group interaction.",
    "Q2":  "I’m motivated to learn new things that expand my abilities and knowledge beyond my day-to-day responsibilities.",
    "Q3":  "I am more interested in training that enhances my individual skills than in activities focused on team-building.",
    "Q4":  "I’m not interested in using training for networking or career changes; I value it more for improving my day-to-day work.",
    "Q5":  "I enjoy training that broadens my knowledge and helps me grow, even if it’s not directly related to my current role.",
    "Q6":  "I am motivated by training that not only helps me perform better in my current role but also enhances my overall skills and knowledge.",
    "Q7":  "I value training that makes me more effective in my job while also helping me grow professionally.",
    "Q8":  "I find value in training that supports my personal development and inspires me to grow in new directions.",
    "Q9":  "I look for training opportunities that allow me to develop my role-specific skills and learn new concepts that can help me in the future.",
    "Q10": "I’m more interested in improving my current role than in preparing for a new position or career change.",
    "Q11": "I find the most value in training that directly impacts my role, rather than in sessions involving group discussions.",
    "Q12": "I don’t see training as a way to transition into a new job; I prefer to use it to build on what I’m already doing.",
}

# ---------------- Paths & loaders ----------------
ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [
    ROOT / "data" / "analysis-outputs",
    ROOT / "data" / "processed",
    ROOT / "data" / "raw",   # optional for extra maps, if present
]

@st.cache_data(show_spinner=False)
def find_path(name: str):
    for base in SEARCH_DIRS:
        p = base / name
        if p.exists():
            return p
    return None

@st.cache_data(show_spinner=False)
def read_any(name: str):
    p = find_path(name)
    if p is None:
        return None, None
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, low_memory=False), p
    elif p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p), p
    else:
        return None, None

# ---------------- Helpers ----------------
def guess_col(df: pd.DataFrame, candidates, *, prefer_text=True):
    """Return first matching candidate (case-insensitive). Fallback to first text col, else first col."""
    name_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = name_map.get(str(cand).lower())
        if c is not None:
            if prefer_text and not pd.api.types.is_object_dtype(df[c]):
                continue
            return c
    if prefer_text:
        for c in df.columns:
            if pd.api.types.is_object_dtype(df[c]):
                return c
    return df.columns[0] if len(df.columns) else None

def best_course_column(df: pd.DataFrame):
    for cand in ["course_title", "course_name"]:
        for col in df.columns:
            if col.lower() == cand:
                return col
    for cand in ["course", "course_id"]:
        for col in df.columns:
            if col.lower() == cand:
                return col
    return df.columns[0]

def looks_textual_location(s: pd.Series) -> bool:
    s = s.dropna().astype(str)
    if s.empty: return False
    pct_letters = (s.str.contains(r"[A-Za-z]", regex=True)).mean()
    unique_ratio = s.nunique() / max(1, len(s))
    return pct_letters >= 0.7 and unique_ratio <= 0.9

COORD_RE = re.compile(r"^\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*$")
NUMERIC_ONLY_RE = re.compile(r"^\s*-?\d+(\.\d+)?\s*$")

def best_location_column(df: pd.DataFrame):
    for cand in ["location", "city", "office", "region", "country"]:
        for col in df.columns:
            if col.lower() == cand and looks_textual_location(df[col]):
                return col
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) and looks_textual_location(df[col]):
            return col
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            return col
    return df.columns[0]

def apply_q_mapping(series: pd.Series) -> pd.Series:
    """Replace Qn with full text using embedded Q_MAPPING. Falls back to 'Question n' if unseen."""
    def repl(x):
        s = str(x).strip()
        key = s.upper()
        if key in Q_MAPPING:
            return Q_MAPPING[key]
        m = re.match(r"^(Q\d+)", key)
        if m and m.group(1) in Q_MAPPING:
            return Q_MAPPING[m.group(1)]
        m2 = re.match(r"^Q(\d+)", key)
        if m2:
            return f"Question {m2.group(1)}"
        return s
    return series.map(repl)

def as_text_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Always return a clean string Series for a column, even if duplicate names exist."""
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]  # take first if duplicates
    return obj.astype(str).str.strip()

def normalize_pca_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize different PCA loadings shapes to tidy: columns = ['feature','component','loading'].
    Avoids mixing non-feature columns (IDs, city, cluster, etc.).
    """
    cols = list(df.columns)
    lower = [c.lower() for c in cols]

    # Case 1: Already tidy (feature + component [+ loading])
    if "feature" in lower or "variable" in lower:
        feat_col = cols[lower.index("feature")] if "feature" in lower else cols[lower.index("variable")]
        if "component" in lower and "loading" in lower:
            comp_col = cols[lower.index("component")]
            load_col = cols[lower.index("loading")]
            tidy = df[[feat_col, comp_col, load_col]].rename(columns={feat_col:"feature", comp_col:"component", load_col:"loading"}).copy()
        else:
            comp_col = cols[lower.index("component")] if "component" in lower else None
            others = [c for c in cols if c not in {feat_col, comp_col} and pd.api.types.is_numeric_dtype(df[c])]
            if comp_col:
                # features are rows, multiple loading columns -> melt
                tidy = df.melt(id_vars=[feat_col, comp_col], var_name="unknown", value_name="loading")
                tidy = tidy[[feat_col, comp_col, "loading"]].rename(columns={feat_col:"feature", comp_col:"component"})
            else:
                # features are rows, PC columns -> melt all numeric columns
                use_cols = [feat_col] + others
                tidy = df[use_cols].melt(id_vars=feat_col, var_name="component", value_name="loading").rename(columns={feat_col:"feature"})
        return tidy

    # Case 2: Component column present and many Qnn columns (features in columns, components in rows)
    qid_cols = [c for c in cols if re.match(r"(?i)^Q\d+", str(c).strip())]
    if "component" in lower and len(qid_cols) >= max(3, len(cols)//3):
        comp_col = cols[lower.index("component")]
        tidy = df.melt(id_vars=comp_col, var_name="feature", value_name="loading").rename(columns={comp_col:"component"})
        return tidy

    # Case 3: First column is component (values like PC1), other columns are features
    first = cols[0]
    first_vals = df[first].astype(str).str.upper().str.match(r"^PC\d+|^COMPONENT")
    if first_vals.mean() >= 0.5:
        tidy = df.melt(id_vars=first, var_name="feature", value_name="loading").rename(columns={first:"component"})
        return tidy

    # Case 4: Likely features in first column and PC columns after
    pc_cols = [c for c in cols if re.match(r"(?i)^PC\d+", str(c).strip())]
    if pc_cols:
        tidy = df.melt(id_vars=first, var_name="component", value_name="loading").rename(columns={first:"feature"})
        return tidy

    # Fallback: assume first is feature, melt numeric columns
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        tidy = df.melt(id_vars=first, var_name="component", value_name="loading").rename(columns={first:"feature"})
        return tidy

    # Last resort: return empty tidy
    return pd.DataFrame(columns=["feature","component","loading"])

def clean_feature_labels(series: pd.Series) -> pd.Series:
    """Map QIDs to full text and drop clearly non-feature tokens."""
    s = apply_q_mapping(series)
    # Remove obviously non-feature labels
    bad_tokens = ("employee", "emp_id", "employee_id", "city", "office", "region", "cluster", "segment", "pc", "component", "response")
    mask = ~s.str.lower().str.contains("|".join(bad_tokens))
    return s.where(mask)

# ---------------- Header ----------------
st.title("Workforce Analytics — Interactive Insights")
st.caption("Explore training enrollments, employee segments (PCA + KMeans), and curriculum experiment outcomes.")

# ---------------- KPIs (useful) ----------------
kpi_vals = {}
# Enrollments
enr_df, _ = read_any("country_enrollment_summary.csv")
if enr_df is None:
    enr_df, _ = read_any("Country-wise_Enrollment_Summary.csv")
if enr_df is not None and not enr_df.empty:
    c_country = guess_col(enr_df, ["country", "country_name", "nation"])
    c_enroll  = guess_col(enr_df, ["enrollments", "enrollment", "total_enrollments", "count"], prefer_text=False)
    enr_df[c_enroll] = pd.to_numeric(enr_df[c_enroll], errors="coerce")
    enr_df = enr_df.dropna(subset=[c_enroll])
    if not enr_df.empty:
        kpi_vals["Total enrollments"] = f"{int(enr_df[c_enroll].sum()):,}"
        kpi_vals["Countries represented"] = as_text_series(enr_df, c_country).nunique()
# Courses
ass_df, _ = read_any("course_assessment_by_course.csv")
if ass_df is None:
    ass_df, _ = read_any("Course_wise_assessment.csv")
if ass_df is not None and not ass_df.empty:
    c_course = best_course_column(ass_df)
    kpi_vals["Courses analyzed"] = as_text_series(ass_df, c_course).nunique()
# Segments
seg_df, _ = read_any("pca_kmeans_results.xlsx")
if seg_df is None:
    seg_df, _ = read_any("pca_kmeans_results.csv")
if seg_df is not None and not seg_df.empty:
    c_seg = guess_col(seg_df, ["segment", "cluster", "group", "label"])
    if c_seg:
        kpi_vals["Employee segments discovered"] = as_text_series(seg_df, c_seg).nunique()
# Experiment
exp_df, _ = read_any("experiment_curriculum_cleaned.csv")
if exp_df is None:
    exp_df, _ = read_any("nls_experiment_cleaned.csv")
if exp_df is not None and not exp_df.empty:
    pre_p  = guess_col(exp_df, ["pre_proficiency", "proficiency_pre", "pre_prof"], prefer_text=False)
    post_p = guess_col(exp_df, ["post_proficiency", "proficiency_post", "post_prof"], prefer_text=False)
    if pre_p and post_p:
        imp = pd.to_numeric(exp_df[post_p], errors="coerce") - pd.to_numeric(exp_df[pre_p], errors="coerce")
        if not imp.dropna().empty:
            kpi_vals["Median proficiency improvement"] = f"{imp.median():.2f}"

if kpi_vals:
    cols = st.columns(min(4, len(kpi_vals)))
    for (label, value), col in zip(kpi_vals.items(), cols):
        col.metric(label, value)

st.markdown("---")

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs([
    "📍 Enrollments by Country",
    "🧩 Employee Segments (PCA + KMeans)",
    "🧪 Program Improvements (Experiment)",
])

# ==================== TAB 1: ENROLLMENTS BY COUNTRY ====================
with tab1:
    st.markdown("### Enrollments by Country")
    df_u, path_u = read_any("country_enrollment_summary.csv")
    if df_u is None:
        df_u, path_u = read_any("Country-wise_Enrollment_Summary.csv")

    if df_u is not None and not df_u.empty:
        col_country = guess_col(df_u, ["country", "country_name", "nation"])
        col_enroll  = guess_col(df_u, ["enrollments", "enrollment", "total_enrollments", "count"], prefer_text=False)
        df_u[col_enroll] = pd.to_numeric(df_u[col_enroll], errors="coerce")
        df_u = df_u.dropna(subset=[col_enroll])

        country_series = as_text_series(df_u, col_country)
        # Default top 10 by enrollments
        top10 = df_u.assign(_country=country_series).sort_values(col_enroll, ascending=False)["_country"].head(10).tolist()
        countries = st.multiselect(
            "Countries to include (default: top 10 by enrollments)",
            options=sorted(country_series.unique()),
            default=top10,
        )
        sort_by = st.radio("Sort bars by", ["Enrollments (descending)", "Country (A–Z)"], horizontal=True)

        filtered = df_u.assign(_country=country_series)
        if countries:
            filtered = filtered[filtered["_country"].isin(countries)]

        if sort_by.startswith("Enrollments"):
            filtered = filtered.sort_values(col_enroll, ascending=False)
        else:
            filtered = filtered.sort_values("_country", ascending=True)

        if filtered.empty:
            st.info("No countries selected.")
        else:
            fig = px.bar(
                filtered,
                x="_country", y=col_enroll,
                height=430,
                labels={"_country": "Country", col_enroll: "Enrollments"},
                title="Enrollments for selected countries",
            )
            st.plotly_chart(fig, use_container_width=True, key="enrollments_by_country")
        st.caption(f"Data source: {path_u.relative_to(ROOT) if path_u else 'N/A'}")
    else:
        st.info("Add `country_enrollment_summary.csv` to `data/analysis-outputs/`.")

    st.markdown("### Assessment Outcomes by Delivery Mode")
    df_a, path_a = read_any("course_assessment_by_course.csv")
    if df_a is None:
        df_a, path_a = read_any("Course_wise_assessment.csv")

    if df_a is not None and not df_a.empty:
        # Prefer course names; normalize & always treat as strings
        col_course = best_course_column(df_a)
        course_series = as_text_series(df_a, col_course)
        course_options = sorted(course_series.dropna().unique())
        course_sel = st.selectbox("Course", course_options)

        col_delivery = guess_col(df_a, ["delivery", "mode", "format", "delivery_mode"])
        delivery_series = as_text_series(df_a, col_delivery)

        # Robust numeric metrics — compute deltas if columns missing or empty
        col_dprof = guess_col(df_a, ["delta_proficiency", "prof_delta", "proficiency_delta"], prefer_text=False)
        col_dapps = guess_col(df_a, ["delta_applications", "apps_delta", "applications_delta"], prefer_text=False)

        def ensure_delta(df, delta_col, pre_cands, post_cands, fallback_name):
            ok = delta_col and (pd.to_numeric(df[delta_col], errors="coerce").notna().any())
            if ok:
                df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce")
                return delta_col
            pre = guess_col(df, pre_cands, prefer_text=False)
            post= guess_col(df, post_cands, prefer_text=False)
            if pre and post:
                df[fallback_name] = pd.to_numeric(df[post], errors="coerce") - pd.to_numeric(df[pre], errors="coerce")
                return fallback_name
            return None

        col_dprof = ensure_delta(df_a, col_dprof,
                                 ["pre_proficiency","proficiency_pre","pre_prof"],
                                 ["post_proficiency","proficiency_post","post_prof"],
                                 "__delta_prof")
        col_dapps = ensure_delta(df_a, col_dapps,
                                 ["pre_applications","applications_pre","pre_apps"],
                                 ["post_applications","applications_post","post_apps"],
                                 "__delta_apps")

        metric_pick = st.radio("Outcome", ["Change in Proficiency", "Change in Applications"], horizontal=True)
        ycol = col_dprof if metric_pick == "Change in Proficiency" else col_dapps

        # Filter safely (casefold on clean series)
        course_norm = course_series.str.casefold()
        mask = course_norm == str(course_sel).casefold()
        sub = df_a[mask].copy()
        sub["_delivery"] = delivery_series[mask].values  # aligned textual delivery

        if not ycol:
            st.warning("Could not locate or compute the selected outcome columns. Please check the input file headers.")
        elif sub[ycol].dropna().empty:
            st.warning("No numeric values for the selected outcome. Try switching the outcome or course.")
        else:
            fig_box = px.box(
                sub, x="_delivery", y=ycol, points="all", height=400,
                labels={"_delivery": "Delivery mode", ycol: metric_pick},
                title=f"{metric_pick} by delivery mode — {course_sel}",
            )
            st.plotly_chart(fig_box, use_container_width=True, key="assessment_delivery")
        st.caption(f"Data source: {path_a.relative_to(ROOT) if path_a else 'N/A'}")
    else:
        st.info("Add `course_assessment_by_course.csv` to `data/analysis-outputs/`.")

# ==================== TAB 2: EMPLOYEE SEGMENTS (PCA + KMEANS) ====================
with tab2:
    st.markdown("### Top Questions/Features by Component Influence")
    st.caption("Shows which questions or metrics contribute most to each component (higher |loading| ⇒ stronger influence).")

    comp, path_comp = read_any("pca_components.xlsx")
    if comp is None:
        comp, path_comp = read_any("pca_components.csv")

    if comp is not None and not comp.empty:
        tidy = normalize_pca_table(comp).copy()
        # Ensure correct dtypes
        tidy["loading"] = pd.to_numeric(tidy["loading"], errors="coerce")
        tidy = tidy.dropna(subset=["loading"])
        # Clean feature labels (no IDs/locations/clusters) & map QIDs
        tidy["feature"] = clean_feature_labels(tidy["feature"])
        tidy = tidy.dropna(subset=["feature"])

        # Components list should be components only (no Q/IDs/etc)
        comps = sorted(as_text_series(tidy, "component").dropna().unique())
        comp_sel = st.selectbox("Component", comps)

        topN = st.slider("Number of top questions/features", 5, 30, 15)
        sub = tidy[as_text_series(tidy, "component") == str(comp_sel)].copy()
        if sub.empty:
            st.info("No numeric loadings for this component.")
        else:
            sub["abs_loading"] = sub["loading"].abs()
            sub = sub.sort_values("abs_loading", ascending=False).head(topN)
            fig_pca = px.bar(
                sub, x="feature", y="loading", height=430,
                labels={"feature": "Question/Feature", "loading": "Loading"},
                title=f"Top {topN} questions/features — component {comp_sel}",
            )
            st.plotly_chart(fig_pca, use_container_width=True, key="pca_top_features")
        st.caption(f"Data source: {path_comp.relative_to(ROOT) if path_comp else 'N/A'}")
    else:
        st.info("Add `pca_components.xlsx` (or .csv) to `data/analysis-outputs/`.")

    st.markdown("### Segments by Location")
    seg, path_seg = read_any("pca_kmeans_results.xlsx")
    if seg is None:
        seg, path_seg = read_any("pca_kmeans_results.csv")

    if seg is not None and not seg.empty:
        # Let the user confirm fields (with smart defaults)
        seg_col_guess = guess_col(seg, ["segment", "cluster", "group", "label"])
        loc_col_guess = best_location_column(seg)

        c1, c2 = st.columns(2)
        with c1:
            seg_col = st.selectbox("Segment field", list(seg.columns),
                                   index=list(seg.columns).index(seg_col_guess) if seg_col_guess in seg.columns else 0)
        with c2:
            loc_col = st.selectbox("Location field", list(seg.columns),
                                   index=list(seg.columns).index(loc_col_guess) if loc_col_guess in seg.columns else 0)

        seg_series = as_text_series(seg, seg_col)
        loc_series = as_text_series(seg, loc_col)

        # Remove coordinate-like and numeric-only values from locations
        loc_options = [v for v in sorted(loc_series.dropna().unique())
                       if not COORD_RE.match(v) and not NUMERIC_ONLY_RE.match(v)]

        # Default = top 15 locations by frequency (readable)
        top_locs = pd.Series(loc_series).value_counts().index.tolist()[:15]
        top_locs = [v for v in top_locs if v in loc_options]

        pick_locs = st.multiselect("Locations to include", options=loc_options, default=top_locs,
                                   help="Clear all to include every location.")

        view = seg.assign(_loc=loc_series, _seg=seg_series)
        if pick_locs:
            view = view[view["_loc"].isin(pick_locs)]

        if view.empty:
            st.info("No rows match the selected locations.")
        else:
            fig_seg = px.histogram(
                view, x="_seg", color="_seg", height=400,
                labels={"_seg": "Segment"},
                title="Distribution of employee segments",
            )
            st.plotly_chart(fig_seg, use_container_width=True, key="segment_distribution")
        st.caption(f"Data source: {path_seg.relative_to(ROOT) if path_seg else 'N/A'}")
    else:
        st.info("Add `pca_kmeans_results.xlsx` (or .csv) to `data/analysis-outputs/`.")

# ==================== TAB 3: PROGRAM IMPROVEMENTS (EXPERIMENT) ====================
with tab3:
    st.markdown("### Pre → Post Program Improvements")

    exp, path_exp = read_any("experiment_curriculum_cleaned.csv")
    if exp is None:
        exp, path_exp = read_any("nls_experiment_cleaned.csv")

    if exp is not None and not exp.empty:
        st.caption("Confirm the correct fields if the defaults look off.")
        # Guesses (safe indexing)
        def idx_of(df, col_name):
            try:
                return list(df.columns).index(col_name) if col_name in df.columns else 0
            except Exception:
                return 0

        prog_guess  = guess_col(exp, ['program','curriculum','group'])
        pre_p_guess = guess_col(exp, ['pre_proficiency','proficiency_pre','pre_prof'], prefer_text=False)
        post_p_guess= guess_col(exp, ['post_proficiency','proficiency_post','post_prof'], prefer_text=False)
        pre_a_guess = guess_col(exp, ['pre_applications','applications_pre','pre_apps'], prefer_text=False)
        post_a_guess= guess_col(exp, ['post_applications','applications_post','post_apps'], prefer_text=False)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: col_prog   = st.selectbox("Program field", list(exp.columns), index=idx_of(exp, prog_guess))
        with c2: col_pre_p  = st.selectbox("Pre proficiency", list(exp.columns), index=idx_of(exp, pre_p_guess))
        with c3: col_post_p = st.selectbox("Post proficiency", list(exp.columns), index=idx_of(exp, post_p_guess))
        with c4: col_pre_a  = st.selectbox("Pre applications", list(exp.columns), index=idx_of(exp, pre_a_guess))
        with c5: col_post_a = st.selectbox("Post applications", list(exp.columns), index=idx_of(exp, post_a_guess))

        # Compute deltas robustly
        exp = exp.copy()
        exp["Δ proficiency"]  = pd.to_numeric(exp[col_post_p], errors="coerce") - pd.to_numeric(exp[col_pre_p], errors="coerce")
        exp["Δ applications"] = pd.to_numeric(exp[col_post_a], errors="coerce") - pd.to_numeric(exp[col_pre_a], errors="coerce")

        metric = st.radio("Outcome", ["Δ proficiency", "Δ applications"], horizontal=True)

        prog_series = as_text_series(exp, col_prog)
        progs = sorted(prog_series.dropna().unique())
        # Default = all programs (keep label professional)
        pick = st.multiselect("Programs", options=progs, default=[], help="Leave empty to include all programs.")

        view = exp.assign(_prog=prog_series)
        if pick:
            view = view[view["_prog"].isin(pick)]

        view_numeric = view.dropna(subset=[metric])
        if view_numeric.empty:
            st.warning("No numeric results for the chosen fields. Verify the selected columns.")
        else:
            fig_exp = px.box(
                view_numeric, x="_prog", y=metric, color="_prog", points="all", height=430,
                labels={"_prog": "Program", metric: metric},
                title=f"{metric} — by program",
            )
            st.plotly_chart(fig_exp, use_container_width=True, key="experiment_deltas")
        st.caption(f"Data source: {path_exp.relative_to(ROOT) if path_exp else 'N/A'}")
    else:
        st.info("Add `experiment_curriculum_cleaned.csv` to `data/analysis-outputs/`.")
