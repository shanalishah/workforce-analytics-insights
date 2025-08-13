# streamlit_app/workforce_analytics_dashboard.py
from pathlib import Path
import re
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# ─────────────────────────────
# App config
# ─────────────────────────────
st.set_page_config(page_title="Workforce Analytics Dashboard", page_icon="📊", layout="wide")
st.title("Workforce Analytics Dashboard")
st.caption("Enrollments · Training Outcomes · PCA (Dimensionality Reduction) · K-Means Segmentation")

# Reduce top padding and ensure headings aren’t clipped
st.markdown("""
<style>
.block-container { padding-top: 1.4rem; }
h1 { line-height: 1.18 !important; margin-top: 0.35rem !important; padding-bottom: 0.1rem !important; }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
SEARCH_DIRS = [
    ROOT / "data" / "analysis-outputs",
    ROOT / "data" / "processed",
    ROOT / "data",
    ROOT / "data" / "raw",
]

FILENAMES = {
    "enroll": ["country_enrollment_summary.csv", "Country-wise_Enrollment_Summary.csv"],
    "ass_course": ["course_assessment_by_course.csv"],
    "ass_summed": ["course_assessment_summed.csv"],
    "ass_improve": ["assessment_improvement.csv", "AssessmentImprovement.csv"],
    "seg_city_csv": ["city_cluster_distribution.csv", "City_Cluster_Distribution.csv"],
    "experiment": ["experiment_curriculum_cleaned.csv", "nls_experiment_cleaned.csv"],
    "pca_workbook": ["pca_components.xlsx"],  # Excel with sheets described below
    "survey_qs": ["survey_questions.xlsx", "survey_questions.csv"],
}

# ─────────────────────────────
# Small utilities
# ─────────────────────────────
def set_jump(target_id: str):
    st.session_state["jump_to"] = target_id

def jump_back():
    target = st.session_state.get("jump_to")
    if not target:
        return
    components.html(
        f"""
        <script>
          const el = document.getElementById("{target}");
          if (el) {{ el.scrollIntoView({{behavior: "instant", block: "start"}}); }}
        </script>
        """,
        height=0,
    )
    st.session_state["jump_to"] = ""  # clear after use

@st.cache_data(show_spinner=False)
def find_first(candidates):
    for base in SEARCH_DIRS:
        for name in candidates:
            p = base / name
            if p.exists():
                return p
    return None

def read_csv_forgiving(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            try:
                return pd.read_csv(path, low_memory=False, encoding=enc, engine="python")
            except Exception:
                continue
    return pd.read_csv(path, low_memory=False, encoding="utf-8", engine="python", on_bad_lines="skip", dtype=str)

@st.cache_data(show_spinner=False)
def read_any_csv(kind):
    p = find_first(FILENAMES[kind])
    if p is None:
        return None, None
    try:
        return read_csv_forgiving(p), p
    except Exception:
        try:
            return pd.read_excel(p), p
        except Exception:
            return None, p

def ensure_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def as_text(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return obj.astype(str).str.strip()

def wrap_text(s: str, width: int = 28) -> str:
    words, line, out = str(s).split(), "", []
    for w in words:
        if len(line) + len(w) + 1 <= width:
            line = (line + " " + w).strip()
        else:
            out.append(line); line = w
    if line: out.append(line)
    return "<br>".join(out)

def pc_index(x):
    m = re.search(r"PC\s*(\d+)", str(x), re.I)
    return int(m.group(1)) if m else 1_000

def cluster_index(x):
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else 1_000

# ─────────────────────────────
# Load datasets
# ─────────────────────────────
enr, _          = read_any_csv("enroll")
ass_course, _   = read_any_csv("ass_course")
ass_summed, _   = read_any_csv("ass_summed")
ass_improve, _  = read_any_csv("ass_improve")
seg_city_csv, _ = read_any_csv("seg_city_csv")
experiment, _   = read_any_csv("experiment")

# Survey questions dictionary (Q1..Qn → text), ignoring “Response Scale” rows
@st.cache_data(show_spinner=False)
def load_question_map():
    p = find_first(FILENAMES["survey_qs"])
    if p is None:
        return {}
    try:
        df = read_csv_forgiving(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        qid_col = cols_lower.get("qid") or cols_lower.get("q_id") or cols_lower.get("question id") or list(df.columns)[0]
        text_cand = [c for c in df.columns if "question" in str(c).strip().lower()]
        text_col = text_cand[0] if text_cand else list(df.columns)[1]
        dd = {}
        for _, r in df[[qid_col, text_col]].dropna().iterrows():
            key_raw = str(r[qid_col]).strip()
            if not re.match(r"^Q\d+\s*$", key_raw, re.I):
                continue  # skip “Response Scale”, 1..7 rows, etc.
            key = key_raw.upper() if key_raw.upper().startswith("Q") else f"Q{key_raw}"
            dd[key] = str(r[text_col]).strip()
        return dd
    except Exception:
        return {}

QTEXT = load_question_map()

# PCA workbook loader (expects sheets: Loadings, ExplainedVariance*, ClusterCenters, CityClusterDistribution optional)
@st.cache_data(show_spinner=False)
def load_pca_workbook():
    xlsx = find_first(FILENAMES["pca_workbook"])
    combo = {"loadings": None, "explained": None, "centers": None, "city_pct": None, "path": xlsx}
    if not xlsx:
        return combo

    # Loadings
    try:
        loadings = pd.read_excel(xlsx, sheet_name="Loadings")
        if "Response" not in loadings.columns:
            loadings = loadings.rename(columns={loadings.columns[0]: "Response"})
        combo["loadings"] = loadings
    except Exception:
        pass

    # Explained variance (accept several sheet name variants)
    def read_explained(sheet_name):
        try:
            ev = pd.read_excel(xlsx, sheet_name=sheet_name)
            pc_col  = next((c for c in ev.columns if "principal" in str(c).lower()), ev.columns[0])
            var_col = next((c for c in ev.columns if "variance"  in str(c).lower()), ev.columns[1])
            ev = ev[[pc_col, var_col]].rename(columns={pc_col: "Principal Component", var_col: "Explained Variance (%)"})
            ev["Explained Variance (%)"] = (
                ev["Explained Variance (%)"]
                .astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
            )
            ev["Explained Variance (%)"] = pd.to_numeric(ev["Explained Variance (%)"], errors="coerce")
            ev = ev.dropna(subset=["Explained Variance (%)"])
            return ev if not ev.empty else None
        except Exception:
            return None

    for cand in ("ExplainedVariance", "Explained Variance", "EV", "Variance"):
        ev = read_explained(cand)
        if ev is not None:
            combo["explained"] = ev
            break

    # Cluster centers (optional)
    try:
        centers = pd.read_excel(xlsx, sheet_name="ClusterCenters")
        if "Cluster" not in centers.columns:
            centers = centers.rename(columns={centers.columns[0]: "Cluster"})
        combo["centers"] = centers
    except Exception:
        pass

    # City / Cluster / Percentage (optional)
    try:
        city_pct = pd.read_excel(xlsx, sheet_name="CityClusterDistribution")
        ren = {city_pct.columns[0]: "City", city_pct.columns[1]: "Cluster", city_pct.columns[2]: "Percentage"}
        city_pct = city_pct.rename(columns=ren)
        city_pct["Percentage"] = (
            city_pct["Percentage"].astype(str).str.replace("%", "", regex=False).str.strip()
        )
        city_pct["Percentage"] = pd.to_numeric(city_pct["Percentage"], errors="coerce") / 100.0
        combo["city_pct"] = city_pct
    except Exception:
        pass

    return combo

pca_combo = load_pca_workbook()

# Fallback for city segments (melt CSV pivot to long)
if pca_combo["city_pct"] is None and seg_city_csv is not None and not seg_city_csv.empty:
    sc = seg_city_csv.copy()
    city_col = "City_y" if "City_y" in sc.columns else sc.columns[0]
    cluster_cols = [c for c in sc.columns if str(c).strip().isdigit()]
    if cluster_cols:
        long_df = sc.melt(id_vars=[city_col], value_vars=cluster_cols, var_name="Cluster", value_name="Employees")
        long_df = long_df.rename(columns={city_col: "City"})
        long_df["Cluster"] = long_df["Cluster"].apply(lambda x: f"Cluster {int(x)}" if str(x).strip().isdigit() else str(x))
        pca_combo["city_pct"] = long_df

# ─────────────────────────────
# KPI row (scale EV if fractional)
# ─────────────────────────────
kpi = {}
if enr is not None and not enr.empty:
    c_country = "Country_Regional_Center" if "Country_Regional_Center" in enr.columns else enr.columns[0]
    c_enroll  = "Total_Enrollments"       if "Total_Enrollments"       in enr.columns else enr.columns[1]
    enr[c_enroll] = ensure_numeric(enr[c_enroll])
    kpi["Total Enrollments"]     = f"{int(enr[c_enroll].sum(skipna=True)):,}"
    kpi["Countries Represented"] = as_text(enr, c_country).nunique()

if ass_course is not None and not ass_course.empty and "Course_Title" in ass_course.columns:
    kpi["Courses Analyzed"] = as_text(ass_course, "Course_Title").nunique()

ev_df = pca_combo.get("explained")
if isinstance(ev_df, pd.DataFrame) and not ev_df.empty:
    vals = ensure_numeric(ev_df["Explained Variance (%)"])
    total_var = float(vals.sum())
    if total_var <= 1.5:
        total_var *= 100.0  # convert 0–1 to %
    kpi["Variance Explained (PC1–PC3)"] = f"{total_var:.1f}%"

if kpi:
    cols = st.columns(min(4, len(kpi)))
    for (label, value), c in zip(kpi.items(), cols):
        c.metric(label, value)

st.markdown("---")

# Sidebar anchor (stable keys reduce jumping)
with st.sidebar:
    st.header("Filters")

# ─────────────────────────────
# Tabs
# ─────────────────────────────
tab1, tab2, tab3 = st.tabs(["📍 Enrollments", "🎯 Training Outcomes", "🧩 PCA & Segmentation"])

# ── TAB 1: Enrollments
with tab1:
    st.subheader("Enrollments by Country")
    if enr is None or enr.empty:
        st.info("Add `country_enrollment_summary.csv`.")
    else:
        c_country = "Country_Regional_Center" if "Country_Regional_Center" in enr.columns else enr.columns[0]
        c_enroll  = "Total_Enrollments"       if "Total_Enrollments"       in enr.columns else enr.columns[1]
        enr[c_enroll] = ensure_numeric(enr[c_enroll])
        view = enr.dropna(subset=[c_enroll]).copy()
        view["_country"] = as_text(view, c_country)

        top10 = view.sort_values(c_enroll, ascending=False).head(10)["_country"].tolist()
        with st.sidebar:
            st.subheader("Enrollments")
            picks = st.multiselect("Countries (default: Top 10)", options=sorted(view["_country"].unique()),
                                   default=top10, key="enr_countries")
            order = st.radio("Sort by", ["Enrollments (desc)", "Country (A–Z)"],
                             horizontal=False, key="enr_sort")

        if picks:
            view = view[view["_country"].isin(picks)]
        view = view.sort_values(c_enroll, ascending=False) if order.startswith("Enrollments") else view.sort_values("_country")

        if view.empty:
            st.info("No countries selected.")
        else:
            fig = px.bar(view, x="_country", y=c_enroll, height=420,
                         labels={"_country": "Country", c_enroll: "Enrollments"},
                         title="Enrollments for Selected Countries")
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_title_standoff=12)
            st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: Training Outcomes
with tab2:
    st.subheader("Training Outcomes by Course and Delivery Mode")
    # Anchor to jump back after selections
    st.markdown('<div id="outcomes_anchor"></div>', unsafe_allow_html=True)

    # Methodology lives only here
    with st.expander("Methodology & Definitions", expanded=False):
        st.markdown(
            "- **Proficiency**: Learners’ self-rated skill level in the training domain.\n"
            "- **Application**: Learners’ confidence in applying those skills in real scenarios.\n"
            "- **Intake**: Baseline measurement before training.\n"
            "- **Outcome**: Measurement after training completes.\n"
            "- **Change**: Improvement from Intake to Outcome (Outcome − Intake)."
        )

    st.caption("Choose a metric and courses (optional) to compare delivery modes and identify top-improving courses.")

    if ass_course is None or ass_course.empty or "Course_Title" not in ass_course.columns:
        st.info("Add `course_assessment_by_course.csv`.")
    else:
        df = ass_course.copy()

        # Delivery mode inferred from title
        df["Delivery Mode"] = df["Course_Title"].apply(
            lambda t: "Virtual" if isinstance(t, str) and "virtual" in t.lower() else "In-Person"
        )

        # Measures (professional labels)
        df["Δ Proficiency"]      = ensure_numeric(df["Outcome_Proficiency_Score"])  - ensure_numeric(df["Intake_Proficiency_Score"])
        df["Δ Application"]      = ensure_numeric(df["Outcome_Applications_Score"]) - ensure_numeric(df["Intake_Applications_Score"])
        df["Proficiency (post)"] = ensure_numeric(df["Outcome_Proficiency_Score"])
        df["Application (post)"] = ensure_numeric(df["Outcome_Applications_Score"])

        metric_options = [
            "Proficiency — Change",
            "Application — Change",
            "Proficiency — Post-training score",
            "Application — Post-training score",
        ]
        col_map = {
            "Proficiency — Change": "Δ Proficiency",
            "Application — Change": "Δ Application",
            "Proficiency — Post-training score": "Proficiency (post)",
            "Application — Post-training score": "Application (post)",
        }

        # Controls in sidebar — jump back to charts on change
        with st.sidebar:
            st.subheader("Outcomes")
            metric_label_ui = st.selectbox(
                "Metric",
                metric_options,
                index=1,
                key="out_metric",
                on_change=set_jump,
                args=("outcomes_anchor",),
            )
            metric_col = col_map[metric_label_ui]

            course_picks = st.multiselect(
                "Courses (optional)",
                options=sorted(df["Course_Title"].dropna().unique()),
                default=[],
                key="out_courses",
                on_change=set_jump,
                args=("outcomes_anchor",),
            )

        # Return to the charts after a selection
        jump_back()

        df_plot = df if not course_picks else df[df["Course_Title"].isin(course_picks)]
        df_plot = df_plot.dropna(subset=[metric_col])

        if df_plot.empty:
            st.info("No data matches the current selection. Try broadening courses.")
        else:
            left, right = st.columns([1.1, 1])

            with left:
                by_mode = df_plot.groupby("Delivery Mode", as_index=False)[metric_col].mean()
                fig = px.bar(
                    by_mode, x="Delivery Mode", y=metric_col, height=400,
                    labels={"Delivery Mode": "Delivery Mode", metric_col: metric_label_ui},
                    title=f"{metric_label_ui} by Delivery Mode",
                )
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_title_standoff=14)
                st.plotly_chart(fig, use_container_width=True)

            with right:
                top = (df_plot.groupby("Course_Title", as_index=False)[metric_col]
                       .mean()
                       .sort_values(metric_col, ascending=False)
                       .head(15))
                top["_Course_Wrapped"] = top["Course_Title"].apply(lambda s: wrap_text(s, 28))
                fig2 = px.bar(
                    top, y="_Course_Wrapped", x=metric_col, orientation="h", height=520,
                    labels={"_Course_Wrapped": "Course", metric_col: metric_label_ui},
                    title=f"{metric_label_ui} — Top 15 Courses",
                )
                fig2.update_traces(text=top[metric_col].round(2), textposition="outside", cliponaxis=False)
                fig2.update_layout(
                    margin=dict(l=140, r=30, t=60, b=10),
                    yaxis={"categoryorder": "total ascending"},
                    xaxis_title_standoff=14,
                )
                st.plotly_chart(fig2, use_container_width=True)

# ── TAB 3: PCA & Segmentation
with tab3:
    st.subheader("PCA Summary & K-Means Segmentation (k = 4)")

    # Segments by City (robust header handling)
    def normalize_city_cluster(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        original_cols = list(df.columns)
        cols_norm = [str(c).strip().lower() for c in original_cols]
        mapping = {}
        for i, c in enumerate(cols_norm):
            if "city" in c:
                mapping[original_cols[i]] = "City"
            elif "cluster" in c:
                mapping[original_cols[i]] = "Cluster"
            elif "percent" in c or c in {"%", "pct", "share"}:
                mapping[original_cols[i]] = "Percentage"
            elif "employee" in c or "count" in c or "num" in c:
                mapping[original_cols[i]] = "Employees"
        df = df.rename(columns=mapping)
        if "Cluster" in df.columns:
            df["Cluster"] = df["Cluster"].apply(lambda x: f"Cluster {int(x)}" if str(x).strip().isdigit() else str(x))
        if "Percentage" in df.columns:
            df["Percentage"] = (
                df["Percentage"].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
            )
            df["Percentage"] = pd.to_numeric(df["Percentage"], errors="coerce")
            if df["Percentage"].max(skipna=True) > 1.5:
                df["Percentage"] = df["Percentage"] / 100.0
        if "Employees" in df.columns and "Percentage" not in df.columns and "City" in df.columns:
            df["Employees"] = pd.to_numeric(df["Employees"], errors="coerce")
            city_tot = df.groupby("City")["Employees"].transform("sum")
            df["Percentage"] = df["Employees"] / city_tot
        return df

    st.markdown("#### Segments by City")
    city_df = None
    if isinstance(pca_combo.get("city_pct"), pd.DataFrame):
        city_df = pca_combo["city_pct"].copy()
    elif seg_city_csv is not None and not seg_city_csv.empty:
        sc = seg_city_csv.copy()
        city_col = "City_y" if "City_y" in sc.columns else sc.columns[0]
        clust_cols = [c for c in sc.columns if str(c).strip().isdigit()]
        long_df = sc.melt(id_vars=[city_col], value_vars=clust_cols, var_name="Cluster", value_name="Employees")
        long_df = long_df.rename(columns={city_col: "City"})
        city_df = long_df

    if city_df is None or city_df.empty:
        st.info("Add a 'CityClusterDistribution' sheet (City, Cluster, Percentage) or the CSV `city_cluster_distribution.csv`.")
    else:
        city_df = normalize_city_cluster(city_df)
        if "Percentage" in city_df.columns:
            fig_loc = px.bar(
                city_df, x="City", y="Percentage", color="Cluster", height=380,
                labels={"Percentage": "Share of Employees", "City": "City", "Cluster": "Segment"},
                title="Segment Share by City",
                category_orders={"Cluster": sorted(city_df["Cluster"].unique(), key=cluster_index)}
            )
            fig_loc.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_title_standoff=12)
            st.plotly_chart(fig_loc, use_container_width=True)
        elif "Employees" in city_df.columns:
            fig_loc = px.bar(
                city_df, x="City", y="Employees", color="Cluster", height=380,
                labels={"Employees": "Employees", "City": "City", "Cluster": "Segment"},
                title="Segment Counts by City",
                category_orders={"Cluster": sorted(city_df["Cluster"].unique(), key=cluster_index)}
            )
            fig_loc.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_title_standoff=12)
            st.plotly_chart(fig_loc, use_container_width=True)
        else:
            st.warning("Could not find a 'Percentage' or 'Employees' column after normalization. Please check sheet headers.")

    # PCA — Explained Variance
    st.markdown("#### PCA — Explained Variance")
    ev = pca_combo.get("explained")
    if isinstance(ev, pd.DataFrame) and not ev.empty:
        ev = ev.copy()
        ev["__order"] = ev["Principal Component"].apply(pc_index)
        ev = ev.sort_values("__order").drop(columns="__order")
        disp = ev.copy()
        if disp["Explained Variance (%)"].sum() <= 1.5:
            disp["Explained Variance (%)"] = disp["Explained Variance (%)"] * 100.0
        fig_ev = px.bar(
            disp, x="Principal Component", y="Explained Variance (%)", height=320,
            labels={"Principal Component": "Principal Component", "Explained Variance (%)": "Explained Variance (%)"},
            title="Explained Variance by Component"
        )
        fig_ev.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_title_standoff=12)
        st.plotly_chart(fig_ev, use_container_width=True)
        st.caption("PC1–PC3 summarize the survey into themes (e.g., Skill Development, Operational Focus, Career Advancement).")
    else:
        st.warning("PCA explained variance not detected — ensure the workbook has a sheet named **ExplainedVariance** (or **Explained Variance**) with two columns: *Principal Component* and *Explained Variance* (values like `31.90%` or `0.319`).")

    # PCA — Top Contributing Survey Questions (with full text)
    st.markdown("#### PCA — Top Contributing Survey Questions")
    loadings = pca_combo.get("loadings")
    ev_for_labels = pca_combo.get("explained")

    if isinstance(loadings, pd.DataFrame) and not loadings.empty:
        if "Response" not in loadings.columns:
            loadings = loadings.rename(columns={loadings.columns[0]: "Response"})
        loadings = loadings.reset_index(drop=True)

        # Component labels from ExplainedVariance (preferred), else PC1..PCk
        if isinstance(ev_for_labels, pd.DataFrame) and not ev_for_labels.empty:
            comp_labels = ev_for_labels.sort_values(
                by="Principal Component", key=lambda s: s.map(pc_index)
            )["Principal Component"].astype(str).tolist()
            if len(comp_labels) != len(loadings):
                comp_labels = [f"PC{i+1}" for i in range(len(loadings))]
        else:
            comp_labels = [f"PC{i+1}" for i in range(len(loadings))]

        with st.sidebar:
            pc_pick = st.selectbox(
                "PCA component",
                comp_labels,
                index=0,
                key="pca_component",
                help="Shows the strongest contributing survey questions for the selected component."
            )

        idx = comp_labels.index(pc_pick)
        row = loadings.iloc[idx]

        question_cols = [c for c in loadings.columns if re.match(r"^Q\d+", str(c), re.I)]
        contrib = sorted(((q, float(row[q])) for q in question_cols),
                         key=lambda x: abs(x[1]), reverse=True)[:8]
        disp = pd.DataFrame({
            "Survey Question": [QTEXT.get(q, q) for q, _ in contrib],
            "Loading (± strength)": [v for _, v in contrib]
        })
        st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── K-Means Cluster Centers (clean + robust)
    centers = pca_combo.get("centers")
    if isinstance(centers, pd.DataFrame) and not centers.empty:
        st.markdown("#### K-Means Cluster Centers in PCA Space")

        # Drop Excel index columns
        centers = centers.loc[:, ~centers.columns.astype(str).str.startswith("Unnamed")].copy()

        # Normalize column names: remove NBSP, collapse spaces, trim
        def norm_col(c):
            s = str(c).replace("\u00A0", " ")
            s = re.sub(r"\s+", " ", s).strip()
            return s
        centers.columns = [norm_col(c) for c in centers.columns]

        # Ensure 'Cluster' exists
        if "Cluster" not in centers.columns:
            centers = centers.rename(columns={centers.columns[0]: "Cluster"})

        # Normalize cluster labels and order
        centers["Cluster"] = centers["Cluster"].apply(
            lambda x: f"Cluster {int(x)}" if str(x).strip().isdigit() else str(x)
        )

        # Detect PC columns broadly: any header containing "PC" + digit
        pc_cols = [c for c in centers.columns if re.search(r"PC\\s*\\d+", c, flags=re.I)]
        if not pc_cols:
            pc_cols = [c for c in centers.columns if "pc" in c.lower()]

        # Keep only Cluster + PC columns; order by PC number if present
        def pc_order(name):
            m = re.search(r"PC\\s*(\\d+)", name, flags=re.I)
            return int(m.group(1)) if m else 999
        pc_cols = sorted(pc_cols, key=pc_order)
        show_cols = ["Cluster"] + [c for c in pc_cols if c in centers.columns]
        centers = centers[show_cols].copy()

        # Final sort Cluster 0..k
        centers = centers.sort_values("Cluster", key=lambda s: s.map(cluster_index))

        st.dataframe(centers, use_container_width=True, hide_index=True)
        # Debug headers if needed:
        # st.write({"columns": centers.columns.tolist()})
