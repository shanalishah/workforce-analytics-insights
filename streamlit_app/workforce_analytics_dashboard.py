import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import re
from typing import List, Dict, Optional

# ─────────────────────────────
# App Configuration & Models
# ─────────────────────────────
# Using Pydantic for robust data validation would be a great next step.
# For now, we'll focus on clear error handling and structure.

# ─────────────────────────────
# Page Config & Styling
# ─────────────────────────────
st.set_page_config(
    page_title="NLS Workforce Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Workforce Analytics Dashboard")
st.caption("Enrollments • Training Outcomes • Segmentation • Curriculum Experiment")

st.markdown("""
<style>
.block-container { padding-top: 2.0rem !important; }
h1, h2, h3, h4 { line-height: 1.25 !important; margin-top: 0.5rem !important; margin-bottom: 0.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────
# Helper Functions
# ─────────────────────────────
def pc_order_val(label: str) -> int:
    """Sorts strings like 'PC1', 'PC2' correctly."""
    m = re.search(r"PC\s*(\d+)", str(label), re.I)
    return int(m.group(1)) if m else 10_000

def wrap_text(s: str, width: int = 28) -> str:
    """Wraps text for better display on chart axes."""
    words = str(s).split()
    lines, current_line = [], ""
    for w in words:
        if len(current_line) + len(w) + 1 <= width:
            current_line = (current_line + " " + w).strip()
        else:
            lines.append(current_line)
            current_line = w
    if current_line:
        lines.append(current_line)
    return "<br>".join(lines)

def tidy_legend(fig, orientation="h", y=-0.2, x=0.5):
    """Applies a consistent legend style to Plotly figures."""
    fig.update_layout(
        legend=dict(orientation=orientation, yanchor="bottom", y=y, xanchor="center", x=x),
    )

# ─────────────────────────────
# Cached Data Loaders
# ─────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data(file_key: str, path_key: str) -> Optional[pd.DataFrame]:
    """Generic data loader that reads CSV or Excel files."""
    try:
        base_path = Path(st.secrets.paths[path_key])
        filename = st.secrets.files[file_key]
        filepath = base_path / filename

        if not filepath.exists():
            st.error(f"File not found: `{filepath}`. Please check your `secrets.toml` config.")
            return None

        if filepath.suffix.lower() == ".csv":
            return pd.read_csv(filepath, encoding='utf-8')
        else:
            return pd.read_excel(filepath)

    except Exception as e:
        st.error(f"Error loading `{st.secrets.files[file_key]}`: {e}")
        return None

@st.cache_data(show_spinner="Loading PCA workbook...")
def load_pca_workbook() -> Dict:
    """Loads and validates sheets from the PCA results workbook."""
    res = {"loadings": None, "explained": None}
    try:
        filepath = Path(st.secrets.paths.analysis_outputs) / st.secrets.files.pca_workbook
        if not filepath.exists():
            st.error(f"File not found: `{filepath}`. Check `secrets.toml`.")
            return res

        # Explained Variance
        df_ev = pd.read_excel(filepath, sheet_name="ExplainedVariance")
        df_ev = df_ev.rename(columns={df_ev.columns[0]: "Principal Component", df_ev.columns[1]: "Explained Variance (%)"})
        df_ev["Explained Variance (%)"] = pd.to_numeric(df_ev["Explained Variance (%)"].astype(str).str.replace('%', ''), errors='coerce')
        if df_ev["Explained Variance (%)"].max(skipna=True) <= 1.5:
             df_ev["Explained Variance (%)"] *= 100
        res["explained"] = df_ev

        # Loadings
        df_ld = pd.read_excel(filepath, sheet_name="Loadings")
        if "Response" not in df_ld.columns:
             df_ld = df_ld.rename(columns={df_ld.columns[0]: "Response"})
        res["loadings"] = df_ld

    except Exception as e:
        st.error(f"Error processing `{st.secrets.files.pca_workbook}`: {e}. Ensure 'ExplainedVariance' and 'Loadings' sheets exist and are correctly formatted.")
    return res

@st.cache_data(show_spinner="Loading K-Means results...")
def load_kmeans_centers(labels: Dict) -> Optional[pd.DataFrame]:
    """Loads and processes K-Means cluster center data."""
    try:
        filepath = Path(st.secrets.paths.analysis_outputs) / st.secrets.files.kmeans_centers
        if not filepath.exists():
            st.error(f"File not found: `{filepath}`. Check `secrets.toml`.")
            return None

        df = pd.read_excel(filepath, sheet_name="KMeans_Cluster_Centers")
        if "Cluster" not in df.columns:
            df.insert(0, "Cluster", [f"Cluster {i}" for i in range(len(df))])
        else:
            df["Cluster"] = df["Cluster"].apply(lambda x: f"Cluster {x}" if str(x).isdigit() else str(x))

        # Apply persona labels
        df["Persona"] = df["Cluster"].map(labels)
        return df

    except Exception as e:
        st.error(f"Error processing `{st.secrets.files.kmeans_centers}`: {e}. Ensure 'KMeans_Cluster_Centers' sheet exists.")
        return None


# ─────────────────────────────
# Main App
# ─────────────────────────────
# Load all data initially
enroll_df = load_data("enrollments", "analysis_outputs")
assess_df = load_data("assessments_by_course", "analysis_outputs")
city_dist_df = load_data("city_clusters", "analysis_outputs")
experiment_df = load_data("experiment", "processed")
pca_data = load_pca_workbook()
qmap_df = load_data("survey_questions", "raw")

# Make question map for lookups
Q_MAP = {str(r.iloc[0]).upper(): r.iloc[1] for _, r in qmap_df.iterrows()} if qmap_df is not None else {}

# --- Sidebar for Editable Personas ---
st.sidebar.header("Segmentation Settings")
st.sidebar.markdown("Define the personas for each employee segment.")
cluster_labels_config = st.secrets.get("settings", {}).get("cluster_labels", {})
editable_labels = {}
for i in range(4): # Assuming 4 clusters
    key = f"Cluster {i}"
    default_val = cluster_labels_config.get(key, key)
    editable_labels[key] = st.sidebar.text_input(f"Persona for {key}", value=default_val)

# Load K-Means data with editable labels
centers_df = load_kmeans_centers(editable_labels)

# --- KPI Row ---
kpi_cols = st.columns(4)
if enroll_df is not None:
    kpi_cols[0].metric("Total Enrollments", f"{int(enroll_df['Total_Enrollments'].sum()):,}")
    kpi_cols[1].metric("Countries Represented", enroll_df['Country_Regional_Center'].nunique())
if assess_df is not None:
    kpi_cols[2].metric("Courses Analyzed", assess_df['Course_Title'].nunique())
if pca_data.get("explained") is not None:
    total_var = pca_data['explained']['Explained Variance (%)'].sum()
    kpi_cols[3].metric("Variance Explained (Top 3 PCs)", f"{total_var:.1f}%")

st.markdown("---")

# --- Tabs for Main Content ---
tab1, tab2, tab3, tab4 = st.tabs(["📍 Enrollments", "🎯 Training Outcomes", "🧩 PCA & Segmentation", "🔬 Curriculum Experiment"])

with tab1:
    st.header("Enrollments by Country")
    if enroll_df is not None:
        fig = px.bar(
            enroll_df.sort_values("Total_Enrollments", ascending=False),
            x="Country_Regional_Center",
            y="Total_Enrollments",
            title="Total Employee Enrollments by Country",
            labels={"Country_Regional_Center": "Country", "Total_Enrollments": "Total Enrollments"}
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not load enrollment data.")

with tab2:
    st.header("Training Outcomes by Course and Delivery")
    if assess_df is not None:
        df = assess_df.copy()
        df["Delivery Mode"] = df["Course_Title"].apply(lambda t: "Virtual" if "virtual" in str(t).lower() else "In-Person")
        df["Improvement_Proficiency"] = df["Outcome_Proficiency_Score"] - df["Intake_Proficiency_Score"]
        df["Improvement_Application"] = df["Outcome_Applications_Score"] - df["Intake_Applications_Score"]

        metric_options = {
            "Proficiency Improvement": "Improvement_Proficiency",
            "Application Improvement": "Improvement_Application",
        }
        selected_metric = st.selectbox("Select a Performance Metric", options=list(metric_options.keys()))
        metric_col = metric_options[selected_metric]

        c1, c2 = st.columns([1, 1.5])
        with c1:
            by_mode = df.groupby("Delivery Mode")[metric_col].mean().reset_index()
            fig_mode = px.bar(by_mode, x="Delivery Mode", y=metric_col, title=f"Avg. {selected_metric} by Delivery", height=400)
            fig_mode.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_mode, use_container_width=True)

        with c2:
            top_courses = df.groupby("Course_Title")[metric_col].mean().nlargest(15).sort_values(ascending=True).reset_index()
            top_courses["_Course_Wrapped"] = top_courses["Course_Title"].apply(lambda s: wrap_text(s, 35))
            fig_courses = px.bar(top_courses, y="_Course_Wrapped", x=metric_col, orientation="h", title=f"Top 15 Courses by Avg. {selected_metric}", height=400)
            fig_courses.update_layout(margin=dict(l=200, r=10, t=60, b=10), yaxis_title=None)
            st.plotly_chart(fig_courses, use_container_width=True)
    else:
        st.warning("Could not load assessment data.")

with tab3:
    st.header("Employee Segmentation Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Explained Variance by Component")
        if pca_data.get("explained") is not None:
            fig_ev = px.bar(
                pca_data["explained"],
                x="Principal Component",
                y="Explained Variance (%)",
                height=350
            )
            fig_ev.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_ev, use_container_width=True)
        else:
            st.warning("Could not load Explained Variance data.")

    with c2:
        st.subheader("Top Influencing Questions for PC1")
        if pca_data.get("loadings") is not None:
            ld = pca_data["loadings"]
            q_cols = [c for c in ld.columns if c.startswith('Q')]
            pc1_loadings = ld[q_cols].iloc[0].abs().nlargest(5).sort_values()
            pc1_loadings = pc1_loadings.reset_index().rename(columns={'index': 'QuestionID', 0: 'Absolute Loading'})
            pc1_loadings['Question'] = pc1_loadings['QuestionID'].map(Q_MAP).apply(lambda s: wrap_text(s, 40))

            fig_ld = px.bar(pc1_loadings, y='Question', x='Absolute Loading', orientation='h', height=350)
            fig_ld.update_layout(margin=dict(l=250, r=10, t=30, b=10), yaxis_title=None)
            st.plotly_chart(fig_ld, use_container_width=True)
        else:
            st.warning("Could not load PCA Loadings data.")

    st.markdown("---")
    st.subheader("Segment Distribution and Characteristics")

    c3, c4 = st.columns([1.8, 1])
    with c3:
        st.markdown("#### Segment Distribution by City")
        if city_dist_df is not None:
            city_dist_df['Cluster'] = city_dist_df['Cluster'].map(lambda x: f"Cluster {int(x)}" if str(x).isdigit() else str(x))
            city_dist_df['Persona'] = city_dist_df['Cluster'].map(editable_labels)
            fig_city = px.bar(city_dist_df, x="City", y="Percentage", color="Persona", title="Employee Segment Share by City")
            tidy_legend(fig_city)
            st.plotly_chart(fig_city, use_container_width=True)
        else:
            st.warning("Could not load City Distribution data.")
    with c4:
        st.markdown("#### Cluster Centers in PCA Space")
        if centers_df is not None:
            st.dataframe(
                centers_df[['Persona', 'PC1', 'PC2', 'PC3']].round(2),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("Could not load K-Means Centers data.")

with tab4:
    st.header("Analysis of Curriculum Experiment (Course 103)")
    if experiment_df is not None:
        df_exp = experiment_df.copy()
        df_exp["Improvement_Proficiency"] = df_exp["Outcome_Proficiency_Score"] - df_exp["Intake_Proficiency_Score"]
        df_exp["Improvement_Application"] = df_exp["Outcome_Applications_Score"] - df_exp["Intake_Applications_Score"]

        avg_improvement = df_exp.groupby("Training_Program")[["Improvement_Proficiency", "Improvement_Application"]].mean().reset_index()

        st.markdown("This experiment tested two potential new curricula (A and B) for Course 103 against the current program. The charts below show the average improvement in employee scores (Outcome - Intake) for each program.")

        c1, c2 = st.columns(2)
        with c1:
            fig_prof = px.bar(
                avg_improvement.sort_values("Improvement_Proficiency", ascending=False),
                x="Training_Program",
                y="Improvement_Proficiency",
                title="Average Proficiency Score Improvement",
                labels={"Training_Program": "Curriculum", "Improvement_Proficiency": "Avg. Point Improvement"}
            )
            fig_prof.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_prof, use_container_width=True)

        with c2:
            fig_app = px.bar(
                avg_improvement.sort_values("Improvement_Application", ascending=False),
                x="Training_Program",
                y="Improvement_Application",
                title="Average Application Score Improvement",
                labels={"Training_Program": "Curriculum", "Improvement_Application": "Avg. Point Improvement"}
            )
            fig_app.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_app, use_container_width=True)

        st.markdown("#### Key Observations")
        st.info(
            "**Curriculum B** shows the highest average improvement in both Proficiency and Application scores. "
            "**Curriculum A** performs similarly to the current program in Proficiency but slightly better in Application."
        )

    else:
        st.warning("Could not load Curriculum Experiment data.")


# ─────────────────────────────
# Footer
# ─────────────────────────────
st.markdown("---")
st.caption("© Shan Ali Shah — Workforce Analytics Portfolio")
