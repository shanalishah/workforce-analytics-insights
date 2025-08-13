# workforce_analytics_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --------------------------------------------------------------------------
# Page Configuration and Styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="NLS Analytics Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Custom CSS for a cleaner look */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .stApp {
        background-color: #F0F2F6;
    }
    h1, h2, h3 {
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data Loading (with Caching) - Corrected for Your File Structure
# --------------------------------------------------------------------------
@st.cache_data
def load_all_data():
    """
    Loads all necessary data from the nested 'data' directory structure.
    This function is cached for performance.
    """
    # Get the root path of the project (assuming the script is in a subfolder like 'streamlit_app')
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data"

    if not data_path.exists():
        st.error(f"The 'data' directory was not found at the expected location: {data_path}. Please ensure your file structure is correct.")
        return {}

    # Map keys to their exact file paths relative to the 'data' directory
    # This now matches your GitHub repository screenshots.
    file_paths = {
        "enrollment": "analysis-outputs/country_enrollment_summary.csv",
        "assessments": "processed/assessment_improvement.csv",
        "courses": "raw/nls_courses.csv",
        "employees": "raw/nls_employees.csv",
        "experiment": "processed/experiment_curriculum_cleaned.csv",
        "city_clusters": "analysis-outputs/city_cluster_distribution.csv",
        "pca_loadings": "analysis-outputs/pca_kmeans_results.xlsx - PCA_Loadings.csv",
        "cluster_centers": "analysis-outputs/pca_kmeans_results.xlsx - KMeans_Cluster_Centers.csv",
        "flyers": "reports/genai_flyers.pdf", # Assuming a 'reports' folder for PDFs
        "gpt_docs": "reports/genai_custom_gpt_documentation.pdf",
        "exec_summary": "reports/genai_executive_summary.pdf"
    }

    dataframes = {}
    for key, rel_path in file_paths.items():
        filepath = data_path / rel_path
        if filepath.exists():
            try:
                if str(filepath).endswith('.csv'):
                    dataframes[key] = pd.read_csv(filepath)
                elif str(filepath).endswith('.xlsx'):
                    dataframes[key] = pd.read_excel(filepath)
                else:
                    dataframes[key] = str(filepath) # Store path for PDF files
            except Exception as e:
                st.error(f"Error loading '{filepath.name}': {e}")
                dataframes[key] = None
        else:
            # For now, we will let missing optional files (like PDFs) fail silently.
            # You could add specific warnings here if a file is absolutely required.
            dataframes[key] = None

    # Post-processing to combine dataframes as needed
    if dataframes.get("courses") is not None:
        dataframes["courses"]["Delivery_Mode"] = dataframes["courses"]["Course_Title"].apply(
            lambda x: "Virtual" if "virtual" in str(x).lower() else "In-Person"
        )
    if dataframes.get("assessments") is not None and dataframes.get("courses") is not None:
        dataframes["assessments"] = pd.merge(
            dataframes["assessments"],
            dataframes["courses"][['Course_ID', 'Delivery_Mode', 'Course_Title']],
            on='Course_ID',
            how='left'
        )

    return dataframes

# Load data once
data = load_all_data()


# --------------------------------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------------------------------
st.sidebar.title("Nexus Logistics Solutions")
st.sidebar.markdown("Strategic Analytics Initiative")

PAGES = {
    "🏠 Project Overview": "page_overview",
    "1. Program Utilization": "page_case1",
    "2. Training Effectiveness": "page_case2",
    "3. Employee Segmentation": "page_case3",
    "4. GenAI Showcase": "page_case4",
    "5. Curriculum Experiment": "page_case5",
}

selection = st.sidebar.radio("Go to", list(PAGES.keys()))


# --------------------------------------------------------------------------
# Reusable Plotting Functions
# --------------------------------------------------------------------------
def format_fig(fig):
    """Applies consistent styling to plotly figures."""
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="sans-serif", size=12)
    )
    return fig

# --------------------------------------------------------------------------
# Page Functions (No changes needed in these functions)
# --------------------------------------------------------------------------

def page_overview():
    st.title("🏠 Project Overview")
    st.markdown("This dashboard presents the consolidated findings from a five-part Strategic Analytics Initiative for Nexus Logistics Solutions (NLS). The goal was to leverage data to enhance the Employee Development Program (EDP).")
    
    st.subheader("Project Journey")
    st.info("""
    The project followed a logical progression from broad exploration to specific, actionable recommendations:
    - **Case 1: Program Utilization:** Understood who is taking what training and where.
    - **Case 2: Training Effectiveness:** Analyzed how much employees improved after training.
    - **Case 3: Employee Segmentation:** Identified key employee motivations to tailor engagement.
    - **Case 4: GenAI Targeted Promotion:** Used AI to create personalized promotional materials.
    - **Case 5: Curriculum Experiment:** Conducted an A/B test to find the most effective new curriculum.

    Use the sidebar to navigate through the findings for each case study.
    """)
    
    st.subheader("Key Performance Indicators Across the Project")
    kpi_cols = st.columns(4)
    if data.get("enrollment") is not None:
        kpi_cols[0].metric("Total Enrollments Analyzed", f"{data['enrollment']['Total_Enrollments'].sum():,}")
    if data.get("employees") is not None:
        kpi_cols[1].metric("Employees in Dataset", f"{data['employees']['Employee_ID'].nunique():,}")
    if data.get("courses") is not None:
        kpi_cols[2].metric("Unique Courses Offered", data['courses']['Course_ID'].nunique())
    if data.get("cluster_centers") is not None:
        kpi_cols[3].metric("Employee Segments Identified", data['cluster_centers']['Cluster'].nunique())


def page_case1():
    st.title("1. Program Utilization Analysis")
    st.markdown("This analysis explored historical enrollment data to understand participation trends across courses and locations.")

    enroll_df = data.get("enrollment")
    courses_df = data.get("courses")

    if enroll_df is None or courses_df is None:
        st.warning("Enrollment or Course data is not available.")
        return

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Enrollments by Country")
        fig = px.bar(
            enroll_df.sort_values("Total_Enrollments", ascending=False),
            x="Country_Regional_Center", y="Total_Enrollments",
            title="Total Enrollments by Country",
            labels={"Country_Regional_Center": "Country", "Total_Enrollments": "Number of Enrollments"}
        )
        st.plotly_chart(format_fig(fig), use_container_width=True)

    with c2:
        st.subheader("Enrollments by Delivery Mode")
        mode_counts = courses_df['Delivery_Mode'].value_counts().reset_index()
        fig = px.pie(
            mode_counts, names='Delivery_Mode', values='count',
            title="Share of Courses by Delivery Mode",
            hole=0.4
        )
        fig.update_traces(textinfo='percent+label', marker=dict(colors=['#1f77b4', '#ff7f0e']))
        st.plotly_chart(format_fig(fig), use_container_width=True)
    
    st.subheader("Key Insight")
    st.info("The analysis of program utilization revealed significant variations in course popularity and participation across different geographic locations, highlighting the need for tailored regional training strategies.")

def page_case2():
    st.title("2. Training Effectiveness Analysis")
    st.markdown("This analysis measured the impact of training on employee skills by comparing pre-training (Intake) and post-training (Outcome) assessment scores.")

    assess_df = data.get("assessments")
    if assess_df is None:
        st.warning("Assessment data is not available.")
        return

    st.subheader("Average Skill Improvement by Course")
    
    metric_choice = st.selectbox(
        "Select a metric to analyze:",
        ("Proficiency Improvement", "Application Improvement")
    )
    
    metric_col = "Improvement_Proficiency" if metric_choice == "Proficiency Improvement" else "Improvement_Application"

    top_courses = assess_df.groupby('Course_Title')[metric_col].mean().nlargest(15).sort_values().reset_index()

    fig = px.bar(
        top_courses,
        x=metric_col, y="Course_Title",
        orientation='h',
        title=f"Top 15 Courses by {metric_choice}",
        labels={metric_col: "Average Score Improvement (Points)", "Course_Title": "Course"}
    )
    fig.update_layout(height=500, yaxis=dict(tickfont=dict(size=10)))
    st.plotly_chart(format_fig(fig), use_container_width=True)
    
    st.subheader("In-Person vs. Virtual Training Effectiveness")
    mode_comparison = assess_df.groupby('Delivery_Mode')[[ "Improvement_Proficiency", "Improvement_Application"]].mean().reset_index()
    
    fig_comp = go.Figure(data=[
        go.Bar(name='Proficiency', x=mode_comparison['Delivery_Mode'], y=mode_comparison['Improvement_Proficiency']),
        go.Bar(name='Application', x=mode_comparison['Delivery_Mode'], y=mode_comparison['Improvement_Application'])
    ])
    fig_comp.update_layout(
        barmode='group',
        title="Average Improvement: In-Person vs. Virtual",
        xaxis_title="Delivery Mode",
        yaxis_title="Average Score Improvement"
    )
    st.plotly_chart(format_fig(fig_comp), use_container_width=True)
    
    st.subheader("Key Insight")
    st.info("The analysis showed that while both formats are effective, there are measurable differences in performance, with certain courses benefiting more from a specific delivery mode. This data was crucial for decisions like adapting Course 103 to a virtual format.")


def page_case3():
    st.title("3. Employee Segmentation")
    st.markdown("To better understand what motivates employees, we analyzed survey responses from 600 team members to identify distinct employee segments based on their attitudes towards the EDP.")

    city_clusters = data.get("city_clusters")
    cluster_centers = data.get("cluster_centers")

    if city_clusters is None or cluster_centers is None:
        st.warning("Segmentation data is not available.")
        return

    personas = {
        0: "Career-Oriented Implementers",
        1: "Operational Specialists",
        2: "Skill Growth Seekers",
        3: "Foundation Builders",
    }
    city_clusters['Persona'] = city_clusters['Cluster'].map(personas)
    cluster_centers['Persona'] = cluster_centers['Cluster'].map(personas)

    st.subheader("Employee Segments Identified")
    st.dataframe(cluster_centers[['Persona', 'PC1', 'PC2', 'PC3', 'Percentage']].rename(columns={'Percentage': 'Share of Workforce (%)'}), use_container_width=True)

    st.subheader("Segment Distribution by City")
    fig = px.bar(
        city_clusters,
        x="City", y="Percentage", color="Persona",
        title="Share of Employee Segments in Major Cities",
        labels={"Percentage": "Share of Employees (%)"}
    )
    st.plotly_chart(format_fig(fig), use_container_width=True)
    
    st.subheader("Key Insight")
    st.info("Four distinct employee segments with unique motivations were identified. The distribution of these segments varies by location, confirming that a one-size-fits-all engagement strategy for the EDP would be ineffective. This analysis formed the basis for the targeted GenAI campaign.")


def page_case4():
    st.title("4. GenAI Showcase: Targeted EDP Flyers")
    st.markdown("Building on the segmentation analysis, we used a custom Generative AI model to create personalized promotional flyers tailored to each employee segment's motivations.")
    
    st.subheader("Generated Flyers")
    st.info("Below are the draft flyers created by the custom GPT. Each one is designed to resonate with a specific employee persona.")
    
    flyer_personas = {
        "Career-Oriented Implementers": "Focus on promotions, leadership, and long-term growth.",
        "Operational Specialists": "Highlight efficiency gains, system mastery, and practical skills.",
        "Skill Growth Seekers": "Emphasize learning new technologies, innovation, and staying current.",
        "Foundation Builders": "Stress community, collaboration, and building core competencies.",
    }
    
    cols = st.columns(len(flyer_personas))
    for i, (persona, text) in enumerate(flyer_personas.items()):
        with cols[i]:
            st.markdown(f"**Flyer for: {persona}**")
            st.markdown(f"<div style='border:2px solid #4A4A4A; border-radius: 5px; padding: 10px; height: 150px; background-color: #FFFFFF;'>{text}</div>", unsafe_allow_html=True)
            
    with st.expander("View Project Documentation"):
        st.subheader("Project Goal")
        st.write("The objective was to increase EDP engagement by producing targeted flyers that emphasize benefits relevant to each segment's interests.")
        st.subheader("Custom GPT Approach")
        st.write("We leveraged OpenAI's Custom GPT capabilities to generate personalized content. The process involved creating a custom model for each segment, providing it with specific instructions and knowledge about the EDP and the segment's motivations.")
        st.subheader("Evaluation")
        st.write("The project was successful, demonstrating that a custom GPT can efficiently produce high-quality, compelling content tailored to different audience groups. This approach is recommended for future internal marketing campaigns.")


def page_case5():
    st.title("5. Curriculum Experiment (Course 103)")
    st.markdown("An experiment was conducted to evaluate two new training curricula (A and B) for the Advanced Warehouse Management Systems course against the current program.")

    exp_df = data.get("experiment")
    if exp_df is None:
        st.warning("Experiment data is not available.")
        return

    st.subheader("Impact on Employee Performance")
    st.markdown("Performance was measured by the improvement between intake and outcome assessment scores.")

    exp_df['Proficiency_Improvement'] = exp_df['Outcome_Proficiency_Score'] - exp_df['Intake_Proficiency_Score']
    exp_df['Application_Improvement'] = exp_df['Outcome_Applications_Score'] - exp_df['Intake_Applications_Score']

    results = exp_df.groupby('Training_Program')[['Proficiency_Improvement', 'Application_Improvement']].mean().reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            results.sort_values("Proficiency_Improvement", ascending=False),
            x="Training_Program", y="Proficiency_Improvement",
            title="Proficiency Score Improvement",
            labels={"Training_Program": "Curriculum", "Proficiency_Improvement": "Average Point Improvement"}
        )
        st.plotly_chart(format_fig(fig), use_container_width=True)
    with c2:
        fig = px.bar(
            results.sort_values("Application_Improvement", ascending=False),
            x="Training_Program", y="Application_Improvement",
            title="Application Score Improvement",
            labels={"Training_Program": "Curriculum", "Application_Improvement": "Average Point Improvement"}
        )
        st.plotly_chart(format_fig(fig), use_container_width=True)

    st.subheader("Recommendation")
    st.success("""
    **Recommendation: Adopt Curriculum B.**
    
    The analysis shows that Curriculum B delivered the highest average improvement in both Proficiency and Application scores. Therefore, it is recommended that NLS adopt Curriculum B as the new standard for the Advanced Warehouse Management Systems course to maximize employee skill development and practical application ability.
    """)


# --------------------------------------------------------------------------
# Main App Router
# --------------------------------------------------------------------------
if selection in PAGES:
    page_function = locals()[PAGES[selection]]
    page_function()
