import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Config ---
st.set_page_config(layout="wide", page_title="📊 Education Abroad Applications Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    # Replace with your actual data loading logic
    df = pd.read_csv("data/processed/applications.csv")
    return df

df = load_data()

# --- Preprocess for User-Friendly Fields ---
if "course_name" in df.columns and "course_id" in df.columns:
    course_display_map = dict(zip(df["course_id"], df["course_name"]))
else:
    course_display_map = {}

# --- Dashboard Title & Intro ---
st.title("🌍 Education Abroad Applications Dashboard")
st.markdown("""
This dashboard provides an interactive view of applications for study abroad programs.  
Use the filters next to each chart to refine the view and explore trends by country, course, and time.
""")

# --- KPI Metrics ---
total_apps = len(df)
unique_countries = df["country"].nunique() if "country" in df.columns else 0
acceptance_rate = round((df["status"].eq("Accepted").mean()) * 100, 1) if "status" in df.columns else None

col1, col2, col3 = st.columns(3)
col1.metric("Total Applications Received", total_apps)
col2.metric("Countries Represented", unique_countries)
if acceptance_rate is not None:
    col3.metric("Acceptance Rate", f"{acceptance_rate}%")

st.divider()

# --- Applications by Country ---
st.subheader("Applications by Country")
country_col1, country_col2 = st.columns([1, 3])
with country_col1:
    top_n_countries = st.slider(
        "Number of Top Countries to Show",
        min_value=5,
        max_value=min(25, len(df["country"].unique())),
        value=10
    )
with country_col2:
    country_counts = df["country"].value_counts().nlargest(top_n_countries)
    fig_country = px.bar(
        country_counts,
        x=country_counts.index,
        y=country_counts.values,
        labels={"x": "Country", "y": "Number of Applications"},
        title=f"Top {top_n_countries} Countries by Applications"
    )
    st.plotly_chart(fig_country, use_container_width=True, key="chart_country")

st.divider()

# --- Applications by Course ---
st.subheader("Applications by Course")
course_col1, course_col2 = st.columns([1, 3])
with course_col1:
    if course_display_map:
        selected_course = st.selectbox(
            "Select Course",
            sorted(set(course_display_map.values()))
        )
        filtered_df = df[df["course_name"] == selected_course]
    else:
        selected_course = st.selectbox(
            "Select Course ID",
            sorted(df["course_id"].unique())
        )
        filtered_df = df[df["course_id"] == selected_course]
with course_col2:
    if not filtered_df.empty:
        course_counts = filtered_df["country"].value_counts()
        fig_course = px.bar(
            course_counts,
            x=course_counts.index,
            y=course_counts.values,
            labels={"x": "Country", "y": "Number of Applications"},
            title=f"Applications for {selected_course} by Country"
        )
        st.plotly_chart(fig_course, use_container_width=True, key="chart_course")
    else:
        st.info("No applications found for the selected course.")

st.divider()

# --- Applications Over Time ---
st.subheader("Applications Over Time")
time_col1, time_col2 = st.columns([1, 3])
with time_col1:
    start_date = st.date_input("Start Date", value=None)
    end_date = st.date_input("End Date", value=None)

if "application_date" in df.columns:
    df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")

    time_filtered_df = df.copy()
    if start_date:
        time_filtered_df = time_filtered_df[time_filtered_df["application_date"] >= pd.to_datetime(start_date)]
    if end_date:
        time_filtered_df = time_filtered_df[time_filtered_df["application_date"] <= pd.to_datetime(end_date)]

    apps_over_time = time_filtered_df.groupby("application_date").size().reset_index(name="Applications")
    fig_time = px.line(
        apps_over_time,
        x="application_date",
        y="Applications",
        title="Applications Trend Over Time",
        labels={"application_date": "Date", "Applications": "Number of Applications"}
    )
    st.plotly_chart(fig_time, use_container_width=True, key="chart_time")
else:
    st.warning("No 'application_date' column found in the dataset.")

# --- Insights Section ---
st.divider()
st.subheader("📌 Key Insights")
st.markdown("""
- **Top Country** shows the nations sending the highest number of applications.  
- **Course Filter** allows you to view application distribution for specific courses.  
- **Date Range** lets you focus on a specific time period to identify seasonal trends.
""")
