import streamlit as st

from main import build_sorted_results, fetch_jobs_real_api, job_posts, my_skills


st.set_page_config(page_title="Job Matcher Testing", page_icon=":mag:", layout="wide")


def parse_skill_input(raw_text):
    return [skill.strip() for skill in raw_text.split(",") if skill.strip()]


def load_jobs():
    api_jobs = fetch_jobs_real_api()
    return api_jobs if api_jobs else job_posts


st.title("Job Matcher Testing")
st.caption("QA/SDET practice app for skill matching, API failure handling, and ranked results.")

skill_input = st.text_input(
    "Skills",
    value=", ".join(my_skills),
    help="Enter skills separated by commas.",
)

current_skills = parse_skill_input(skill_input)
jobs = load_jobs()
results = build_sorted_results(jobs, current_skills)

summary_columns = st.columns(3)
summary_columns[0].metric("Skills Entered", len(current_skills))
summary_columns[1].metric("Jobs Evaluated", len(results))
summary_columns[2].metric(
    "Best Match",
    f"{results[0]['match_percent']:.0f}%" if results else "0%",
)

st.subheader("Ranked Matches")

if not results:
    st.warning("No jobs available to evaluate.")
else:
    for result in results:
        with st.container(border=True):
            top_columns = st.columns([3, 1, 1])
            top_columns[0].markdown(f"### {result['title']}")
            top_columns[1].metric("Match", f"{result['match_percent']:.0f}%")
            top_columns[2].metric("Level", result["level"])

            details_columns = st.columns(2)
            details_columns[0].write(f"Required Skills: {', '.join(result['skills']) or 'None'}")
            details_columns[1].write(
                f"Matched Skills: {', '.join(result['matched_skills']) or 'None'}"
            )
