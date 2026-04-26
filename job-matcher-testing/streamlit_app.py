import streamlit as st
import streamlit.components.v1 as components
import json
from pathlib import Path

from main import (
    analyze_job_text,
    extract_skills_from_resume_text,
    get_skill_patterns,
    generate_resume_improvement_comments,
    infer_job_title_from_text,
    load_custom_skill_keywords,
    read_resume_file,
    remove_custom_skill,
    upsert_custom_skill_keywords,
)


st.set_page_config(page_title="Job Matcher Testing", page_icon=":mag:", layout="wide")

JD_HISTORY_PATH = Path(__file__).with_name("jd_history.json")
RESUME_CACHE_PATH = Path(__file__).with_name("resume_cache.json")


LEVEL_COLORS = {
    "PERFECT": {"bg": "#dcfce7", "border": "#22c55e", "text": "#166534"},
    "GOOD": {"bg": "#ecfccb", "border": "#84cc16", "text": "#3f6212"},
    "OK": {"bg": "#ffedd5", "border": "#f97316", "text": "#9a3412"},
    "BAD": {"bg": "#fee2e2", "border": "#ef4444", "text": "#991b1b"},
}


def get_level_colors(level):
    return LEVEL_COLORS.get(level, LEVEL_COLORS["BAD"])


def render_status_card(label, value, level):
    colors = get_level_colors(level)
    st.markdown(
        f"""
        <div style="
            background:{colors['bg']};
            border:1px solid {colors['border']};
            border-radius:16px;
            padding:16px 18px;
            min-height:96px;
        ">
            <div style="
                font-size:0.9rem;
                color:{colors['text']};
                opacity:0.9;
                margin-bottom:8px;
            ">{label}</div>
            <div style="
                font-size:2rem;
                font-weight:700;
                color:{colors['text']};
                line-height:1.1;
            ">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_card(label, value):
    value_text = str(value)
    value_font_size = "1.7rem"
    if len(value_text) > 18:
        value_font_size = "1.35rem"
    if len(value_text) > 28:
        value_font_size = "1.1rem"

    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:1px solid #e5e7eb;
            border-radius:16px;
            padding:16px 18px;
            min-height:96px;
            box-shadow:0 1px 2px rgba(15, 23, 42, 0.04);
        ">
            <div style="
                font-size:0.9rem;
                color:#6b7280;
                margin-bottom:8px;
            ">{label}</div>
            <div style="
                font-size:{value_font_size};
                font-weight:700;
                color:#111827;
                line-height:1.2;
                word-break:break-word;
                overflow-wrap:anywhere;
            ">{value_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis(analysis, resume_skills):
    job = analysis["job"]
    result = analysis["result"]
    feedback = analysis["feedback"]
    resume_comments = analysis["resume_comments"]
    metadata = job.get("metadata", {})

    st.subheader("Flow Status")
    flow_columns = st.columns(5)
    flow_columns[0].success("Resume received")
    flow_columns[1].success("Resume skills extracted")
    flow_columns[2].success("Job skills extracted")
    flow_columns[3].success("Score calculated")
    flow_columns[4].success("Feedback generated")

    title_columns = st.columns([3, 1, 1])
    title_columns[0].markdown(f"## {job['title']}")
    with title_columns[1]:
        render_status_card("Match", f"{result['match_percent']:.0f}%", result["level"])
    with title_columns[2]:
        render_status_card("Level", result["level"], result["level"])

    summary_columns = st.columns(5)
    summary_columns[0].metric("Resume Skills", len(resume_skills))
    summary_columns[1].metric("Job Skills", len(result["skills"]))
    summary_columns[2].metric("Matched Skills", result["match_count"])
    summary_columns[3].metric("Apply Score", f"{result['apply_score']:.0f}")
    summary_columns[4].metric("Recommendation", result["apply_recommendation"])

    st.subheader("Extracted Resume Skills")
    st.write(", ".join(resume_skills) or "No recognized skills found in the resume.")

    st.subheader("Feedback")
    st.markdown('<div id="feedback-section"></div>', unsafe_allow_html=True)
    st.write(feedback["summary"])

    detail_columns = st.columns(2)
    detail_columns[0].write(f"Matched Skills: {feedback['matched_skills_text']}")
    detail_columns[1].write(f"Missing Skills: {feedback['missing_skills_text']}")

    st.subheader("JD Summary")
    summary_columns = st.columns(4)
    with summary_columns[0]:
        render_summary_card("Wanted Skills", len(result["skills"]))
    with summary_columns[1]:
        render_summary_card("Salary", metadata.get("salary", "Not found"))
    with summary_columns[2]:
        render_summary_card("Location", metadata.get("location", "Not found"))
    with summary_columns[3]:
        render_summary_card("Work Model", metadata.get("work_model", "Not found"))

    extra_columns = st.columns(2)
    extra_columns[0].write(f"Employment Type: {metadata.get('employment_type', 'Not found')}")
    extra_columns[1].write(f"Wanted Skills: {', '.join(result['skills']) or 'No recognized skills found.'}")

    fit_columns = st.columns(3)
    fit_columns[0].metric("Core Skill Match", f"{result['core_skill_match_percent']:.0f}%")
    fit_columns[1].metric("Role Fit", f"{result['role_fit_score']:.0f}")
    fit_columns[2].metric("Experience Fit", f"{result['experience_fit_score']:.0f}")

    st.write(f"Required Skills: {', '.join(job.get('required_skills', [])) or 'Not clearly identified.'}")
    st.write(f"Preferred Skills: {', '.join(job.get('preferred_skills', [])) or 'Not clearly identified.'}")
    st.write(f"Core Skills: {', '.join(result['core_skills']) or 'No core skills identified.'}")
    st.write(f"Matched Core Skills: {', '.join(result['matched_core_skills']) or 'None'}")
    st.write(f"Job Role: {result['job_role']} | Candidate Role: {result['candidate_role']}")
    st.write(
        f"Experience Gap: required {result['required_years'] if result['required_years'] is not None else 'not specified'} years, "
        f"candidate estimated {result['candidate_years']} years"
    )

    st.subheader("Resume Improvement Comments")
    for comment in resume_comments:
        st.write(f"- {comment}")

    components.html(
        """
        <script>
          const target = window.parent.document.getElementById('feedback-section');
          if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        </script>
        """,
        height=0,
    )


def load_jd_history():
    if not JD_HISTORY_PATH.exists():
        return []

    try:
        return json.loads(JD_HISTORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_jd_history(history):
    JD_HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_resume_cache():
    if not RESUME_CACHE_PATH.exists():
        return {
            "resume_text": "",
            "resume_skills": [],
            "resume_source_label": "",
        }

    try:
        loaded = json.loads(RESUME_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "resume_text": "",
            "resume_skills": [],
            "resume_source_label": "",
        }

    return {
        "resume_text": loaded.get("resume_text", ""),
        "resume_skills": loaded.get("resume_skills", []),
        "resume_source_label": loaded.get("resume_source_label", ""),
    }


def save_resume_cache(resume_text, resume_skills, resume_source_label):
    RESUME_CACHE_PATH.write_text(
        json.dumps(
            {
                "resume_text": resume_text,
                "resume_skills": resume_skills,
                "resume_source_label": resume_source_label,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def clear_resume_cache():
    if RESUME_CACHE_PATH.exists():
        RESUME_CACHE_PATH.unlink()


if "resume_text_cache" not in st.session_state:
    st.session_state.resume_text_cache = ""

if "resume_skills_cache" not in st.session_state:
    st.session_state.resume_skills_cache = []

if "resume_source_label" not in st.session_state:
    st.session_state.resume_source_label = ""

if "show_resume_replace" not in st.session_state:
    st.session_state.show_resume_replace = False

if "job_description_input" not in st.session_state:
    st.session_state.job_description_input = ""

if "selected_history_index" not in st.session_state:
    st.session_state.selected_history_index = None

saved_resume_cache = load_resume_cache()
if not st.session_state.resume_text_cache and saved_resume_cache["resume_text"]:
    st.session_state.resume_text_cache = saved_resume_cache["resume_text"]
    st.session_state.resume_skills_cache = saved_resume_cache["resume_skills"]
    st.session_state.resume_source_label = saved_resume_cache["resume_source_label"]

st.session_state.jd_history = load_jd_history()


with st.sidebar:
    st.header("Custom Skills / Keywords")
    st.caption("Add missing skill names and keywords. Changes apply immediately to both resume and JD extraction.")

    with st.form("custom-skill-form", clear_on_submit=True):
        custom_skill_name = st.text_input("Skill Name", placeholder="Jira")
        custom_skill_keywords = st.text_input(
            "Keywords",
            placeholder="jira, atlassian jira",
            help="Use comma-separated keywords or phrases.",
        )
        submitted_custom_skill = st.form_submit_button("Add or Update Skill", use_container_width=True)

    if submitted_custom_skill:
        keywords = [keyword.strip() for keyword in custom_skill_keywords.split(",")]
        if upsert_custom_skill_keywords(custom_skill_name, keywords):
            st.success("Custom skill updated.")
            st.rerun()
        else:
            st.error("Enter both a skill name and at least one keyword.")

    current_custom_skills = load_custom_skill_keywords()
    if current_custom_skills:
        for skill_name, keywords in current_custom_skills.items():
            st.markdown(f"**{skill_name}**")
            st.caption(", ".join(keywords))
            if st.button("Delete Skill", key=f"delete-custom-skill-{skill_name}", use_container_width=True):
                remove_custom_skill(skill_name)
                st.rerun()
    else:
        st.info("No custom skills yet.")

    st.markdown("---")
    st.header("Saved JD List")

    if st.session_state.jd_history:
        if st.button("Clear All JD History", use_container_width=True):
            st.session_state.jd_history = []
            save_jd_history([])
            st.rerun()

        for index, item in enumerate(st.session_state.jd_history, start=1):
            st.markdown(f"**{index}. {item['title']}**")
            st.caption(item["description"][:140] + ("..." if len(item["description"]) > 140 else ""))
            if st.button("View Analysis", key=f"sidebar-view-jd-{index}", use_container_width=True):
                st.session_state.selected_history_index = index - 1
                st.rerun()
            if st.button("Delete JD", key=f"sidebar-delete-jd-{index}", use_container_width=True):
                del st.session_state.jd_history[index - 1]
                save_jd_history(st.session_state.jd_history)
                if st.session_state.selected_history_index == index - 1:
                    st.session_state.selected_history_index = None
                st.rerun()
            st.markdown("---")
    else:
        st.info("No saved job descriptions yet.")


st.title("Job Matcher Testing")
st.caption("Upload your resume, extract skills automatically, then compare them with a pasted job description.")
st.caption(f"Currently tracking {len(get_skill_patterns())} total skill patterns.")
st.caption("Resume is saved locally for this app instance, so you do not need to re-upload it after refresh.")

if st.session_state.resume_skills_cache:
    st.subheader("Current Resume")
    st.success(
        f"Using saved resume: {st.session_state.resume_source_label or 'Previously uploaded resume'}"
    )
    control_columns = st.columns(2)
    if control_columns[0].button("Replace Resume", use_container_width=True):
        st.session_state.show_resume_replace = True
    if control_columns[1].button("Clear Resume", use_container_width=True):
        st.session_state.resume_text_cache = ""
        st.session_state.resume_skills_cache = []
        st.session_state.resume_source_label = ""
        st.session_state.show_resume_replace = True
        clear_resume_cache()
        st.rerun()
else:
    st.session_state.show_resume_replace = True

resume_file = None

if st.session_state.show_resume_replace:
    resume_file = st.file_uploader(
        "Resume File",
        type=["pdf", "txt"],
        help="Upload a PDF or TXT resume to extract skills automatically.",
    )

job_description = st.text_area(
    "Job Description",
    placeholder="Paste the full job description here...",
    height=260,
    key="job_description_input",
)

if st.button("Analyze Job", type="primary", use_container_width=True):
    resolved_resume_text = st.session_state.resume_text_cache or ""

    if st.session_state.show_resume_replace and resume_file is not None:
        resolved_resume_text = read_resume_file(resume_file).strip()
        st.session_state.resume_source_label = resume_file.name

    if not resolved_resume_text:
        st.error("Upload a resume file first.")
    elif not job_description.strip():
        st.error("Paste a job description first.")
    else:
        extracted_resume_skills = extract_skills_from_resume_text(resolved_resume_text)
        st.session_state.resume_text_cache = resolved_resume_text
        st.session_state.resume_skills_cache = extracted_resume_skills
        st.session_state.show_resume_replace = False
        save_resume_cache(
            resolved_resume_text,
            extracted_resume_skills,
            st.session_state.resume_source_label,
        )
        current_job_description = job_description.strip()
        current_job_title = infer_job_title_from_text(current_job_description)

        st.session_state.jd_history.insert(
            0,
            {
                "title": current_job_title,
                "description": current_job_description,
                "analysis": None,
            },
        )
        analysis = analyze_job_text(
            current_job_description,
            extracted_resume_skills,
            title=current_job_title,
            resume_text=resolved_resume_text,
        )
        analysis["resume_comments"] = generate_resume_improvement_comments(analysis["result"])
        analysis["resume_skills"] = extracted_resume_skills
        st.session_state.jd_history[0]["analysis"] = analysis
        st.session_state.jd_history = st.session_state.jd_history[:10]
        st.session_state.selected_history_index = 0
        save_jd_history(st.session_state.jd_history)

        render_analysis(analysis, extracted_resume_skills)

elif (
    st.session_state.selected_history_index is not None
    and 0 <= st.session_state.selected_history_index < len(st.session_state.jd_history)
):
    selected_item = st.session_state.jd_history[st.session_state.selected_history_index]
    stored_analysis = selected_item.get("analysis")
    if stored_analysis:
        render_analysis(
            stored_analysis,
            stored_analysis.get("resume_skills", st.session_state.resume_skills_cache),
        )

if st.session_state.resume_skills_cache:
    st.subheader("Saved Resume Skills")
    st.write(", ".join(st.session_state.resume_skills_cache))

st.markdown("---")
if st.button("Analyze Another JD", use_container_width=True):
    st.session_state.job_description_input = ""
    components.html(
        """
        <script>
          window.parent.scrollTo({ top: 0, behavior: 'smooth' });
        </script>
        """,
        height=0,
    )
