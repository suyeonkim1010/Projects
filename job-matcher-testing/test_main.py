from unittest.mock import Mock, patch

import main
import pytest

import requests

from main import (
    analyze_job_text,
    analyze_job_url,
    build_sorted_results,
    build_job_from_user,
    build_scraped_job,
    calculate_match,
    calculate_experience_fit,
    calculate_role_fit,
    extract_bucketed_skills,
    extract_core_skills,
    extract_job_description,
    extract_job_metadata,
    extract_skills_from_resume_text,
    extract_skills_from_text,
    fetch_jobs_real_api,
    generate_feedback,
    generate_resume_improvement_comments,
    get_skill_patterns,
    get_match_level,
    is_valid_user_payload,
    load_custom_skill_keywords,
    main as run_main,
    remove_custom_skill,
    save_custom_skill_keywords,
    should_apply,
    upsert_custom_skill_keywords,
    validate_api_response,
)


def test_normal_job_match_returns_perfect():
    job = {
        "title": "QA Automation Engineer",
        "skills": ["Python", "Selenium", "Pytest"],
    }

    result = calculate_match(job, ["python", "selenium", "pytest", "sql"])

    assert result["title"] == "QA Automation Engineer"
    assert result["match_count"] == 3
    assert result["match_percent"] == 100
    assert result["level"] == "PERFECT"


def test_broken_job_without_skills_returns_bad():
    broken_job = {
        "title": "Broken Job No Skills",
    }

    result = calculate_match(broken_job, ["python"])

    assert result["skills"] == []
    assert result["matched_skills"] == []
    assert result["match_count"] == 0
    assert result["match_percent"] == 0
    assert result["level"] == "BAD"


def test_weird_data_is_cleaned_before_matching():
    weird_job = {
        "title": "Broken Job Weird Skills",
        "skills": ["Python", "", 123, None, "  "],
    }

    result = calculate_match(weird_job, ["python"])

    assert result["skills"] == ["Python"]
    assert result["matched_skills"] == ["Python"]
    assert result["match_count"] == 1
    assert result["match_percent"] == 100
    assert result["level"] == "PERFECT"


def test_real_api_success_returns_job_list():
    mock_response = Mock()
    mock_response.json.return_value = [
        {
            "company": {"name": "Acme"},
        },
        {
            "company": {"name": "Beta"},
        },
    ]
    mock_response.raise_for_status.return_value = None

    with patch("main.requests.get", return_value=mock_response):
        result = fetch_jobs_real_api()

    assert len(result) == 2
    assert result[0]["title"] == "Acme QA Role"
    assert result[0]["skills"] == ["Python", "Selenium", "Pytest"]
    assert result[1]["title"] == "Beta QA Role"


def test_validate_api_response_returns_true_for_valid_users():
    data = [
        {"company": {"name": "Acme"}},
        {"company": {"name": "Beta"}},
    ]

    assert validate_api_response(data) is True


def test_validate_api_response_returns_false_for_invalid_users():
    data = [
        {"company": {"name": "Acme"}},
        {"company": {}},
    ]

    assert validate_api_response(data) is False


def test_validate_api_response_returns_false_for_non_list_input():
    assert validate_api_response({"company": {"name": "Acme"}}) is False


def test_is_valid_user_payload_returns_false_for_non_dict():
    assert is_valid_user_payload("not-a-dict") is False


def test_is_valid_user_payload_returns_false_when_company_is_not_dict():
    assert is_valid_user_payload({"company": "Acme"}) is False


def test_build_job_from_user_cycles_skill_templates():
    result = build_job_from_user({"company": {"name": "Gamma"}}, 2)

    assert result["title"] == "Gamma QA Role"
    assert result["skills"] == ["API Testing", "Pytest"]


def test_real_api_timeout_returns_empty_job_list():
    with patch("main.requests.get", side_effect=requests.exceptions.Timeout):
        result = fetch_jobs_real_api()

    assert result == []


def test_real_api_request_error_returns_empty_job_list():
    with patch("main.requests.get", side_effect=requests.exceptions.RequestException):
        result = fetch_jobs_real_api()

    assert result == []


def test_real_api_bad_response_returns_empty_job_list():
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"wrong_key": "wrong_value"}

    with patch("main.requests.get", return_value=mock_response):
        result = fetch_jobs_real_api()

    assert result == []


def test_real_api_invalid_json_returns_empty_job_list():
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.side_effect = ValueError

    with patch("main.requests.get", return_value=mock_response):
        result = fetch_jobs_real_api()

    assert result == []


def test_real_api_invalid_user_schema_returns_empty_job_list():
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = [
        {"company": {"name": "Acme"}},
        {"company": {}},
    ]

    with patch("main.requests.get", return_value=mock_response):
        result = fetch_jobs_real_api()

    assert result == []


def test_real_api_retries_and_then_succeeds():
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = [
        {"company": {"name": "Retry Success Company"}},
    ]

    with patch(
        "main.requests.get",
        side_effect=[requests.exceptions.Timeout, mock_response],
    ) as mock_get:
        result = fetch_jobs_real_api(retries=1)

    assert mock_get.call_count == 2
    assert result[0]["title"] == "Retry Success Company QA Role"


def test_real_api_returns_empty_list_after_retry_limit():
    with patch(
        "main.requests.get",
        side_effect=requests.exceptions.RequestException,
    ) as mock_get:
        result = fetch_jobs_real_api(retries=2)

    assert mock_get.call_count == 3
    assert result == []


def test_get_match_level_returns_bad_when_rules_are_empty(monkeypatch):
    monkeypatch.setitem(main.matching_rules, "levels", [])

    assert get_match_level(25) == "BAD"


def test_results_are_sorted_from_highest_match():
    jobs = [
        {"title": "Job A", "skills": ["Python", "SQL"]},
        {"title": "Job B", "skills": ["Python", "SQL", "Selenium"]},
        {"title": "Job C", "skills": ["Docker"]},
    ]

    results = build_sorted_results(jobs, ["python", "sql"])

    assert results[0]["title"] == "Job A"
    assert results[0]["match_percent"] == 100
    assert results[-1]["title"] == "Job C"
    assert results[-1]["level"] == "BAD"


@pytest.mark.xfail(reason="Intentional example of a failing test")
def test_intentional_failure_example():
    job = {
        "title": "Intentional Failure Job",
        "skills": ["Python"],
    }

    result = calculate_match(job, ["python"])

    assert result["level"] == "BAD"


def test_empty_job_dict_returns_safe_defaults():
    result = calculate_match({}, ["python"])

    assert result["title"] == "Untitled Job"
    assert result["skills"] == []
    assert result["matched_skills"] == []
    assert result["match_count"] == 0
    assert result["match_percent"] == 0
    assert result["level"] == "BAD"


def test_string_skills_value_is_treated_as_invalid():
    job = {"skills": "Python"}

    result = calculate_match(job, ["python"])

    assert result["title"] == "Untitled Job"
    assert result["skills"] == []
    assert result["matched_skills"] == []
    assert result["match_percent"] == 0
    assert result["level"] == "BAD"


def test_none_my_skills_is_treated_as_empty_list():
    job = {
        "title": "QA Automation Engineer",
        "skills": ["Python", "Selenium"],
    }

    result = calculate_match(job, None)

    assert result["matched_skills"] == []
    assert result["match_count"] == 0
    assert result["match_percent"] == 0
    assert result["level"] == "BAD"


def test_49_percent_is_ok():
    job = {
        "title": "Boundary 49 Percent Job",
        "skills": [f"Skill{i}" for i in range(1, 101)],
    }

    result = calculate_match(job, [f"skill{i}" for i in range(1, 50)])

    assert result["match_percent"] == 49
    assert result["level"] == "OK"


def test_50_percent_is_good():
    job = {
        "title": "Boundary 50 Percent Job",
        "skills": ["Python", "Skill2"],
    }

    result = calculate_match(job, ["python"])

    assert result["match_percent"] == 50
    assert result["level"] == "GOOD"


def test_30_percent_is_ok():
    job = {
        "title": "Boundary 30 Percent Job",
        "skills": [f"Skill{i}" for i in range(1, 11)],
    }

    result = calculate_match(job, [f"skill{i}" for i in range(1, 4)])

    assert result["match_percent"] == 30
    assert result["level"] == "OK"


def test_0_percent_is_bad():
    job = {
        "title": "Boundary 0 Percent Job",
        "skills": ["Python", "Selenium"],
    }

    result = calculate_match(job, ["docker"])

    assert result["match_percent"] == 0
    assert result["level"] == "BAD"


def test_10_percent_is_bad():
    job = {
        "title": "Boundary 10 Percent Job",
        "skills": [f"Skill{i}" for i in range(1, 11)],
    }

    result = calculate_match(job, ["skill1"])

    assert result["match_percent"] == 10
    assert result["level"] == "BAD"


def test_29_percent_is_bad():
    job = {
        "title": "Boundary 29 Percent Job",
        "skills": [f"Skill{i}" for i in range(1, 101)],
    }

    result = calculate_match(job, [f"skill{i}" for i in range(1, 30)])

    assert result["match_percent"] == pytest.approx(29)
    assert result["level"] == "BAD"


def test_duplicate_job_skills_are_counted_once():
    job = {
        "title": "Duplicate Job Skills",
        "skills": ["Python", "Python", "Selenium"],
    }

    result = calculate_match(job, ["python"])

    assert result["skills"] == ["Python", "Selenium"]
    assert result["matched_skills"] == ["Python"]
    assert result["match_count"] == 1
    assert result["match_percent"] == 50
    assert result["level"] == "GOOD"


def test_duplicate_my_skills_are_counted_once():
    job = {
        "title": "Duplicate My Skills",
        "skills": ["Python", "Selenium"],
    }

    result = calculate_match(job, ["python", "python", "python"])

    assert result["matched_skills"] == ["Python"]
    assert result["match_count"] == 1
    assert result["match_percent"] == 50
    assert result["level"] == "GOOD"


def test_end_to_end_api_to_sorted_results_flow():
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = [
        {"company": {"name": "Acme"}},
        {"company": {"name": "Beta"}},
        {"company": {"name": "Gamma"}},
    ]

    with patch("main.requests.get", return_value=mock_response):
        jobs = fetch_jobs_real_api()

    results = build_sorted_results(jobs, ["python", "selenium", "pytest"])

    assert len(jobs) == 3
    assert len(results) == 3
    assert results[0]["title"] == "Acme QA Role"
    assert results[0]["match_percent"] == 100
    assert results[0]["level"] == "PERFECT"
    assert results[-1]["title"] == "Gamma QA Role"
    assert results[-1]["match_percent"] == 50
    assert results[-1]["level"] == "GOOD"


def test_main_falls_back_to_local_job_posts_when_api_returns_empty(capsys):
    with patch("main.fetch_jobs_real_api", return_value=[]):
        run_main()

    captured = capsys.readouterr()

    assert "Job Match Results" in captured.out
    assert "QA Automation Engineer" in captured.out
    assert "Best Job Match" in captured.out


def test_extract_skills_from_text_returns_expected_skills():
    text = """
    We are hiring a QA Automation Engineer with Python, Selenium, Pytest, API testing,
    GitHub Actions, and CI/CD experience.
    """

    result = extract_skills_from_text(text)

    assert result == [
        "Python",
        "Selenium",
        "Pytest",
        "API Testing",
        "GitHub Actions",
        "CI/CD",
    ]


def test_extract_skills_from_resume_text_returns_expected_skills():
    text = """
    Frontend Developer
    React, JavaScript, HTML, CSS, Playwright, GitHub Actions
    Built API integration flows with Python and SQL support.
    """

    result = extract_skills_from_resume_text(text)

    assert result == [
        "Python",
        "SQL",
        "Playwright",
        "JavaScript",
        "React",
        "HTML",
        "CSS",
        "GitHub Actions",
    ]


def test_extract_core_skills_prioritizes_required_sentences():
    text = """
    Nice to have: Git and HTML.
    Required: Python, Selenium, and Pytest.
    Minimum qualifications include API testing.
    """
    all_skills = ["Python", "Selenium", "Pytest", "API Testing", "Git", "HTML"]

    result = extract_core_skills(text, all_skills)

    assert result == ["Python", "Selenium", "Pytest", "API Testing"]


def test_extract_bucketed_skills_splits_required_and_preferred():
    text = """
    Required: Python, Selenium, and Pytest.
    Preferred: Playwright and GitHub Actions.
    Nice to have: HTML.
    """
    all_skills = ["Python", "Selenium", "Pytest", "Playwright", "GitHub Actions", "HTML"]

    required_result = extract_bucketed_skills(text, all_skills, main.REQUIRED_SENTENCE_HINTS)
    preferred_result = extract_bucketed_skills(text, all_skills, main.PREFERRED_SENTENCE_HINTS)

    assert required_result == ["Python", "Selenium", "Pytest"]
    assert preferred_result == ["Playwright", "GitHub Actions", "HTML"]


def test_calculate_role_fit_detects_qa_alignment():
    result = calculate_role_fit(
        "QA Automation Engineer with Selenium and Pytest",
        "Junior SDET with Python, Selenium, and test automation projects",
    )

    assert result["job_role"] == "QA/SDET"
    assert result["candidate_role"] == "QA/SDET"
    assert result["role_fit_score"] == 100


def test_calculate_experience_fit_detects_gap():
    result = calculate_experience_fit(
        "We require 5+ years of QA automation experience.",
        "Entry-level SDET candidate with internship experience.",
    )

    assert result["required_years"] == 5
    assert result["candidate_years"] == 1
    assert result["experience_fit_score"] == 10


def test_should_apply_returns_expected_recommendation():
    assert should_apply(85) == "STRONG APPLY"
    assert should_apply(70) == "APPLY"
    assert should_apply(59.9) == "SKIP"


def test_custom_skill_keywords_are_loaded_and_used(tmp_path):
    custom_path = tmp_path / "custom_skill_patterns.json"

    with patch.object(main, "CUSTOM_SKILL_PATTERNS_PATH", custom_path):
        save_custom_skill_keywords({"Jira": ["jira", "atlassian jira"]})

        loaded = load_custom_skill_keywords()
        patterns = get_skill_patterns()
        result = extract_skills_from_text("Experience with Jira and Python.")

    assert loaded == {"Jira": ["jira", "atlassian jira"]}
    assert "Jira" in patterns
    assert result == ["Python", "Jira"]


def test_custom_skill_can_be_added_and_removed(tmp_path):
    custom_path = tmp_path / "custom_skill_patterns.json"

    with patch.object(main, "CUSTOM_SKILL_PATTERNS_PATH", custom_path):
        created = upsert_custom_skill_keywords("GraphQL", ["graphql", "graph ql"])
        result_before_delete = extract_skills_from_text("Built GraphQL API tests.")
        removed = remove_custom_skill("GraphQL")
        result_after_delete = extract_skills_from_text("Built GraphQL API tests.")

    assert created is True
    assert removed is True
    assert result_before_delete == ["GraphQL"]
    assert result_after_delete == []


def test_extract_job_description_prefers_description_like_sections():
    html = """
    <html>
      <body>
        <div id="job-description">
          Build automated tests. Validate API responses. Improve CI reliability.
        </div>
        <div>Short footer text.</div>
      </body>
    </html>
    """

    result = extract_job_description(html)

    assert "Build automated tests." in result
    assert "Improve CI reliability." in result


def test_build_scraped_job_extracts_title_skills_and_apply_url():
    html = """
    <html>
      <head><title>QA Automation Engineer</title></head>
      <body>
        <h1>QA Automation Engineer</h1>
        <section class="job-description">
          Python Selenium Pytest API testing
        </section>
        <a href="/apply-now">Apply Now</a>
      </body>
    </html>
    """

    result = build_scraped_job("https://example.com/jobs/123", html)

    assert result["title"] == "QA Automation Engineer"
    assert result["skills"] == ["Python", "Selenium", "Pytest", "API Testing"]
    assert result["apply_url"] == "https://example.com/apply-now"
    assert result["metadata"]["salary"] == "Not found"


def test_extract_job_metadata_returns_salary_location_and_work_model():
    text = """
    QA Automation Engineer
    Location: Edmonton, AB
    Hybrid
    Full-time
    Salary: $80,000 - $95,000 a year
    Looking for Python, Selenium, and Pytest experience.
    """

    result = extract_job_metadata(text)

    assert result["salary"] == "$80,000 - $95,000 a year"
    assert result["location"] == "Edmonton, AB"
    assert result["work_model"] == "Hybrid"
    assert result["employment_type"] == "Full-time"


def test_extract_job_metadata_treats_in_person_as_on_site():
    text = """
    QA Analyst
    Location: Toronto, ON
    In-person
    Contract
    """

    result = extract_job_metadata(text)

    assert result["location"] == "Toronto, ON"
    assert result["work_model"] == "On-site"
    assert result["employment_type"] == "Contract"


def test_extract_job_metadata_strips_work_model_from_location_value():
    text = """
    Software Tester
    Location: Halton District, ON in person
    Full-time
    """

    result = extract_job_metadata(text)

    assert result["location"] == "Halton District, ON"
    assert result["work_model"] == "On-site"


def test_extract_job_metadata_prefers_fifth_non_empty_line_for_location():
    text = """
    QA Analyst
    ABC Company
    Full-time
    Hybrid
    Halton District, ON
    Salary: $75,000 a year
    """

    result = extract_job_metadata(text)

    assert result["location"] == "Halton District, ON"
    assert result["work_model"] == "Hybrid"


def test_generate_feedback_lists_missing_skills():
    match_result = {
        "title": "QA Automation Engineer",
        "skills": ["Python", "Selenium", "Pytest"],
        "matched_skills": ["Python"],
        "match_count": 1,
        "match_percent": 33.3333,
        "level": "OK",
    }

    feedback = generate_feedback(match_result)

    assert "Partial match" in feedback["summary"]
    assert feedback["matched_skills_text"] == "Python"
    assert feedback["missing_skills_text"] == "Selenium, Pytest"


def test_generate_resume_improvement_comments_references_missing_skills():
    match_result = {
        "title": "QA Automation Engineer",
        "skills": ["Python", "Selenium", "Pytest"],
        "matched_skills": ["Python"],
        "match_count": 1,
        "match_percent": 33.3333,
        "level": "OK",
    }

    comments = generate_resume_improvement_comments(match_result)

    assert any("Selenium, Pytest" in comment for comment in comments)
    assert any("ATS" in comment for comment in comments)


def test_analyze_job_url_runs_end_to_end_with_mocked_html():
    html = """
    <html>
      <head><title>Junior SDET</title></head>
      <body>
        <div class="job-description">
          Looking for Python, Pytest, and API testing experience.
        </div>
        <a href="/apply">Apply</a>
      </body>
    </html>
    """

    with patch("main.fetch_page_html", return_value=html):
        result = analyze_job_url("https://example.com/job", ["python", "pytest"])

    assert result["job"]["title"] == "Junior SDET"
    assert result["result"]["matched_skills"] == ["Python", "Pytest"]
    assert result["result"]["level"] == "GOOD"
    assert result["job"]["apply_url"] == "https://example.com/apply"


def test_analyze_job_text_runs_end_to_end_without_url_fetch():
    text = """
    QA Automation Engineer
    Location: Calgary, AB
    Remote
    Contract
    $45 per hour
    Required: Python, Selenium, and Pytest experience in a QA Automation Engineer role.
    Preferred: Playwright and GitHub Actions.
    """

    resume_text = """
    Junior SDET
    Entry-level QA candidate with Python, Pytest, and internship experience.
    """

    result = analyze_job_text(
        text,
        ["python", "pytest"],
        title="QA Automation Engineer",
        resume_text=resume_text,
    )

    assert result["job"]["title"] == "QA Automation Engineer"
    assert result["job"]["skills"] == [
        "Python",
        "Selenium",
        "Pytest",
        "Playwright",
        "GitHub Actions",
    ]
    assert result["job"]["core_skills"] == ["Python", "Selenium", "Pytest"]
    assert result["job"]["required_skills"] == ["Python", "Selenium", "Pytest"]
    assert result["job"]["preferred_skills"] == ["Playwright", "GitHub Actions"]
    assert result["job"]["metadata"]["location"] == "Calgary, AB"
    assert result["job"]["metadata"]["work_model"] == "Remote"
    assert result["job"]["metadata"]["employment_type"] == "Contract"
    assert result["job"]["metadata"]["salary"] == "$45 per hour"
    assert result["result"]["matched_skills"] == ["Python", "Pytest"]
    assert result["result"]["core_skill_match_percent"] == pytest.approx(66.7, rel=1e-2)
    assert result["result"]["job_role"] == "QA/SDET"
    assert result["result"]["candidate_role"] == "QA/SDET"
    assert result["result"]["role_fit_score"] == 100
    assert result["result"]["experience_fit_score"] == 100
    assert result["result"]["apply_score"] == pytest.approx(63.3, rel=1e-2)
    assert result["result"]["apply_recommendation"] == "APPLY"
    assert result["feedback"]["missing_skills_text"] == "Selenium, Playwright, GitHub Actions"
