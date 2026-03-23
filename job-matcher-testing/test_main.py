import pytest

from main import build_sorted_results, calculate_match, fetch_jobs_from_api


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


def test_timeout_api_returns_empty_job_list():
    response = {
        "status": "timeout",
        "data": None,
    }

    result = fetch_jobs_from_api(response)

    assert result == []


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


def test_1_percent_is_ok():
    job = {
        "title": "Boundary 1 Percent Job",
        "skills": ["Python"] + [f"Skill{i}" for i in range(2, 101)],
    }

    result = calculate_match(job, ["python"])

    assert result["match_percent"] == 1
    assert result["level"] == "OK"


def test_0_percent_is_bad():
    job = {
        "title": "Boundary 0 Percent Job",
        "skills": ["Python", "Selenium"],
    }

    result = calculate_match(job, ["docker"])

    assert result["match_percent"] == 0
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
