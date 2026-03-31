from unittest.mock import Mock, patch

import main
import pytest

import requests

from main import (
    build_sorted_results,
    build_job_from_user,
    calculate_match,
    fetch_jobs_real_api,
    get_match_level,
    is_valid_user_payload,
    main as run_main,
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
