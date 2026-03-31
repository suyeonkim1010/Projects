import logging

import requests


logger = logging.getLogger(__name__)


my_skills = [
    "python",
    "selenium",
    "pytest",
    "sql",
]

matching_rules = {
    "levels": [
        {"name": "PERFECT", "min_percent": 100},
        {"name": "GOOD", "min_percent": 50},
        {"name": "OK", "min_percent": 1},
        {"name": "BAD", "min_percent": 0},
    ]
}

REAL_API_URL = "https://jsonplaceholder.typicode.com/users"
MAX_API_RETRIES = 2


job_posts = [
    {
        "title": "QA Automation Engineer",
        "skills": ["Python", "Selenium", "Pytest"],
    },
    {
        "title": "Data QA Tester",
        "skills": ["SQL", "Python"],
    },
    {
        "title": "Junior SDET",
        "skills": ["Python", "API Testing", "Pytest"],
    },
    {
        "title": "Broken Job No Skills",
    },
    {
        "title": "Broken Job None Skills",
        "skills": None,
    },
    {
        "title": "Broken Job Empty Skills",
        "skills": [],
    },
    {
        "title": "Broken Job Weird Skills",
        "skills": ["Python", "", 123, None],
    },
]


def clean_skills(skills):
    if not isinstance(skills, list):
        return []

    cleaned_skills = []
    seen_skills = set()

    for skill in skills:
        if isinstance(skill, str) and skill.strip():
            cleaned_skill = skill.strip()
            normalized_skill = normalize_skill(cleaned_skill)

            if normalized_skill not in seen_skills:
                cleaned_skills.append(cleaned_skill)
                seen_skills.add(normalized_skill)

    return cleaned_skills


def normalize_skill(skill):
    return skill.strip().lower()


def get_match_level(match_percent):
    for rule in matching_rules["levels"]:
        if match_percent >= rule["min_percent"]:
            return rule["name"]

    return "BAD"


def is_valid_user_payload(user):
    if not isinstance(user, dict):
        return False

    company = user.get("company")

    if not isinstance(company, dict):
        return False

    company_name = company.get("name")

    return isinstance(company_name, str) and bool(company_name.strip())


def validate_api_response(data):
    if not isinstance(data, list):
        return False

    return all(is_valid_user_payload(user) for user in data)


def build_job_from_user(user, index):
    skill_templates = [
        ["Python", "Selenium", "Pytest"],
        ["SQL", "Python"],
        ["API Testing", "Pytest"],
    ]
    company_name = user.get("company", {}).get("name", "Unknown Company")
    template = skill_templates[index % len(skill_templates)]

    return {
        "title": f"{company_name} QA Role",
        "skills": template,
    }


def fetch_jobs_real_api(url=REAL_API_URL, timeout=5, retries=MAX_API_RETRIES):
    for attempt in range(1, retries + 2):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            logger.error("API timeout occurred on attempt %s", attempt)

            if attempt <= retries:
                logger.info("Retrying API request after timeout")
                continue

            return []
        except requests.exceptions.RequestException:
            logger.error("API request failed on attempt %s", attempt)

            if attempt <= retries:
                logger.info("Retrying API request after request error")
                continue

            return []
        except ValueError:
            logger.error("API returned invalid JSON")
            return []

        if not validate_api_response(data):
            logger.error("API returned unexpected response format")
            return []

        logger.info("API request succeeded on attempt %s", attempt)
        return [build_job_from_user(user, index) for index, user in enumerate(data)]

    return []


def calculate_match(job, my_skill_list):
    job_title = job.get("title", "Untitled Job")
    job_skills = clean_skills(job.get("skills"))
    normalized_my_skills = {normalize_skill(skill) for skill in clean_skills(my_skill_list)}
    matched_skills = [
        skill for skill in job_skills if normalize_skill(skill) in normalized_my_skills
    ]
    match_count = len(matched_skills)
    match_percent = (match_count / len(job_skills)) * 100 if job_skills else 0
    level = get_match_level(match_percent)

    return {
        "title": job_title,
        "skills": job_skills,
        "matched_skills": matched_skills,
        "match_count": match_count,
        "match_percent": match_percent,
        "level": level,
    }


def build_sorted_results(jobs, my_skill_list):
    results = []

    for job in jobs:
        result = calculate_match(job, my_skill_list)
        results.append(result)

    return sorted(
        results,
        key=lambda job: (job["match_percent"], job["match_count"]),
        reverse=True,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    api_jobs = fetch_jobs_real_api()

    if not api_jobs:
        api_jobs = job_posts

    sorted_results = build_sorted_results(api_jobs, my_skills)

    print("My Skills:", my_skills)
    print()
    print("Job Match Results")

    for job in sorted_results:
        print(f"- {job['title']}")
        print(f"  Required Skills: {job['skills']}")
        print(f"  Matched Skills: {job['matched_skills']}")
        print(f"  Match Count: {job['match_count']}")
        print(f"  Match Percent: {job['match_percent']:.0f}%")
        print(f"  Level: {job['level']}")
        print()

    if sorted_results:
        best_job = sorted_results[0]
        print("Best Job Match")
        print(f"{best_job['title']} ({best_job['match_percent']:.0f}%)")


if __name__ == "__main__":
    main()
