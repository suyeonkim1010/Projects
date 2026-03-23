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


fake_api_responses = [
    {
        "status": "success",
        "data": job_posts,
    },
    {
        "status": "timeout",
        "data": None,
    },
    {
        "status": "error",
        "data": None,
    },
    {
        "status": "bad_response",
        "data": {"wrong_key": "wrong_value"},
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


def fetch_jobs_from_api(response):
    status = response.get("status")

    if status == "timeout":
        print("API Result: TIMEOUT")
        return []

    if status == "error":
        print("API Result: ERROR")
        return []

    if status == "bad_response":
        print("API Result: BAD RESPONSE")
        return []

    data = response.get("data")

    if not isinstance(data, list):
        print("API Result: INVALID DATA")
        return []

    print("API Result: SUCCESS")
    return data


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
    api_jobs = fetch_jobs_from_api(fake_api_responses[0])
    sorted_results = build_sorted_results(api_jobs, my_skills)

    print("My Skills:", my_skills)
    print()
    print("API Failure Test")

    for response in fake_api_responses:
        fetch_jobs_from_api(response)

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
