import json
import logging
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


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
        {"name": "OK", "min_percent": 30},
        {"name": "BAD", "min_percent": 0},
    ]
}

REAL_API_URL = "https://jsonplaceholder.typicode.com/users"
MAX_API_RETRIES = 2

CUSTOM_SKILL_PATTERNS_PATH = Path(__file__).with_name("custom_skill_patterns.json")

DEFAULT_SKILL_PATTERNS = {
    "Python": [r"\bpython\b"],
    "Selenium": [r"\bselenium\b"],
    "Pytest": [r"\bpytest\b"],
    "SQL": [r"\bsql\b"],
    "API Testing": [r"\bapi testing\b", r"\brest api\b", r"\brestful api\b"],
    "Postman": [r"\bpostman\b"],
    "Playwright": [r"\bplaywright\b"],
    "Cypress": [r"\bcypress\b"],
    "JavaScript": [r"\bjavascript\b"],
    "TypeScript": [r"\btypescript\b"],
    "React": [r"\breact\b"],
    "HTML": [r"\bhtml\b"],
    "CSS": [r"\bcss\b"],
    "Git": [r"\bgit\b"],
    "GitHub Actions": [r"\bgithub actions\b"],
    "CI/CD": [r"\bci/?cd\b", r"\bcontinuous integration\b", r"\bcontinuous delivery\b"],
    "Automation Testing": [r"\bautomation testing\b", r"\btest automation\b"],
    "Manual Testing": [r"\bmanual testing\b"],
    "Docker": [r"\bdocker\b"],
    "AWS": [r"\baws\b", r"\bamazon web services\b"],
}

DESCRIPTION_HINTS = [
    "job-description",
    "job_description",
    "jobdescription",
    "description",
    "posting",
    "details",
    "content",
    "requirements",
    "qualifications",
    "responsibilities",
]

WORK_MODEL_PATTERNS = [
    ("Remote", r"\bremote\b"),
    ("Hybrid", r"\bhybrid\b"),
    ("On-site", r"\bon[-\s]?site\b|\bin[-\s]?person\b"),
]

EMPLOYMENT_TYPE_PATTERNS = [
    ("Full-time", r"\bfull[-\s]?time\b"),
    ("Part-time", r"\bpart[-\s]?time\b"),
    ("Contract", r"\bcontract\b"),
    ("Temporary", r"\btemporary\b"),
    ("Internship", r"\bintern(ship)?\b"),
]

CORE_SENTENCE_HINTS = [
    "required",
    "must",
    "minimum",
    "requirements",
    "qualifications",
]

NON_CORE_SENTENCE_HINTS = [
    "nice to have",
    "preferred",
    "bonus",
]

REQUIRED_SENTENCE_HINTS = [
    "required",
    "must",
    "minimum",
    "required qualifications",
    "minimum qualifications",
]

PREFERRED_SENTENCE_HINTS = [
    "preferred",
    "nice to have",
    "bonus",
    "asset",
]

ROLE_PATTERNS = {
    "QA/SDET": [
        r"\bqa\b",
        r"\bsdet\b",
        r"\btest automation\b",
        r"\bautomation testing\b",
        r"\bquality assurance\b",
        r"\bselenium\b",
        r"\bpytest\b",
        r"\bplaywright\b",
        r"\bcypress\b",
    ],
    "Frontend": [
        r"\bfrontend\b",
        r"\bfront-end\b",
        r"\breact\b",
        r"\bjavascript\b",
        r"\btypescript\b",
        r"\bhtml\b",
        r"\bcss\b",
        r"\bui\b",
    ],
    "Backend": [
        r"\bbackend\b",
        r"\bback-end\b",
        r"\bapi\b",
        r"\bnode\b",
        r"\bexpress\b",
        r"\bdjango\b",
        r"\bflask\b",
        r"\bjava\b",
        r"\b\.net\b",
    ],
    "Data": [
        r"\bdata\b",
        r"\banalyst\b",
        r"\bmachine learning\b",
        r"\bsql\b",
        r"\bpower bi\b",
        r"\bpandas\b",
    ],
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


def fetch_page_html(url, timeout=10, retries=MAX_API_RETRIES):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }

    for attempt in range(1, retries + 2):
        try:
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            logger.info("Fetched HTML from %s on attempt %s", url, attempt)
            return response.text
        except requests.exceptions.Timeout:
            logger.error("HTML fetch timeout on attempt %s", attempt)
            if attempt <= retries:
                continue
            return None
        except requests.exceptions.RequestException:
            logger.error("HTML fetch failed on attempt %s", attempt)
            if attempt <= retries:
                continue
            return None

    return None


def clean_text(raw_text):
    return re.sub(r"\s+", " ", raw_text).strip()


def keyword_to_pattern(keyword):
    escaped_keyword = re.escape(clean_text(keyword))
    spaced_keyword = escaped_keyword.replace(r"\ ", r"\s+")
    return rf"(?<!\w){spaced_keyword}(?!\w)"


def load_custom_skill_keywords():
    if not CUSTOM_SKILL_PATTERNS_PATH.exists():
        return {}

    try:
        raw_data = json.loads(CUSTOM_SKILL_PATTERNS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.error("Could not load custom skill keywords")
        return {}

    if not isinstance(raw_data, dict):
        return {}

    cleaned_data = {}

    for skill_name, keywords in raw_data.items():
        if not isinstance(skill_name, str) or not skill_name.strip():
            continue

        if not isinstance(keywords, list):
            continue

        cleaned_keywords = clean_skills(keywords)

        if cleaned_keywords:
            cleaned_data[clean_text(skill_name)] = cleaned_keywords

    return cleaned_data


def save_custom_skill_keywords(custom_skills):
    cleaned_data = {}

    for skill_name, keywords in custom_skills.items():
        if not isinstance(skill_name, str) or not skill_name.strip():
            continue

        cleaned_keywords = clean_skills(keywords)

        if cleaned_keywords:
            cleaned_data[clean_text(skill_name)] = cleaned_keywords

    CUSTOM_SKILL_PATTERNS_PATH.write_text(
        json.dumps(cleaned_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upsert_custom_skill_keywords(skill_name, keywords):
    cleaned_skill_name = clean_text(skill_name)
    cleaned_keywords = clean_skills(keywords)

    if not cleaned_skill_name or not cleaned_keywords:
        return False

    custom_skills = load_custom_skill_keywords()
    custom_skills[cleaned_skill_name] = cleaned_keywords
    save_custom_skill_keywords(custom_skills)
    return True


def remove_custom_skill(skill_name):
    custom_skills = load_custom_skill_keywords()

    if skill_name in custom_skills:
        del custom_skills[skill_name]
        save_custom_skill_keywords(custom_skills)
        return True

    return False


def get_skill_patterns():
    merged_patterns = dict(DEFAULT_SKILL_PATTERNS)

    for skill_name, keywords in load_custom_skill_keywords().items():
        merged_patterns[skill_name] = [keyword_to_pattern(keyword) for keyword in keywords]

    return merged_patterns


def extract_job_title(soup):
    title_candidates = [
        soup.find("meta", property="og:title"),
        soup.find("meta", attrs={"name": "twitter:title"}),
    ]

    for candidate in title_candidates:
        if candidate and candidate.get("content"):
            return clean_text(candidate["content"])

    heading = soup.find("h1")
    if heading:
        return clean_text(heading.get_text(" ", strip=True))

    if soup.title and soup.title.string:
        return clean_text(soup.title.string)

    return "Untitled Job"


def extract_apply_url(soup, base_url):
    for link in soup.find_all("a", href=True):
        link_text = clean_text(link.get_text(" ", strip=True)).lower()
        href = link["href"].strip()
        if "apply" in link_text or "apply" in href.lower():
            return urljoin(base_url, href)

    return base_url


def extract_job_description(html):
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in ["script", "style", "noscript"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    candidate_sections = []

    for tag in soup.find_all(["main", "article", "section", "div"]):
        tag_id = (tag.get("id") or "").lower()
        tag_classes = " ".join(tag.get("class", [])).lower()
        haystack = f"{tag_id} {tag_classes}"

        if any(hint in haystack for hint in DESCRIPTION_HINTS):
            candidate_sections.append(clean_text(tag.get_text(" ", strip=True)))

    if candidate_sections:
        return max(candidate_sections, key=len)

    body = soup.body
    if body:
        return clean_text(body.get_text(" ", strip=True))

    return clean_text(soup.get_text(" ", strip=True))


def extract_skills_from_text(text):
    found_skills = []
    normalized_text = text.lower()

    for skill_name, patterns in get_skill_patterns().items():
        if any(re.search(pattern, normalized_text) for pattern in patterns):
            found_skills.append(skill_name)

    return clean_skills(found_skills)


def extract_skills_from_resume_text(resume_text):
    return extract_skills_from_text(resume_text)


def split_into_sentences(text):
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", text) if sentence.strip()]


def get_non_empty_lines(text):
    return [clean_text(line) for line in text.splitlines() if clean_text(line)]


def extract_core_skills(job_text, all_skills):
    core_skills = []
    sentences = split_into_sentences(job_text)
    prioritized_sentences = [
        sentence for sentence in sentences
        if any(hint in sentence.lower() for hint in CORE_SENTENCE_HINTS)
    ]
    early_sentences = [
        sentence for sentence in sentences[:3]
        if not any(hint in sentence.lower() for hint in NON_CORE_SENTENCE_HINTS)
    ]

    for skill in all_skills:
        skill_patterns = get_skill_patterns().get(skill, [])
        found_in_priority = any(
            any(re.search(pattern, sentence.lower()) for pattern in skill_patterns)
            for sentence in prioritized_sentences
        )
        found_early = any(
            any(re.search(pattern, sentence.lower()) for pattern in skill_patterns)
            for sentence in early_sentences
        )
        if found_in_priority or found_early:
            core_skills.append(skill)

    if not core_skills:
        return all_skills[: min(3, len(all_skills))]

    return clean_skills(core_skills)


def extract_bucketed_skills(job_text, all_skills, sentence_hints):
    bucketed_skills = []
    sentences = split_into_sentences(job_text)
    matched_sentences = [
        sentence for sentence in sentences
        if any(hint in sentence.lower() for hint in sentence_hints)
    ]

    for skill in all_skills:
        skill_patterns = get_skill_patterns().get(skill, [])
        if any(
            any(re.search(pattern, sentence.lower()) for pattern in skill_patterns)
            for sentence in matched_sentences
        ):
            bucketed_skills.append(skill)

    return clean_skills(bucketed_skills)


def extract_required_years(job_text):
    matches = re.findall(r"(\d+)\+?\s*(?:years|yrs)", job_text, flags=re.IGNORECASE)
    if not matches:
        return None

    return max(int(value) for value in matches)


def infer_candidate_years(resume_text):
    text = resume_text.lower()
    matches = re.findall(r"(\d+)\+?\s*(?:years|yrs)", text, flags=re.IGNORECASE)
    if matches:
        return max(int(value) for value in matches)

    if any(keyword in text for keyword in ["entry-level", "new grad", "junior", "intern"]):
        return 1

    return 0


def infer_role_category(text):
    lowered_text = text.lower()
    best_role = "Unknown"
    best_score = 0

    for role_name, patterns in ROLE_PATTERNS.items():
        score = sum(1 for pattern in patterns if re.search(pattern, lowered_text))
        if score > best_score:
            best_role = role_name
            best_score = score

    return best_role


def calculate_role_fit(job_text, resume_text):
    job_role = infer_role_category(job_text)
    candidate_role = infer_role_category(resume_text)

    if job_role == "Unknown" or candidate_role == "Unknown":
        return {
            "job_role": job_role,
            "candidate_role": candidate_role,
            "role_fit_score": 50,
        }

    if job_role == candidate_role:
        score = 100
    elif {job_role, candidate_role} <= {"QA/SDET", "Backend"}:
        score = 60
    elif {job_role, candidate_role} <= {"Frontend", "Backend"}:
        score = 55
    else:
        score = 20

    return {
        "job_role": job_role,
        "candidate_role": candidate_role,
        "role_fit_score": score,
    }


def calculate_experience_fit(job_text, resume_text):
    required_years = extract_required_years(job_text)
    candidate_years = infer_candidate_years(resume_text)

    if required_years is None:
        experience_fit_score = 100
    elif required_years <= 0:
        experience_fit_score = 100
    else:
        experience_fit_score = min((candidate_years / required_years) * 100, 100)
        if candidate_years < required_years:
            experience_fit_score *= 0.5

    return {
        "required_years": required_years,
        "candidate_years": candidate_years,
        "experience_fit_score": round(experience_fit_score, 1),
    }


def should_apply(score):
    if score >= 80:
        return "STRONG APPLY"
    if score >= 60:
        return "APPLY"
    return "SKIP"


def clean_location_value(location_text):
    cleaned_location = clean_text(location_text)
    cleaned_location = re.sub(r"(?i)^location\s*:\s*", "", cleaned_location)
    cleaned_location = re.sub(
        r"\b(?:in[-\s]?person|on[-\s]?site|remote|hybrid)\b.*$",
        "",
        cleaned_location,
        flags=re.IGNORECASE,
    )
    return clean_text(cleaned_location.rstrip(",-/|"))


def extract_job_metadata(job_text):
    cleaned_text = clean_text(job_text)

    salary_match = re.search(
        r"(\$[\d,]+(?:\.\d{1,2})?\s*(?:-|to)\s*\$[\d,]+(?:\.\d{1,2})?(?:\s*(?:a year|per year|an hour|per hour))?)",
        cleaned_text,
        flags=re.IGNORECASE,
    ) or re.search(
        r"(\$[\d,]+(?:\.\d{1,2})?\s*(?:a year|per year|an hour|per hour))",
        cleaned_text,
        flags=re.IGNORECASE,
    )

    non_empty_lines = get_non_empty_lines(job_text)
    preferred_location_line = None
    if len(non_empty_lines) >= 5:
        fifth_line = non_empty_lines[4]
        if re.search(r"(?i)\blocation\b", fifth_line) or re.search(
            r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*,\s*(?:AB|BC|MB|NB|NL|NS|NT|NU|ON|PE|QC|SK|YT)\b",
            fifth_line,
        ):
            preferred_location_line = fifth_line

    location_match = re.search(
        r"(?:location\s*:\s*)([^.;\n]+)",
        job_text,
        flags=re.IGNORECASE,
    ) or re.search(
        r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*,\s*(?:AB|BC|MB|NB|NL|NS|NT|NU|ON|PE|QC|SK|YT))\b",
        job_text,
    )

    work_model = "Not found"
    for label, pattern in WORK_MODEL_PATTERNS:
        if re.search(pattern, cleaned_text, flags=re.IGNORECASE):
            work_model = label
            break

    employment_type = "Not found"
    for label, pattern in EMPLOYMENT_TYPE_PATTERNS:
        if re.search(pattern, cleaned_text, flags=re.IGNORECASE):
            employment_type = label
            break

    return {
        "salary": clean_text(salary_match.group(1)) if salary_match else "Not found",
        "location": (
            clean_location_value(preferred_location_line)
            if preferred_location_line
            else clean_location_value(location_match.group(1)) if location_match else "Not found"
        ),
        "work_model": work_model,
        "employment_type": employment_type,
    }


def read_resume_file(uploaded_file):
    file_name = (uploaded_file.name or "").lower()

    if file_name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    raw_bytes = uploaded_file.read()
    return raw_bytes.decode("utf-8", errors="ignore")


def build_scraped_job(url, html):
    soup = BeautifulSoup(html, "html.parser")
    title = extract_job_title(soup)
    description = extract_job_description(html)
    skills = extract_skills_from_text(description)
    apply_url = extract_apply_url(soup, url)

    return {
        "url": url,
        "title": title,
        "description": description,
        "skills": skills,
        "apply_url": apply_url,
        "metadata": extract_job_metadata(description),
    }


def generate_feedback(result):
    matched = result.get("matched_skills", [])
    missing = [
        skill for skill in result.get("skills", []) if normalize_skill(skill) not in {
            normalize_skill(matched_skill) for matched_skill in matched
        }
    ]

    if result["level"] == "PERFECT":
        summary = "Strong match. Your current skill set covers the extracted requirements."
    elif result["level"] == "GOOD":
        summary = "Solid match. You already cover a good portion of the extracted requirements."
    elif result["level"] == "OK":
        summary = "Partial match. You have some relevant skills, but there are clear gaps."
    else:
        summary = "Weak match based on the extracted requirements. Review the missing skills before applying."

    return {
        "summary": summary,
        "matched_skills_text": ", ".join(matched) or "None",
        "missing_skills_text": ", ".join(missing) or "None",
    }


def generate_resume_improvement_comments(result):
    missing = [
        skill for skill in result.get("skills", []) if normalize_skill(skill) not in {
            normalize_skill(matched_skill) for matched_skill in result.get("matched_skills", [])
        }
    ]

    comments = []

    if not result.get("skills"):
        comments.append(
            "The pasted job description did not produce recognized skills. Add a clearer responsibility or requirements section."
        )
        return comments

    if missing:
        comments.append(
            f"Add or strengthen evidence for these missing skills if you have them: {', '.join(missing)}."
        )
        comments.append(
            "If you used any of those skills in projects, make them explicit in bullet points instead of implying them."
        )

    if result["level"] in {"BAD", "OK"}:
        comments.append(
            "Rewrite one project bullet to align more directly with the job's required tools and testing responsibilities."
        )

    comments.append(
        "Mirror the language used in the JD more closely so your resume is easier to scan in ATS and recruiter review."
    )

    return comments


def analyze_job_url(url, my_skill_list, resume_text=""):
    html = fetch_page_html(url)

    if not html:
        return None

    job = build_scraped_job(url, html)
    result = calculate_match(job, my_skill_list, resume_text, job["description"])
    feedback = generate_feedback(result)

    return {
        "job": job,
        "result": result,
        "feedback": feedback,
        "html_length": len(html),
    }


def analyze_job_text(job_text, my_skill_list, title="Pasted Job Description", resume_text=""):
    cleaned_description = clean_text(job_text)
    skills = extract_skills_from_text(cleaned_description)
    core_skills = extract_core_skills(job_text, skills)
    required_skills = extract_bucketed_skills(job_text, skills, REQUIRED_SENTENCE_HINTS)
    preferred_skills = extract_bucketed_skills(job_text, skills, PREFERRED_SENTENCE_HINTS)
    job = {
        "title": title or "Pasted Job Description",
        "description": cleaned_description,
        "skills": skills,
        "core_skills": core_skills,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "apply_url": None,
        "metadata": extract_job_metadata(job_text),
    }
    result = calculate_match(job, my_skill_list, resume_text, cleaned_description)
    feedback = generate_feedback(result)

    return {
        "job": job,
        "result": result,
        "feedback": feedback,
    }


def infer_job_title_from_text(job_text):
    for line in job_text.splitlines():
        cleaned_line = clean_text(line)
        if cleaned_line:
            return cleaned_line[:120]

    return "Pasted Job Description"


def calculate_match(job, my_skill_list, resume_text="", job_text=""):
    job_title = job.get("title", "Untitled Job")
    job_skills = clean_skills(job.get("skills"))
    core_skills = clean_skills(job.get("core_skills"))
    normalized_my_skills = {normalize_skill(skill) for skill in clean_skills(my_skill_list)}
    matched_skills = [
        skill for skill in job_skills if normalize_skill(skill) in normalized_my_skills
    ]
    match_count = len(matched_skills)
    match_percent = (match_count / len(job_skills)) * 100 if job_skills else 0
    matched_core_skills = [
        skill for skill in core_skills if normalize_skill(skill) in normalized_my_skills
    ]
    core_skill_match_percent = (
        (len(matched_core_skills) / len(core_skills)) * 100 if core_skills else match_percent
    )
    role_fit = calculate_role_fit(f"{job_title} {job_text}", resume_text)
    experience_fit = calculate_experience_fit(job_text, resume_text)
    apply_score = (
        0.5 * match_percent
        + 0.2 * core_skill_match_percent
        + 0.2 * role_fit["role_fit_score"]
        + 0.1 * experience_fit["experience_fit_score"]
    )
    level = get_match_level(match_percent)

    return {
        "title": job_title,
        "skills": job_skills,
        "core_skills": core_skills,
        "matched_skills": matched_skills,
        "matched_core_skills": matched_core_skills,
        "match_count": match_count,
        "match_percent": match_percent,
        "core_skill_match_percent": round(core_skill_match_percent, 1),
        "role_fit_score": role_fit["role_fit_score"],
        "job_role": role_fit["job_role"],
        "candidate_role": role_fit["candidate_role"],
        "required_years": experience_fit["required_years"],
        "candidate_years": experience_fit["candidate_years"],
        "experience_fit_score": experience_fit["experience_fit_score"],
        "apply_score": round(apply_score, 1),
        "apply_recommendation": should_apply(apply_score),
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
