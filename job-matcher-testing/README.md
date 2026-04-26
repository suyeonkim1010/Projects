# Job Matcher Testing

QA/SDET practice project for analyzing resumes and pasted job descriptions, extracting skills, calculating match scores, and generating application feedback.

This project is designed as a practical workflow tool rather than a static scoring demo.

## Product Flow

```text
[User]
  -> upload resume
  -> paste job description
[Program]
  -> extract resume skills
  -> extract job skills
  -> calculate match score
  -> generate feedback
  -> generate resume improvement comments
  -> save pasted JD in session history
  -> show results in Streamlit UI
```

## Features

- Extract skills from uploaded PDF/TXT resumes
- Extract skills from job-description text using pattern matching
- Add and remove custom skills and keywords directly in the UI
- Match extracted job skills against extracted resume skills
- Score the match by percent and level
- Generate matched-skills and missing-skills feedback
- Generate resume improvement comments based on missing skills
- Save extracted resume skills locally so they survive refresh
- Save pasted job descriptions in local history
- Show a Streamlit UI for resume-to-JD comparison
- Retry failed HTTP requests
- Validate API-style failure handling with automated tests
- Run automated tests with GitHub Actions CI across multiple Python versions

## Project Files

- `main.py`: extraction, matching, and feedback logic
- `streamlit_app.py`: deployed UI flow
- `test_main.py`: pytest test cases
- `requirements.txt`: project dependencies

## Setup

```bash
cd /Users/suyeonkim/Desktop/Projects/job-matcher-testing
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run Streamlit UI

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown by Streamlit, usually:

```bash
http://localhost:8501
```

On first run, Streamlit may ask for an email address in the terminal.
This is optional and comes from Streamlit itself, not from this project.

## Test

```bash
python3 -m pytest -v --cov=main --cov-report=term-missing
```

## User Guide

### 1. Add Your Resume

You can provide your resume in either of these ways:

- upload a `PDF` file
- upload a `TXT` file

The app extracts recognized skills automatically from the resume content.

### 2. Add a Job Description

Paste the full job description into the text area.  
The app uses the first non-empty line as the job title automatically.

This version does **not** depend on job-board URLs, which makes it more stable than direct scraping.

### 3. Analyze the Match

Click `Analyze Job`.

The app will:

- extract resume skills
- extract job skills
- compare them
- calculate a percent match
- assign a level such as `PERFECT`, `GOOD`, `OK`, or `BAD`
- generate feedback
- generate resume improvement comments

### 4. Review the Results

The UI shows:

- match percent
- match level
- apply score
- apply recommendation
- extracted resume skills
- extracted job skills
- matched skills
- missing skills
- feedback summary
- resume improvement comments

Apply recommendation guide:

- `80+` -> `STRONG APPLY`
- `60-79` -> `APPLY`
- `<60` -> `SKIP`

### 4.1 How the Scores Are Calculated

#### Skill Match

This is the base percentage match between:

- skills extracted from the JD
- skills extracted from the resume

Example:

- JD skills: `Python, Selenium, Pytest`
- Resume skills: `Python, Pytest`
- Skill Match: `2 / 3 = 66.7%`

#### Core Skill Match

Not all skills are treated equally.

The app tries to identify more important JD skills by looking at:

- sentences containing words like `required`, `must`, `minimum`, `requirements`, or `qualifications`
- earlier JD sentences, excluding `preferred`, `nice to have`, or `bonus` wording

Then it calculates how many of those core skills appear in the resume skill list.

#### Required Skills vs Preferred Skills

The app also separates JD skills into two buckets:

- `Required Skills`
- `Preferred Skills`

Rules:

- `required`, `must`, `minimum`, `required qualifications`, `minimum qualifications`
  -> `Required Skills`
- `preferred`, `nice to have`, `bonus`, `asset`
  -> `Preferred Skills`

This helps distinguish hard requirements from optional extras.

#### Role Fit

Role fit is a rule-based category check.

The app tries to infer whether the JD and resume are closer to:

- `QA/SDET`
- `Frontend`
- `Backend`
- `Data`

If both sides fall into the same category, role fit is higher.

Important:

- this is a rough heuristic
- it is less trustworthy than `Skill Match` or `Core Skill Match`

#### Experience Fit

The app looks for year requirements in the JD, such as:

- `5+ years`
- `3 years`
- `2 yrs`

It also tries to estimate experience from the resume:

- explicit year counts if present
- otherwise `entry-level`, `junior`, or `intern` implies a low estimate

Then it compares:

- `candidate_years / required_years * 100`

If the candidate has fewer years than required, the app applies an additional penalty:

- `experience_fit_score *= 0.5`

This makes large experience gaps hurt more realistically.

#### Final Apply Score

The final score is calculated as:

```python
apply_score = (
    0.5 * skill_match +
    0.2 * core_skill_match +
    0.2 * role_fit +
    0.1 * experience_fit
)
```

Interpretation:

- `80+` -> `STRONG APPLY`
- `60-79` -> `APPLY`
- `<60` -> `SKIP`

### 5. Add Missing Skills or Keywords

If the app misses an important skill in either the resume or the JD:

- open the `Custom Skills / Keywords` section in the sidebar
- enter a `Skill Name`
- enter comma-separated keywords or phrases
- click `Add or Update Skill`

Example:

- `Skill Name`: `Jira`
- `Keywords`: `jira, atlassian jira`

After that, the new skill is used immediately for both:

- resume skill extraction
- JD skill extraction

### 6. Reuse the Resume After Refresh

Once a resume has been uploaded or pasted, the app stores:

- the resume text
- the extracted resume skills

in a local cache file for this app instance.

That means you do **not** need to upload or paste the resume again after refreshing the page.

Use `Clear Resume` if you want to remove the saved resume and switch to a different one.

### 7. Review Saved Job Descriptions

Each pasted JD is stored in local history.

The app keeps a short recent list so you can:

- review previous pasted job descriptions
- compare multiple jobs in one session

## Current Limitations

- skill extraction uses rule-based pattern matching, not LLM parsing
- unsupported or unusual skill wording may need to be added through the custom-skills sidebar
- direct job-board scraping is intentionally not the primary workflow now

## Testing Strategy

This project covers:

- normal matching behavior
- broken or missing input data
- invalid JSON / unexpected API response handling
- skill extraction from resume text
- skill extraction from description text
- job-description extraction from HTML
- end-to-end text analysis flow
- resume improvement comment generation

Mocking is still used for the API-oriented test coverage that already exists in this project.

## CI/CD

GitHub Actions automatically runs the test suite on:

- push to the repository
- pull requests
- Python 3.11, 3.12, and 3.13

The workflow installs dependencies and runs:

```bash
python -m pytest -v --cov=main --cov-report=term-missing
```

This is a CI pipeline. Deployment is handled separately through Streamlit Community Cloud.

## Deployment

### Recommended Free Option

For this project, the most practical free deployment target is **Streamlit Community Cloud**.

Why this fits:

- it supports lightweight Python apps well
- it deploys directly from GitHub
- it matches the current UI architecture without needing Flask or FastAPI

### Deploy on Streamlit Community Cloud

1. Push `job-matcher-testing` to GitHub
2. Open Streamlit Community Cloud
3. Choose the repository and branch
4. Set the entrypoint file to `job-matcher-testing/streamlit_app.py`
5. Deploy

After deployment, the app gets a public `streamlit.app` URL and updates when new commits are pushed.
