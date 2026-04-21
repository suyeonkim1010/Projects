# Job Matcher Testing

Simple Python project for SDET practice.

This project matches user skills against job posts, validates real API responses, and tests system behavior under normal, failure, and edge-case conditions.
It handles API failures such as timeout, server errors, and invalid responses.
It uses mocking to simulate external API behavior and validate retry logic without relying on live network failures.
It is designed to test system stability under failure conditions.
It also includes a lightweight Streamlit app so the project can be hosted with a free deployment option.

## Features

- Match job skills against user skills
- Score each job by percent
- Sort jobs from highest match to lowest match
- Handle broken job data safely
- Validate API response schema before processing data
- Retry failed API calls for transient timeout and request errors
- Handle API failures such as timeout, server errors, and invalid responses
- Use mocking to simulate external API behavior
- Test system stability under failure conditions
- Measure coverage with `pytest-cov`
- Run automated tests with GitHub Actions CI across multiple Python versions
- Cover end-to-end API-to-ranking flow with automated tests
- Include a Streamlit UI for free-hosted demo deployment

## Project Files

- `main.py`: matching logic and console output
- `test_main.py`: pytest test cases
- `streamlit_app.py`: lightweight demo UI for deployment
- `requirements.txt`: project dependencies

## Setup

```bash
cd /Users/suyeonkim/Desktop/Projects/job-matcher-testing
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 main.py
```

## Run Streamlit Demo

```bash
streamlit run streamlit_app.py
```

## Test

```bash
python3 -m pytest -v --cov=main --cov-report=term-missing
```

### Run One Test

```bash
python3 -m pytest -v test_main.py::test_real_api_timeout_returns_empty_job_list
```

### What To Test

- Run `python3 -m pytest -v` for automated test validation
- Run `python3 main.py` to check the real program output
- Use single-test execution when you want to debug one case at a time
- Review coverage output to check which branches are tested

## Test Case Types

### Normal Case

A normal case means the input is valid and the program behaves as expected.

Examples:
- a job has a valid `title`
- a job has a valid `skills` list
- `my_skills` is a valid list
- the match result is calculated correctly

### Failure Case

A failure case means something goes wrong, such as bad input or an API problem.
The important point is that the program should handle the problem safely without crashing.

Examples:
- API timeout
- API error
- bad API response
- missing `skills`
- `skills = None`

### Edge Case

An edge case is a boundary or unusual input that can easily be missed during testing.
These cases help verify that the logic still works correctly in tricky situations.

Examples:
- `49%` should be `OK`
- `50%` should be `GOOD`
- `1%` should be `OK`
- `0%` should be `BAD`
- `job = {}`
- `job = {"skills": "Python"}`
- `my_skills = None`
- duplicate skills like `["Python", "Python"]`
- case-insensitive matching like `"Python"` vs `"python"`

## Real API and Mocking

This project uses a real HTTP API with the `requests` library.
The current example API is:

- `https://jsonplaceholder.typicode.com/users`

In `main.py`, the `fetch_jobs_real_api()` function sends a real HTTP request and converts the response into simple job-like data.
It also validates the response schema and retries transient request failures.

In `test_main.py`, API failures are tested with mocking instead of real network calls.
This makes the tests faster, stable, and repeatable.

Mocking is used to simulate:
- success response
- timeout
- request error
- bad response
- invalid response schema
- retry-then-success flow

## Quality Improvements

This project was strengthened with additional SDET-focused improvements:

- response schema validation before transforming API data
- retry logic for transient API failures
- end-to-end flow testing from API response to final ranked output
- coverage reporting for test completeness
- CI execution across multiple Python versions

## CI/CD

GitHub Actions automatically runs the test suite on:

- push to the repository
- pull requests
- Python 3.11, 3.12, and 3.13

The workflow installs dependencies and runs:

```bash
python -m pytest -v --cov=main --cov-report=term-missing
```

This is a CI pipeline, not a full release pipeline. It validates code changes, installs dependencies, and runs the automated test suite across multiple Python versions.

## Deployment

### Recommended Free Option

For this project, the most practical free deployment target is **Streamlit Community Cloud**.

Why this fits:
- It is free for Streamlit apps
- It deploys directly from GitHub
- It matches this project's lightweight Python demo format better than a full web-service host

Other options:
- **Render** also offers free web services, but it is more useful when you already have a web server such as Flask or FastAPI
- **Railway** is no longer the best "fully free" option for an ongoing student demo because its free usage is limited by credits/trial usage

### Deployment Files

This repository now includes:
- `streamlit_app.py` as the demo entrypoint
- `requirements.txt` with `streamlit`

### Deploy on Streamlit Community Cloud

1. Push `job-matcher-testing` to GitHub
2. Open Streamlit Community Cloud
3. Choose the repository and branch
4. Set the entrypoint file to `job-matcher-testing/streamlit_app.py`
5. Deploy

After that, Streamlit hosts the app on a public `streamlit.app` URL and updates it when you push new commits.

## Notes

- One test is marked with `xfail` on purpose to show an intentional failure example.
- Recommended workflow is to use the local virtual environment instead of system Python.
