# Job Matcher Testing

Simple Python project for SDET practice.

This project matches user skills against job posts, handles broken input safely, and includes pytest tests for normal and failure scenarios.
It handles API failures such as timeout, server errors, and invalid responses.
It uses mocking to simulate external API behavior.
It is designed to test system stability under failure conditions.

## Features

- Match job skills against user skills
- Score each job by percent
- Sort jobs from highest match to lowest match
- Handle broken job data safely
- Simulate API timeout, error, and bad response cases
- Handle API failures such as timeout, server errors, and invalid responses
- Use mocking to simulate external API behavior
- Test system stability under failure conditions
- Test behavior with pytest
- Run automated tests with GitHub Actions CI

## Project Files

- `main.py`: matching logic and console output
- `test_main.py`: pytest test cases
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

## Test

```bash
python3 -m pytest -v
```

### Run One Test

```bash
python3 -m pytest -v test_main.py::test_real_api_timeout_returns_empty_job_list
```

### What To Test

- Run `python3 -m pytest -v` for automated test validation
- Run `python3 main.py` to check the real program output
- Use single-test execution when you want to debug one case at a time

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

In `test_main.py`, API failures are tested with mocking instead of real network calls.
This makes the tests faster, stable, and repeatable.

Mocking is used to simulate:
- success response
- timeout
- request error
- bad response

## Notes

- One test is marked with `xfail` on purpose to show an intentional failure example.
- Recommended workflow is to use the local virtual environment instead of system Python.
