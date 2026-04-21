# Frontend Candidate Dashboard

A React portfolio project that simulates a recruiter-facing candidate review dashboard.

It was built to demonstrate practical frontend skills through a realistic UI flow: searching, filtering,
sorting, reviewing, and inspecting candidate profiles in a single responsive interface.

## Highlights

- built with React and Vite
- end-to-end frontend coverage with Playwright
- interactive filtering and search with state-driven rendering
- responsive dashboard layout with reusable UI patterns
- candidate detail view updated without page reload
- live deployed demo for portfolio review

## What it shows

- responsive React dashboard layout
- live search and filtering with state-driven rendering
- sorting by match, experience, or name
- candidate detail panel updates without page reload
- reusable component-style UI patterns
- polished visual hierarchy and interaction states

## Tech

- React
- Vite
- JavaScript
- CSS
- Playwright

## Live Demo

- https://suyeonkim1010.github.io/Projects/frontend-candidate-dashboard/dist/

## Why I Built It

I wanted a frontend project that looked closer to a real product interface than a simple CRUD demo.
This dashboard focuses on interaction design, screen hierarchy, and user flow, which are core parts of
frontend work in entry-level product teams.

## Files

- `index.html`
- `package.json`
- `vite.config.js`
- `src/App.jsx`
- `src/main.jsx`
- `src/index.css`

## Run

```bash
npm install
npm run dev
```

## End-to-End Tests

```bash
npx playwright install
npm run test:e2e
```

Current Playwright coverage includes:
- search filtering
- status filter behavior
- candidate detail panel updates

The tests validate core frontend flows instead of only static rendering.

## Build

```bash
npm run build
```

The production-ready static files are generated in `dist/`.
