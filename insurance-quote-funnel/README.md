# Insurance Quote Funnel (Multi-step Form)

A resume-ready, multi-step insurance quote funnel with URL-based step persistence, validation, and real-world error handling patterns.

## Features
- URL-based step routing: `/apply/step/1` through `/apply/step/4`
- Refresh + back/forward safe (step + data persistence)
- Form state saved to localStorage with a "Saved" timestamp
- Submission lock (24h TTL) to prevent duplicate submissions
- Field errors vs server errors + toast notifications
- Mock API for realistic submit flow (success/error/timeout)
- Support mode: view current state/step/recent error
- Mobile-friendly layout and touch targets

## Metrics and event design (example)
These are the types of events I would wire to GA4/GTM for funnel analysis:\n\n- `quote_step_view` (step)\n- `quote_step_complete` (step)\n- `quote_submit` (success)\n- `quote_submit_error` (error_type)\n- `thankyou_cta_click` (cta)\n\nThis enables drop-off analysis and faster UX iteration.

## Why URL-based steps
Using the URL as the step source of truth makes refresh, browser navigation, and deep linking reliable. It also enables customer support to reproduce issues by sharing a specific URL and step state.

## Error handling strategy
- Inline field errors for validation issues
- Server error banners + toast notifications for API failures
- Timeout handling with clear user messaging
- Submission lock to prevent accidental re-submits

## Local development
```bash
npm install
npm run dev
```

## Mock API
```bash
npm run mock:api
```

## Run locally (required for submit to work)
You must run both the mock API and the frontend, or submissions will fail.

Terminal A:
```bash
cd /Users/suyeonkim/Desktop/Projects/insurance-quote-funnel
npm run mock:api
```

Terminal B:
```bash
cd /Users/suyeonkim/Desktop/Projects/insurance-quote-funnel
npm run dev
```

## Test modes (append to URL)
- `?mode=success` (default)
- `?mode=error`
- `?mode=timeout`
- `?mode=random`

Example:
`http://localhost:5173/apply/step/4?mode=timeout`

## Tests
```bash
npm test -- --run
```

## Deployment
- Vercel: https://insurance-quote-funnel.vercel.app
- Netlify: (optional)

## Source
- GitHub: https://github.com/suyeonkim1010/Projects/tree/main/insurance-quote-funnel

## Resume bullets (example)
- Built a multi-step insurance quote funnel with URL-state persistence, validation, and resilient error handling.
- Implemented localStorage recovery, submission lock, and mock API integration to mirror production workflows.
