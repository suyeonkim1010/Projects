# Job Tracker Studio

A local-first job application tracker that summarizes job descriptions, tags skills, detects duplicates, and syncs across devices with Supabase.

## Features
- Paste a full job description and auto-extract key fields
- Track application status, deadlines, and skills
- Local-first storage with optional Supabase sync
- Optional AI summary using a local Ollama model

## Quick start (local)
1) Start the app server
```
cd /Users/suyeonkim/Desktop/Projects/job-tracker/server
npm install
npm start
```
2) Open the app
```
http://localhost:5177
```

## Optional: AI summaries with Ollama (free)
1) Install Ollama
```
brew install ollama
```
2) Start Ollama
```
brew services start ollama
```
3) Pull a model
```
ollama pull llama3.1:8b
```

The app calls the local Ollama server via `/summarize`. If Ollama is not running, it falls back to rule-based parsing.

## Supabase setup (sync + login)
1) Create a Supabase project
2) Copy your Project URL and **anon public key** (Legacy anon key, usually starts with `eyJ...`)
3) Update these in:
- `job-tracker/app.js`
- `job-tracker/auth.js`

### Database schema
Run this in Supabase SQL Editor:
```sql
create table if not exists public.job_entries (
  id uuid primary key,
  user_id uuid references auth.users(id),
  created_at timestamptz not null default now(),
  jd_text text,
  company text,
  location text,
  role text,
  work_mode text,
  compensation text,
  company_summary text,
  hiring_for text,
  skills text[],
  tags text[],
  status text,
  deadline date,
  fingerprint text
);

alter table public.job_entries enable row level security;

create policy "users read own" on public.job_entries
  for select using (auth.uid() = user_id);

create policy "users insert own" on public.job_entries
  for insert with check (auth.uid() = user_id);

create policy "users update own" on public.job_entries
  for update using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "users delete own" on public.job_entries
  for delete using (auth.uid() = user_id);
```

## Usage
- Main app: `index.html`
- Login page: `auth.html`
- Sign in to sync across devices
- When you sign in, you will be prompted to upload local entries

## Project structure
```
job-tracker/
  index.html
  style.css
  app.js
  auth.html
  auth.js
  server/
    server.js
    package.json
```

## Notes
- Data is stored in browser `localStorage` and synced to Supabase when logged in.
- The Supabase **service_role** key should never be used in the browser.
