# Dashboard Notes: Data Analyst Job Roles in Canada

## Purpose
This document describes how the cleaned dataset (`clean_for_bi.csv`) is used to build an interactive dashboard in **Power BI / Tableau**.  
The goal is to highlight **regional job demand, salary distributions, and skill requirements** for Data Analyst roles in Canada.

---

## Key Dashboard Components

### 1. KPI Cards
- **Total Job Postings**: Count of rows in dataset (after cleaning).
- **Average Salary**: Overall mean of `avg_salary`.
- **Top Skills**: Skills with highest frequency (SQL, Python, Excel, Tableau, Power BI).

### 2. Filters (Slicers)
- **Province**: Allows filtering by location.
- **Skill**: Binary columns (SQL, Python, Excel, Tableau, Power BI).
- **Employer / Job Title**: Optional for detailed exploration.

### 3. Visuals
- **Bar Chart – Job Postings by Province**  
  Shows where demand is highest.

- **Bar Chart – Average Salary by Province**  
  Highlights regions offering higher pay.

- **Bar/Pie Chart – Skill Requirement Share**  
  Percentage of postings requiring each skill.

- **Bar Chart – Salary Delta (With vs. Without Skill)**  
  Compares average salary for jobs that mention a skill vs. those that don’t.

- **Histogram – Salary Distribution**  
  Visualizes how salaries are spread (peaks, outliers).

---

## Insights (Draft)
- **Regional Demand**: Provinces like Ontario, British Columbia, and Alberta dominate postings.  
- **Salary Ranges**: Most jobs fall in the 70k–90k CAD range, with some going above 120k.  
- **Skills in Demand**: SQL and Python are most frequent; Tableau/Power BI appear consistently.  
- **Skills vs Salary**: Jobs requiring BI tools (Tableau, Power BI) and programming skills (Python, SQL) show slightly higher average salaries.  

---

## Next Steps
1. **Publish** the dashboard on Tableau Public or Power BI Service.  
2. **Add screenshots** of the main dashboard pages in this repo (`/outputs`).  
3. **Insert link** in the README so recruiters can explore the interactive version.  

---
