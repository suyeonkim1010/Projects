# Data Analyst Job Roles in Canada

This project explores over **1,700 Data Analyst job postings** across Canada (sourced from Indeed and Glassdoor)  
to identify key insights on **regional demand, salary trends, and technical skill requirements**.  

---

## 📊 Project Workflow

### 1. Data Cleaning (`01_cleaning_raw.ipynb`)
- Removed duplicates and missing values  
- Standardized text fields (job titles, skills, provinces)  
- Parsed salary strings into `min_salary`, `max_salary`, and `avg_salary` numeric columns  

### 2. Exploratory Data Analysis (`02_analysis_cleaned.ipynb`)
- Salary distribution and averages by province  
- Job demand by city and region  
- Most frequently required skills (SQL, Python, Excel, Power BI, Tableau)  
- Skill vs Salary comparison (impact of each skill on pay)

### 3. Power BI Dashboard (`03_create_bi_dataset.ipynb` + Power BI)
- Created `clean_for_bi.csv` for visualization  
- Interactive filters by **province** and **skills**  
- KPI cards for total postings and average salary  
- Bar and pie charts showing skill demand and salary delta  
- (Optional) Tableau version available  

---

## 🗂 Folder Structure
```
data/
│── Raw_Dataset.csv
│── Cleaned_Dataset.csv
│── clean_for_bi.csv        # BI-ready dataset
notebooks/
│── 01_cleaning_raw.ipynb
│── 02_analysis_cleaned.ipynb
│── 03_create_bi_dataset.ipynb
outputs/
│── *.png                   # EDA visualizations
│── dashboard_screenshots/  # Power BI screenshots
README.md
03_dashboard_notes.md        # Dashboard design & insights
```

---

## 🧰 Tools & Technologies
- **Python:** pandas, numpy, matplotlib, re  
- **SQL:** SQLite (basic queries and validation)  
- **Power BI / Tableau:** Interactive dashboard and KPI visualization  
- **Kaggle Notebooks:** For reproducibility and public sharing  

---

## 🔍 Key Insights
- Ontario and British Columbia have the highest demand for Data Analysts  
- Average salaries typically range between **70K–90K CAD**  
- **SQL** and **Python** dominate skill requirements, with **Power BI/Tableau** also highly valued  
- Jobs mentioning BI tools or Python tend to show slightly higher average salaries  

---

## 📈 Next Steps
- Expand dataset with new postings (2025 Q1)  
- Include text analysis (word clouds, NER) on job descriptions  
- Compare trends across years using time-based data  

---
