# 📊 Power BI Sales Analysis Dashboard

**Version:** 1.0  
**Date:** [Insert Date]  
**Author:** [Your Name]  

---

## 📌 Project Overview
### 🎯 Purpose  
This Power BI dashboard provides an **interactive sales analysis**, focusing on:  
- **Total Revenue, Total Boxes Shipped, and Average Order Value**  
- **Sales trends by product, country, and time period**  
- **Key influencers affecting revenue growth**  
- **Comparative sales analysis using Treemap, Line Chart, and Scatter Plot**  

The goal is to help businesses make **data-driven decisions** by identifying trends, top-performing products, and sales patterns across different regions.  

---

## 📂 Data Sources
### 📊 Dataset Used  
The dashboard utilizes sales data from [sample-data-10mins.xlsx](./sample-data-10mins.xlsx).  

- **Columns Included:**  
  - `Sales Person`: Representative responsible for the sale  
  - `Country`: The country where the sale was made  
  - `Product`: The type of product sold  
  - `Date`: The date of the transaction  
  - `Amount`: Total revenue generated  
  - `Boxes Shipped`: Number of boxes shipped  


---

## 📊 Dashboard Components & Visualizations  
### **1️⃣ Key Performance Indicators (KPIs)**  
At the top, summarizing key sales metrics:  
- 💰 **Total Revenue** → `SUM(Amount)`  
- 📦 **Total Boxes Shipped** → `SUM(Boxes Shipped)`  
- 📊 **Average Order Value** → `SUM(Amount) / DISTINCTCOUNT(Sales Person)`

### **2️⃣ Filters (Slicers) for Dynamic Analysis**  
- **Product Selector** → Filter by specific products  
- **Date Selector** → Filter by time range (e.g., last 30 days, last quarter)  
- **Country Selector** → Compare sales by region  

### **3️⃣ Treemap - Product Sales Distribution**  
- Displays **product-wise revenue contribution**  
- Identifies **top-selling products**  

### **4️⃣ Bar Chart - Sales by Country**  
- Shows total revenue by country  
- Highlights **top-performing vs. low-performing regions**  

### **5️⃣ Line Chart - Sales Trends Over Time**  
- Tracks **monthly revenue fluctuations**  
- Helps identify **seasonal patterns & growth trends**  

### **6️⃣ Scatter Plot - Sales Comparison (May vs. June)**  
- Compares **product sales performance across two months**  
- Helps spot **consistent vs. declining products**  

### **7️⃣ Key Influencers - AI-driven Insights**  
- Identifies factors **driving sales growth or decline**  
- Uses **AI-based insights** to determine key revenue influences  

---

## ⚙️ Features & Functionality  
### 🔄 **1. Interactive Filtering & Dynamic Updates**  
- Clicking any **product, country, or date** dynamically updates the visuals  
- Sales data is **filtered in real-time**  

### 🎨 **2. Custom Themes & Formatting**  
- Applied **custom color scheme** for better readability  
- Icons added to **KPI cards** for intuitive understanding  

### 📈 **3. Business Insights Derived**  
- Identified **top 3 best-selling products** using Treemap  
- Detected **seasonal sales trends** through Line Chart  
- Analyzed **key revenue-contributing regions** using Bar Chart  

---

## 📊 Key Insights from the Data  

### **1️⃣ Top-Performing Products**  
- **The highest-grossing product:** `Smooth Silky Sweets` with **$350K** in sales.  
- **Other top products:** `50% Dark Bites`, `White Choc`, and `Peanut Butter Bars`, each contributing significantly to total revenue.  
- **Treemap analysis reveals that the top 5 products account for over **40%** of total sales**, indicating a concentration of demand among a few items.  

### **2️⃣ Regional Sales Distribution**  
- **Highest revenue by country:**  
  - 🇦🇺 **Australia** leads with the highest sales, followed closely by 🇬🇧 **UK** and 🇮🇳 **India**.  
  - **USA & Canada** show moderate performance, while **New Zealand** has the lowest sales contribution.  
- **Bar Chart analysis shows that the top 3 countries contribute to nearly 60% of the total sales.**  

### **3️⃣ Monthly Sales Trends & Seasonality**  
- **Line Chart analysis shows fluctuations in sales trends over time.**  
- **Peak sales months:** **January & May**, with significant spikes in revenue.  
- **Sales dip in February & March**, suggesting possible seasonal or external influences.  
- **Overall upward trend** suggests steady business growth with occasional short-term declines.  

### **4️⃣ Customer Buying Patterns & Correlation Between Months**  
- **Scatter plot analysis (May vs. June sales) shows a positive correlation** between months.  
  - Products that performed well in May generally continued strong performance in June.  
  - However, some products like `99% Dark & Pure` saw a **sales decline**, indicating changing customer preferences.  

### **5️⃣ Key Influencers of Sales Performance (AI Insights)**  
- **Power BI's Key Influencers analysis** suggests that:  
  - **"Country" is a strong driver of total sales** – some regions consistently outperform others.  
  - **"Product Type" impacts order value** – premium chocolate varieties generate higher average order values.  
  - **"Number of Boxes Shipped" directly correlates with revenue**, reinforcing the importance of logistics efficiency.  

---

## 📢 Final Summary & Business Recommendations  

1️⃣ **Stock more of the top-selling products** (Smooth Silky Sweets, 50% Dark Bites) to meet customer demand.  
2️⃣ **Focus marketing & distribution efforts on top-performing regions** (Australia, UK, India) while improving sales in weaker regions.  
3️⃣ **Identify reasons behind seasonal dips** in February & March to optimize sales strategies.  
4️⃣ **Explore promotions for underperforming products** to either boost sales or phase out low-demand items.  
5️⃣ **Leverage AI-driven insights** to refine business strategies based on product performance & customer trends.  

---

## 📌 Conclusion  
This Power BI dashboard provides actionable insights into **product performance, regional sales trends, seasonality, and customer preferences.**  
By leveraging these findings, businesses can **optimize inventory, enhance marketing strategies, and maximize revenue potential.**  

📌 **For further analysis or improvements, additional datasets (e.g., customer demographics, purchase frequency) can be integrated.**  

