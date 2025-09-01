# National Job Market Analysis & Salary Prediction

## 📌 Project Context
A government organization commissioned a consultancy firm to study the **national job market**.  
The goal was to **understand employment trends, identify salary distributions, and recommend strategies** for citizens navigating career transitions.  

The consultancy collected and analyzed survey data across industries, education levels, majors, and roles.  
Using data science and machine learning, this project provides insights for policymakers **and** practical guidance for individuals.  

---

## 🎯 Assignment Objective
1. Explore the cleaned dataset to answer **specific labour market questions**:  
   - Which jobs and industries pay the most/least?  
   - How do salaries vary by education, experience, and major?  
   - Which roles fall below or above the national median benchmark?  

2. Build and compare **machine learning models** to predict salary from features.  

3. Compare **PySpark vs Non-PySpark workflows** in terms of data cleaning and model training.  

4. Provide **career recommendations** for a case study:  
   - A 30-year-old mid-careerist, retrenched from the Web industry.  
   - Profile: 5 years Web, 1 year Service, 2 years Education; interest in fixing things and board games.  
   - Salary = $88k/year, expenses = $4.2k/month, mortgage = $2.5k/month, married with pets.  

---

## 🛠️ Methods & Approach

### 1. Data Cleaning
- Removed duplicates (`job_id` integrity).
- Fixed inconsistent categories and datatypes.
- Flagged outliers (salary, experience, distance).
- Imputed missing values (median for numeric, mode for categorical).
- Final dataset: ~1M rows, clean and complete.

### 2. Exploratory Data Analysis (EDA)
Answered all assignment questions:
- **Highest-paying Web job**: CEO (147k).  
- **Top 10 roles overall**: CEO, CFO, CTO, VP, Manager, Senior, Junior, President, Janitor.  
- **Highest-paying industry**: Finance (128k).  
- **Lowest-paying job**: Janitor (68k).  
- **Lowest-paying industry**: Government (81k).  
- **Benchmark 114k**: Government = 100% below; roles below = Janitor, President, Junior, Senior, Manager.  
- **Experience vs Salary**: Positive correlation.  
- **Education vs Salary**: Doctoral > Masters > Bachelor > High School/None.  
- **Major vs Salary**: Engineering, Math, and Computer Science highest; Biology, Chemistry, Literature mid-tier.  

### 3. Preprocessing & Feature Engineering
- Features engineered:  
  - `exp_level` (junior, mid, senior)  
  - `near_cbd` (distance <10km)  
  - `edu_industry` (Education × Industry)  
  - `role_exp` (Role × Experience level)  
- Rare categories bucketed (<10 → “OTHER”).  
- ColumnTransformer applied: scale numerics, OHE categoricals, passthrough binary.  
- Result: 162 features, leakage-safe split (80/20).  

### 4. Machine Learning Models
- Baseline models tested: Linear, Ridge, Decision Tree.  
- Ensemble models tested: Random Forest, LightGBM, XGBoost.  
- **Best model**: Tuned Random Forest →  
  - R² = 0.75, RMSE ≈ 19.4k, MAE ≈ 15.6k.  
  - Key drivers: Experience, Role, Education, Location.  
  - Fairness check: consistent performance across experience levels.  

### 5. PySpark vs Non-PySpark Workflow
- **Data Cleaning**: Pandas concise & fast; PySpark verbose & slower.  
- **Model Training**: sklearn Random Forest (R²=0.75, 75s) vs PySpark GBT/LR (R²=0.73–0.74, 197–316s).  
- Conclusion: sklearn better for this dataset; PySpark better for massive clusters.  

---

## 💡 Recommendation for Mid-Career Professional

### Industry to Move Into
- **Primary**: Finance (FinTech, Banking, Risk Analytics)  
- **Secondary**: Healthcare / HealthTech  

### Reasons (Beyond Salary)
- **Finance**: High demand for AI/data engineering skills, stable job security, strong digital growth.  
- **Healthcare**: Expanding use of analytics in medical data & AI diagnostics, socially meaningful.  

### Skillsets Needed
- **Finance**: Risk analytics, fraud detection, cloud data engineering, regulatory compliance.  
- **Healthcare**: Medical data pipelines, AI for diagnostics, privacy & compliance.  

### How to Obtain
- Short certifications & bootcamps (FinTech, CFA Data Science, healthcare analytics).  
- Cloud & Big Data training (AWS, GCP, Spark).  
- Leverage 5 years of Web/data engineering experience into new domains.  

### Personal Fit
- Finance supports his **mortgage and $4.2k expenses** with stable pay.  
- Healthcare aligns with his **interest in fixing things** and problem-solving.  

---

## 📊 Key Findings
- Finance is the **best-paying and most stable industry**.  
- Healthcare is a **strong secondary path** with growing demand and impact.  
- Service, Education, and Government are not financially viable.  
- Web remains volatile and risky despite prior experience.  

---

## ✅ Conclusion
- Dataset cleaned, preprocessed, and analyzed thoroughly.  
- Random Forest was the strongest predictive model (R²=0.75).  
- sklearn outperformed PySpark for this dataset, though Spark scales better.  
- Final recommendation: **Finance as primary, Healthcare as secondary**.  

---

