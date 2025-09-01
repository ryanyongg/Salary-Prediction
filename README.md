# National Job Market Analysis & Salary Prediction

## 📌 Project Overview
A government organization engaged a consultancy to analyze the national job market and predict salaries based on survey data.  
This project explores the dataset, identifies insights across industries and job roles, and develops a machine learning model to predict salaries.  
Findings are presented with recommendations tailored for a mid-career professional recently retrenched from the Web industry.  

---

## 🎯 Objectives
1. **Data Cleaning** – Ensure dataset consistency and readiness by handling duplicates, errors, outliers, and missing values.  
2. **Exploratory Data Analysis (EDA)** – Answer key questions on highest/lowest paying roles and industries, benchmark comparisons, and relationships with education, experience, and major.  
3. **Preprocessing & Feature Engineering** – Engineer meaningful features, handle rare categories, and prepare data pipelines for modeling.  
4. **Machine Learning Model Development** – Compare multiple models and identify the best predictive model for salary forecasting.  
5. **Workflow Comparison** – Compare PySpark vs Pandas/sklearn for cleaning and modeling.  
6. **Recommendation** – Provide industry recommendations and career strategies for the retrenched professional.  

---

## 🛠️ Methods

### 1. Data Cleaning
- Removed duplicates (`job_id` integrity).
- Fixed inconsistent categories and datatypes.
- Flagged outliers (salary, experience, distance).
- Imputed missing values (median for numeric, mode for categorical).
- Final dataset: ~1M rows, clean and complete.

### 2. Exploratory Data Analysis
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
- Engineered features: `exp_level`, `near_cbd`, `edu_industry`, `role_exp`.  
- Rare categories bucketed (<10 → “OTHER”).  
- ColumnTransformer applied: scale numerics, OHE categoricals, passthrough binary.  
- Result: 162 features, leakage-safe split (80/20).  

### 4. Machine Learning
- Models tested: Linear/Ridge, Decision Tree, Random Forest, LightGBM, XGBoost.  
- Best model: **Tuned Random Forest** → R² = 0.75, RMSE ≈ 19.4k, MAE ≈ 15.6k.  
- Key drivers: Experience, Role, Education, Location.  
- Consistent across experience levels.  

### 5. PySpark vs Non-PySpark Workflow
- **Data Cleaning**: Pandas faster & concise; PySpark verbose & slower.  
- **Modeling**: sklearn Random Forest (R²=0.75, 75s) vs PySpark GBT/LR (R²=0.73–0.74, 197–316s).  
- Conclusion: sklearn better for this dataset, PySpark better at massive scale.  

---

## 💡 Recommendation

### Industry to Move Into
- **Primary**: Finance (FinTech, Banking, Risk Analytics)  
- **Secondary**: Healthcare / HealthTech  

### Reasons (Beyond Salary)
- **Finance**: High demand for AI/data skills, stable job security, strong digital growth.  
- **Healthcare**: Expanding demand for analytics in medical data & AI diagnostics, socially impactful.  

### Skillsets Needed
- **Finance**: Risk analytics, fraud detection, cloud data engineering, compliance.  
- **Healthcare**: Medical data pipelines, AI for diagnostics, privacy & compliance.  

### How to Obtain Skillsets
- Short certifications & bootcamps (FinTech, CFA Data Science, healthcare analytics).  
- Cloud & Big Data training (AWS, GCP, Spark).  
- Apply existing Web data engineering skills to domain-specific problems.  

### Personal Fit
- Finance offers salary stability to support $2.5k mortgage & $4.2k expenses.  
- Healthcare fits interest in problem-solving and fixing things.  

---

## 📊 Key Findings
- Finance is the **best-paying and most stable industry**.  
- Healthcare is a **strong secondary path** with meaningful impact.  
- Service, Education, and Government are not financially viable.  
- Web remains volatile and risky despite past experience.  

---

## ✅ Conclusion
- Dataset was cleaned, preprocessed, and analyzed thoroughly.  
- Random Forest achieved the best predictive performance (R²=0.75).  
- sklearn is more efficient for this dataset, while PySpark is valuable for scaling.  
- Final recommendation: **Finance (primary) and Healthcare (secondary)** for the retrenched professional.  

---
