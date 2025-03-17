# US Layoff Analysis - Data Source & EDR Process

## Introduction
This project analyzes layoffs in the US, specifically focusing on voluntary and involuntary employment separations. Using the **Iowa Executive Branch Employment Separations dataset** from Data.gov, we apply data preprocessing techniques and **SMOTE-based resampling** to address class imbalances.

## Data Source
The dataset contains **employment separation records** since Fiscal Year 2013, updated biweekly. It includes details like:
- **Record Number** (Unique ID)
- **Fiscal Year** (Year of separation)
- **Department Name** (Department from which employee separated)
- **Sub-Unit of Department** (Department subdivision)
- **EEO Category Name** (Job classification by Equal Employment Opportunity standards)
- **Reason** (Cause of separation: layoff, dismissal, retirement, etc.)
- **Employee Status** (Employment type: permanent, temporary)
- **Pay Grade** (Employee's salary grade at the time of separation)
- **Separation Date** (Date of job separation)
- **Pay Period End Date** (End date of last pay period)
- **Current Fiscal Year** (Boolean: Flag for records from the current year)

## Data Preprocessing & EDR Process

### 1. Data Cleaning
- **Removed duplicates** to prevent bias.
- **Converted date columns** (Separation Date, Pay Period End Date) to datetime format.
- **Handled missing values:**
  - **Numerical columns** (e.g., Pay Grade) imputed using the median value.
  - **Categorical columns** (Reason, Employee Status) imputed using the most frequent value.

### 2. Feature Engineering
- Extracted **Separation Year** from the Separation Date.
- Created **Recent Layoff Flag** (Boolean for separations post-2018).
- Categorized **Voluntary vs. Involuntary Separations** based on the Reason column.

### 3. Encoding Categorical Data
- **Label Encoding** applied to Employee Status.
- **One-Hot Encoding** applied to categorical features like EEO Category Name.

### 4. Addressing Class Imbalance using SMOTE
The dataset was **highly imbalanced**, requiring Synthetic Minority Over-sampling Technique (SMOTE):
- Applied **One-Hot Encoding** before resampling.
- Used **k=2 nearest neighbors** for synthetic sample generation.
- Verified class distribution before and after applying SMOTE.

## Processed Data Output
- The cleaned and processed dataset is saved as **processed_layoff_data.csv**.
- The data is now structured, balanced, and suitable for predictive modeling.

## Repository Structure
```
📂 US_Layoff_Analysis
│── 📁 Dataset
│   ├── raw_layoff_data.csv
│   ├── processed_layoff_data.csv
│
│── 📁 Jupyter_Notebook
│   ├── data_preprocessing.ipynb
│
│── 📁 Documentation
│   ├── EDR_summary.docx
│
│── README.md  
```

## Next Steps
- Perform **exploratory data analysis (EDA)** to identify trends.
- Implement **machine learning models** to predict layoffs.
- Develop a **dashboard or application** for interactive analysis.

## Keywords for Literature Review
- **Layoff trends in the US**
- **Employee turnover analysis**
- **Employment separation data analytics**
- **Handling imbalanced data in HR analytics**
- **Machine learning for workforce analytics**

## Contributions
Each team member has contributed to different aspects of **data preprocessing, encoding, and SMOTE application**. See **JIRA Board & GitHub commit history** for detailed individual contributions.

---

