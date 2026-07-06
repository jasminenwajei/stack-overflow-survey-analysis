# Stack Overflow Developer Income Analysis

A data analytics and machine learning project investigating the factors that influence software developer income using the Stack Overflow Developer Survey dataset.
It follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology to investigate the factors influencing software developer income using the 2024 Stack Overflow Developer Survey.
It demonstrates the complete data analysis lifecycle, from data cleaning and exploratory data analysis through to clustering, predictive modelling and model evaluation.

---

## Project Overview

The aim of this project is to identify the factors associated with higher developer salaries by analysing responses from the Stack Overflow Developer Survey.

Using Python and machine learning techniques, the project explores relationships between developer income and characteristics such as:

- Professional experience
- Education level
- Developer role
- Programming languages
- Country
- Organisation size

The project combines statistical analysis with predictive modelling to generate meaningful business insights from a large real-world dataset.

---

## Dataset

The analysis uses the **Stack Overflow Developer Survey**, containing responses from over 65,000 developers worldwide.

Dataset includes information such as:

- Annual compensation
- Years of professional experience
- Education level
- Developer role
- Programming languages
- Country
- Organisation size

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook / VS Code

---

## Project Workflow

### 1. Exploratory Data Analysis

Performed exploratory analysis to understand the structure and quality of the dataset.

This included:

- Missing value analysis
- Duplicate detection
- Outlier detection
- Distribution analysis
- Correlation exploration
- Feature visualisation

---

### 2. Data Cleaning

Prepared the dataset for modelling by:

- Removing duplicate records
- Handling missing values
- Removing unrealistic outliers
- Applying log transformation to salary
- Median and mode imputation
- Feature engineering

---

### 3. Data Visualisation

Created visualisations to explore relationships between income and:

- Education level
- Professional experience
- Country
- Organisation size
- Developer role
- Programming languages

Visualisations include:

- Histograms
- Box plots
- Scatter plots
- Bar charts
- Distribution plots

---

### 4. Clustering Analysis

Performed unsupervised learning to identify developer groups using:

- Principal Component Analysis (PCA)
- K-Means Clustering
- Hierarchical Clustering
- t-SNE visualisation
- Silhouette analysis
- Elbow method

Developer clusters were analysed to identify common characteristics and career profiles.

---

### 5. Machine Learning

Built several predictive models to classify whether a developer belongs to a high-income group.

Models implemented:

- Logistic Regression
- Random Forest
- XGBoost
- Soft Voting Ensemble

Model performance was evaluated using:

- ROC-AUC
- Cross validation
- Feature importance analysis

---

## Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Statistical Analysis
- Data Visualisation
- Machine Learning
- Classification
- Clustering
- Principal Component Analysis
- Ensemble Learning
- Predictive Analytics

---

## Key Findings
The analysis identified several factors strongly associated with higher developer income:

- Professional coding experience showed a strong positive relationship with annual compensation.
- Developers in countries such as the United States, Switzerland and Australia generally reported higher salaries.
- Data Scientists, DevOps Engineers and Machine Learning Engineers earned higher median salaries than many other developer roles.
- Developers proficient in specialist technologies such as Go and Rust tended to report higher earnings.
- Larger organisations generally offered higher median compensation than smaller companies.

Machine learning models showed that geographical location, years of professional experience and organisation size were among the most important predictors of developer income.

## Example Visualisations
### Income Distribution



## Repository Structure

```
.
├── analysis.py
├── README.md
├── requirements.txt
└── images/
```

---

## Future Improvements

- Hyperparameter optimisation using GridSearchCV
- Interactive dashboards using Power BI or Tableau
- Model deployment using Streamlit
- Automated data pipeline
- Feature selection optimisation
- Additional model comparison
