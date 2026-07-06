#%% md
# <u><h1>Exploratory data analysis
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("viridis")

survey_data = pd.read_csv("survey_results_public.csv")
schema_data = pd.read_csv("survey_results_schema.csv")
#%%
# features I want to analyse
selected_features = ['ConvertedCompYearly', 'YearsCodePro', 'EdLevel',
                    'DevType', 'LanguageHaveWorkedWith', 'OrgSize', 'Country']

#%%
# Missing values visualization with distinct colors and legend
plt.figure(figsize=(12, 6))
missing_data = survey_data[selected_features].isnull().sum() / len(survey_data) * 100
missing_data = missing_data.sort_values(ascending=False)

# Create a custom colormap with distinct colors
colors = plt.cm.tab20(np.linspace(0, 1, len(missing_data)))

# Create the plot
bars = plt.bar(missing_data.index, missing_data.values, color=colors)

# Add a legend on the side
plt.legend(bars, missing_data.index, title='Features',
           loc='upper left', bbox_to_anchor=(1, 1))

# Customize appearance
plt.title('Percentage of Missing Values by Feature', fontsize=14, fontweight='bold')
plt.ylabel('Missing Values (%)', fontsize=12)
plt.xlabel('Features', fontsize=12)
plt.xticks([])  # Remove x-axis labels since we have the legend
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save with extra space for the legend
plt.savefig('missing_values_colored.png', dpi=300, bbox_inches='tight')
plt.show()
#%%
# Duplicate Analysis

# Find duplicates based on selected features to analyse
potential_duplicates = survey_data.duplicated(subset=selected_features, keep='first')
duplicate_count = potential_duplicates.sum()

# Calculate percentage
duplicate_percentage = (duplicate_count / len(survey_data)) * 100

print(f"Number of potential duplicate submissions: {duplicate_count}")
print(f"Percentage of potential duplicate submissions: {duplicate_percentage:.2f}%")


#%%
# visualization for duplicate analysis results
plt.figure(figsize=(10, 6))

# Data from duplicate analysis
categories = ['Unique Responses', 'Potential Duplicates']
counts = [len(survey_data) - duplicate_count, duplicate_count]
percentages = [100 - duplicate_percentage, duplicate_percentage]

# Create a pie chart
colors = plt.cm.tab20([0, 1])
explode = (0, 0.1)  # Explode the duplicate slice

plt.pie(counts, explode=explode, labels=None, colors=colors,
        autopct='%1.1f%%', shadow=False, startangle=90)

plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
plt.title('Analysis of Potential Duplicate Submissions', fontsize=14, fontweight='bold')

# Add a legend with counts
legend_labels = [f"{cat} ({count:,}, {pct:.1f}%)" for cat, count, pct in zip(categories, counts, percentages)]
plt.legend(legend_labels, title="Response Categories",
          loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))

plt.tight_layout()
plt.savefig('duplicate_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
#%%
# 3. Box Plots for Numerical Features to Show Outliers
plt.figure(figsize=(12, 6))
# Filter out non-null values
comp_data = survey_data['ConvertedCompYearly'].dropna()
sns.boxplot(x=comp_data)
plt.title('Box Plot of Annual Compensation')
plt.xlabel('Annual Compensation')
plt.tight_layout()
plt.savefig('compensation_boxplot.png', dpi=300)
plt.show()

# For YearsCodePro
sns.boxplot(x=comp_data)
plt.figure(figsize=(12, 6))
years_data = survey_data['YearsCodePro'].dropna()
# Convert to numeric if it's not already
if years_data.dtype == 'object':
    # Handle potential non-numeric values
    years_data = pd.to_numeric(years_data, errors='coerce')

sns.boxplot(x=years_data)
plt.title('Box Plot of Years of Professional Coding Experience')
plt.xlabel('Years of Professional Experience')
plt.tight_layout()
plt.savefig('years_experience_boxplot.png', dpi=300)
plt.show()
#%%
# Income distribution transformation visualization
plt.figure(figsize=(14, 6))

# Create subplot structure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Get data with proper column name from your existing code
comp_data = survey_data['ConvertedCompYearly'].dropna()

# Plot original income distribution
sns.histplot(comp_data, ax=ax1, kde=True, color='darkblue')
ax1.set_title('Original Income Distribution', fontweight='bold')
ax1.set_xlabel('Annual Compensation (USD)')
ax1.set_ylabel('Frequency')
ax1.ticklabel_format(style='plain', axis='x')

# Calculate and plot log-transformed income
log_income = np.log1p(comp_data)
sns.histplot(log_income, ax=ax2, kde=True, color='teal')
ax2.set_title('Log-Transformed Income Distribution', fontweight='bold')
ax2.set_xlabel('Log Annual Compensation')
ax2.set_ylabel('Frequency')

plt.suptitle('Effect of Log Transformation on Income Distribution', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('income_transformation.png', dpi=300, bbox_inches='tight')
plt.show()
#%% md
# <u><h1>Data Cleaning:
#%%
# In a cell at the top of your notebook, run:
!pip install numpy pandas matplotlib seaborn scikit-learn xgboost graphviz
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
#%% md
# consistant styling
#%%
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
colors = sns.color_palette("viridis", 10)
#%%
# Load the dataset
survey_data = pd.read_csv("survey_results_public.csv")
schema_data = pd.read_csv("survey_results_schema.csv")
#%%
# Selected features for analysis
selected_features = ['ConvertedCompYearly', 'YearsCodePro', 'EdLevel',
                    'DevType', 'LanguageHaveWorkedWith', 'OrgSize', 'Country']

#%%
# Create a copy to work with
df = survey_data[selected_features].copy()
#%%
# 1. Save original data for comparison later
original_data = df.copy()
#%% md
# <h3>Handling Duplicates
#%%
# Identify and remove duplicates
duplicates = df.duplicated(subset=selected_features, keep='first')
df_no_duplicates = df[~duplicates]
print(f"Removed {duplicates.sum()} duplicate rows")
#%% md
# <h3>Handling outliers for CompensationYearly
#%%
# compute statistics on the non-null values
comp_data = df_no_duplicates['ConvertedCompYearly'].dropna()
comp_mean = comp_data.mean()
comp_std = comp_data.std()
lower_bound = comp_mean - 3 * comp_std
upper_bound = comp_mean + 3 * comp_std
#%%
# Replace outliers with NaN
df_no_outliers = df_no_duplicates.copy()
outlier_mask = (df_no_outliers['ConvertedCompYearly'] < lower_bound) | (df_no_outliers['ConvertedCompYearly'] > upper_bound)
df_no_outliers.loc[outlier_mask, 'ConvertedCompYearly'] = np.nan
print(f"Removed {outlier_mask.sum()} outliers from ConvertedCompYearly")
#%% md
# <h3>Handling outliers for YearsCodePro
#%%
# Convert to numeric
if df_no_outliers['YearsCodePro'].dtype == 'object':
    df_no_outliers['YearsCodePro'] = pd.to_numeric(df_no_outliers['YearsCodePro'], errors='coerce')
#%%
# Remove unrealistic values (over 40 years)
years_outliers = df_no_outliers['YearsCodePro'] > 40
df_no_outliers.loc[years_outliers, 'YearsCodePro'] = np.nan
print(f"Removed {years_outliers.sum()} outliers from YearsCodePro")
#%% md
# <h3>Log transformation for ConvertedCompYearly
#%%
# Create a new column with log-transformed values
df_no_outliers['LogCompensation'] = np.log1p(df_no_outliers['ConvertedCompYearly'])
#%% md
# <h3>Handle missing values
#%% md
# Calculate median compensation for high-income threshold
#%%
median_comp = df_no_outliers['ConvertedCompYearly'].dropna().median()
print(f"Median compensation (high-income threshold): ${median_comp:,.2f}")

#%% md
# For ConvertedCompYearly: Listwise deletion
#%%
df_clean = df_no_outliers.dropna(subset=['ConvertedCompYearly']).copy()
print(f"Kept {len(df_clean)} rows with non-null compensation values")
#%% md
# For YearsCodePro - median imputation
#%%
if df_clean['YearsCodePro'].isnull().any():
    median_imputer = SimpleImputer(strategy='median')
    df_clean['YearsCodePro'] = median_imputer.fit_transform(df_clean[['YearsCodePro']])
    print(f"Imputed {df_clean['YearsCodePro'].isnull().sum()} missing values in YearsCodePro")
#%% md
# For OrgSize: Mode imputation for categorical data
#%%
if df_clean['OrgSize'].isnull().any():
    # For categorical variables
    most_common = df_clean['OrgSize'].mode()[0]
    df_clean['OrgSize'].fillna(most_common, inplace=True)
    print(f"Imputed missing values in OrgSize with most common value: {most_common}")
#%% md
# For other categorical variables: Modal imputation
#%%
categorical_cols = ['Country', 'LanguageHaveWorkedWith', 'EdLevel', 'DevType']
for col in categorical_cols:
    if df_clean[col].isnull().any():
        # Find the mode
        mode_value = df_clean[col].mode()[0]
        # Fill nulls with mode
        df_clean[col].fillna(mode_value, inplace=True)
        print(f"Imputed missing values in {col} with mode: {mode_value}")
#%% md
# <u><h1>Exploratory Data Analysis with Cleaned Data
#%%
cleaned_data = df_clean.copy()
#%% md
# Define high income threshold (median compensation)
#%%
income_median = cleaned_data['ConvertedCompYearly'].median()
cleaned_data['IncomeCategory'] = cleaned_data['ConvertedCompYearly'].apply(
    lambda x: 'High Income' if x > income_median else 'Low Income')
#%% md
# Figure 7: Log-transformed annual compensation distribution histogram
#%%
plt.figure(figsize=(10, 6))
log_income = np.log1p(cleaned_data['ConvertedCompYearly'])
sns.histplot(log_income, kde=True, color=colors[3])

plt.axvline(np.log1p(income_median), color='red', linestyle='--',
            label=f'Median: ${income_median:,.0f}')
plt.title('Distribution of Log-Transformed Annual Compensation', fontsize=14, fontweight='bold')
plt.xlabel('Log Annual Compensation', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('figure7_income_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
#%% md
# Figure 8: Box plot of income by education level
#%%
plt.figure(figsize=(12, 8))
# Sort education levels by median income
ed_level_order = cleaned_data.groupby('EdLevel')['ConvertedCompYearly'].median().sort_values().index

# Create box plot
sns.boxplot(x='ConvertedCompYearly', y='EdLevel', data=cleaned_data,
            order=ed_level_order, palette='viridis')

plt.title('Income Distribution by Education Level', fontsize=14, fontweight='bold')
plt.xlabel('Annual Compensation (USD)', fontsize=12)
plt.ylabel('Education Level', fontsize=12)
plt.xscale('log')  # Log scale for better visualization
plt.axvline(income_median, color='red', linestyle='--',
            label=f'Overall Median: ${income_median:,.0f}')
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('figure8_income_by_education.png', dpi=300, bbox_inches='tight')
plt.show()
#%% md
# Figure 9: Scatter plot of income vs. years of professional experience with trend line
# plt.figure(figsize=(10, 6))
#%%
plt.figure(figsize=(10, 6))

# Convert YearsCodePro to numeric if needed
if cleaned_data['YearsCodePro'].dtype == 'object':
    cleaned_data['YearsCodePro'] = pd.to_numeric(cleaned_data['YearsCodePro'], errors='coerce')

# Create scatter plot with transparency for density visualization
sns.scatterplot(x='YearsCodePro', y='ConvertedCompYearly',
                data=cleaned_data, alpha=0.3, color=colors[5])

# Add smoothed trend line
sns.regplot(x='YearsCodePro', y='ConvertedCompYearly',
            data=cleaned_data, scatter=False,
            order=2,
            line_kws={'color': 'red', 'linewidth': 2})

plt.title('Relationship Between Experience and Income', fontsize=14, fontweight='bold')
plt.xlabel('Years of Professional Coding Experience', fontsize=12)
plt.ylabel('Annual Compensation (USD)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('figure9_income_vs_experience.png', dpi=300, bbox_inches='tight')
plt.show()
#%% md
# Figure 10: Bar chart of median income by country
#%%
# Get top 15 countries by respondent count
top_countries = cleaned_data['Country'].value_counts().nlargest(15).index
country_data = cleaned_data[cleaned_data['Country'].isin(top_countries)]

# Calculate median income by country
country_income = country_data.groupby('Country')['ConvertedCompYearly'].median().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
bars = sns.barplot(x=country_income.values, y=country_income.index, palette='viridis')

# Add median income values to bars
for i, v in enumerate(country_income.values):
    bars.text(v + 5000, i, f'${v:,.0f}', va='center')

plt.title('Median Developer Income by Country', fontsize=14, fontweight='bold')
plt.xlabel('Median Annual Compensation (USD)', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.axvline(income_median, color='red', linestyle='--',
            label=f'Overall Median: ${income_median:,.0f}')
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('figure10_income_by_country.png', dpi=300, bbox_inches='tight')
plt.show()
#%% md
# Figure 11: Bar chart of median income by developer type
#%%
# Developer type might be multiple responses, so we need to split them
def explode_dev_types(df):
    """Split multiple DevType selections into separate rows"""
    # Check if DevType is a string column that needs splitting
    if df['DevType'].dtype == 'object':
        # Create a new dataframe with exploded DevType
        dev_type_df = df.copy()
        # Split the semicolon-separated values and explode
        if dev_type_df['DevType'].str.contains(';').any():
            dev_type_df['DevType'] = dev_type_df['DevType'].str.split(';')
            return dev_type_df.explode('DevType')
    return df

# Explode the developer types
exploded_data = explode_dev_types(cleaned_data)

# Get the most common developer types (top 10)
top_dev_types = exploded_data['DevType'].value_counts().nlargest(10).index
dev_type_data = exploded_data[exploded_data['DevType'].isin(top_dev_types)]

# Calculate median income by developer type
dev_type_income = dev_type_data.groupby('DevType')['ConvertedCompYearly'].median().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
bars = sns.barplot(x=dev_type_income.values, y=dev_type_income.index, palette='viridis')

# Add median income values to bars
for i, v in enumerate(dev_type_income.values):
    bars.text(v + 5000, i, f'${v:,.0f}', va='center')

plt.title('Median Income by Developer Type', fontsize=14, fontweight='bold')
plt.xlabel('Median Annual Compensation (USD)', fontsize=12)
plt.ylabel('Developer Type', fontsize=12)
plt.axvline(income_median, color='red', linestyle='--',
            label=f'Overall Median: ${income_median:,.0f}')
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('figure11_income_by_dev_type.png', dpi=300, bbox_inches='tight')
plt.show()
#%% md
# Figure 12: Box plot of income by organisation size
#%%
plt.figure(figsize=(12, 8))

# Sort organisation sizes by median income
# First create an ordered categorical type to ensure logical ordering
size_categories = [
    "Just me", "2 to 9 employees", "10 to 19 employees",
    "20 to 99 employees", "100 to 499 employees",
    "500 to 999 employees", "1,000 to 4,999 employees",
    "5,000 to 9,999 employees", "10,000 or more employees"
]

# Create new column with ordered categories if your data uses these categories
if 'OrgSize' in cleaned_data.columns:
    # Check if your data uses these exact categories first
    if all(size in cleaned_data['OrgSize'].unique() for size in size_categories):
        cleaned_data['OrgSizeOrdered'] = pd.Categorical(
            cleaned_data['OrgSize'],
            categories=size_categories,
            ordered=True
        )
    else:
        # If categories don't match exactly, sort by median income
        org_size_order = cleaned_data.groupby('OrgSize')['ConvertedCompYearly'].median().sort_values().index
        cleaned_data['OrgSizeOrdered'] = pd.Categorical(
            cleaned_data['OrgSize'],
            categories=org_size_order,
            ordered=True
        )

# Create box plot
sns.boxplot(x='ConvertedCompYearly', y='OrgSizeOrdered', data=cleaned_data, palette='viridis')

plt.title('Income Distribution by Organisation Size', fontsize=14, fontweight='bold')
plt.xlabel('Annual Compensation (USD)', fontsize=12)
plt.ylabel('Organisation Size', fontsize=12)
plt.xscale('log')  # Log scale for better visualization
plt.axvline(income_median, color='red', linestyle='--',
            label=f'Overall Median: ${income_median:,.0f}')
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('figure12_income_by_org_size.png', dpi=300, bbox_inches='tight')
plt.show()


#%% md
# Figure 13: Horizontal bar chart of median income by programming language
#%%
# Languages might be multiple responses, so we need to split them
def explode_languages(df):
    """Split multiple language selections into separate rows"""
    # Check if LanguageHaveWorkedWith is a string column that needs splitting
    if df['LanguageHaveWorkedWith'].dtype == 'object':
        # Create a new dataframe with exploded languages
        lang_df = df.copy()
        # Split the semicolon-separated values and explode
        if lang_df['LanguageHaveWorkedWith'].str.contains(';').any():
            lang_df['LanguageHaveWorkedWith'] = lang_df['LanguageHaveWorkedWith'].str.split(';')
            return lang_df.explode('LanguageHaveWorkedWith')
    return df

# Explode the languages
exploded_lang_data = explode_languages(cleaned_data)

# Get the most common languages (top 15)
top_languages = exploded_lang_data['LanguageHaveWorkedWith'].value_counts().nlargest(15).index
lang_data = exploded_lang_data[exploded_lang_data['LanguageHaveWorkedWith'].isin(top_languages)]

# Calculate median income by language
lang_income = lang_data.groupby('LanguageHaveWorkedWith')['ConvertedCompYearly'].median().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
bars = sns.barplot(x=lang_income.values, y=lang_income.index, palette='viridis')

# Add median income values to bars
for i, v in enumerate(lang_income.values):
    bars.text(v + 5000, i, f'${v:,.0f}', va='center')

plt.title('Median Income by Programming Language', fontsize=14, fontweight='bold')
plt.xlabel('Median Annual Compensation (USD)', fontsize=12)
plt.ylabel('Programming Language', fontsize=12)
plt.axvline(income_median, color='red', linestyle='--',
            label=f'Overall Median: ${income_median:,.0f}')
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('figure13_income_by_language.png', dpi=300, bbox_inches='tight')
plt.show()
#%% md
# Figure 14: PCA plot colored by income category
#%%
# 2.5.8 Multivariate Analysis - Experience Distribution by Income Category
plt.figure(figsize=(10, 6))

# Create a boxplot comparing years of experience across income categories
sns.boxplot(x='IncomeCategory', y='YearsCodePro', data=viz_data, palette={'High Income': '#1f77b4', 'Low Income': '#ff7f0e'})

# Add individual points for more detailed visualization
sns.stripplot(x='IncomeCategory', y='YearsCodePro', data=viz_data,
              color='black', alpha=0.2, size=3, jitter=True)

# Add descriptive statistics annotations
for i, category in enumerate(['Low Income', 'High Income']):
    category_data = viz_data[viz_data['IncomeCategory'] == category]['YearsCodePro']
    median_val = category_data.median()
    mean_val = category_data.mean()

    # Annotate with median and mean
    plt.annotate(f'Median: {median_val:.1f}\nMean: {mean_val:.1f}',
                xy=(i, median_val + 1),
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

# Customize the plot appearance
plt.title('Distribution of Professional Experience by Income Category', fontsize=14, fontweight='bold')
plt.xlabel('Income Category', fontsize=12)
plt.ylabel('Years of Professional Coding Experience', fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the figure
plt.savefig('figure14_experience_by_income.png', dpi=300, bbox_inches='tight')
plt.show()
#%% md
# <u><h1>Cluster Analysis
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.cm as cm
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
#%%
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
colors = sns.color_palette("viridis", 10)
#%% md
# <h2>3.1 Data Transformation and Normalization
#%%
cluster_data = cleaned_data.copy()
#%% md
#  Handle categorical variables with one-hot encoding
#%%
def prepare_multi_response(df, column_name, delimiter=';'):
    """
    Converts multi-response columns (like languages or developer types) into binary columns
    """
    # Skip if column doesn't exist
    if column_name not in df.columns:
        return df

    # Ensure the column is string type
    df[column_name] = df[column_name].astype(str)

    # Get all unique values by splitting and flattening
    all_values = []
    for item in df[column_name].dropna():
        all_values.extend(item.split(delimiter))
    unique_values = sorted(list(set(all_values)))

    # Create binary columns for each value
    for value in unique_values:
        value_clean = value.strip()
        col_name = f"{column_name}_{value_clean.replace(' ', '_')}"
        df[col_name] = df[column_name].apply(
            lambda x: 1 if pd.notna(x) and value_clean in x.split(delimiter) else 0
        )

    return df
#%% md
# Apply to DevType and LanguageHaveWorkedWith
#%%
cluster_data = prepare_multi_response(cluster_data, 'DevType')
cluster_data = prepare_multi_response(cluster_data, 'LanguageHaveWorkedWith')
#%% md
# One-hot encode remaining categorical variables
#%%
categorical_cols = ['EdLevel', 'OrgSize', 'Country']
for col in categorical_cols:
    if col in cluster_data.columns:
        dummies = pd.get_dummies(cluster_data[col], prefix=col, drop_first=False)
        cluster_data = pd.concat([cluster_data, dummies], axis=1)
#%% md
# Drop original categorical columns and non-features
#%%
drop_cols = ['ConvertedCompYearly', 'LogCompensation', 'IncomeCategory', 'DevType',
             'LanguageHaveWorkedWith', 'EdLevel', 'OrgSize', 'Country']
feature_cols = [col for col in cluster_data.columns if col not in drop_cols]

# Keep YearsCodePro as a numerical feature
X_features = cluster_data[feature_cols].copy()
#%%
# Handle the categorical column 'OrgSizeOrdered'
if 'OrgSizeOrdered' in X_features.columns:
    # Convert the categorical column to numeric codes
    X_features['OrgSizeOrdered'] = X_features['OrgSizeOrdered'].cat.codes

# Convert all columns to numeric type
X_features_numeric = X_features.astype(float)

# scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features_numeric)
#%%
# Convert any categorical columns to numeric
for col in X_features.columns:
    if pd.api.types.is_categorical_dtype(X_features[col]):
        X_features[col] = X_features[col].cat.codes

# Make sure everything is numeric
X_features_numeric = X_features.astype(float)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features_numeric)
#%%
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)
#%% md
# Apply PCA for dimensionality reduction
#%%
pca = PCA()
pca.fit(X_scaled)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Find number of components for 85% variance
n_components = np.argmax(cumulative_variance >= 0.85) + 1
print(f"Number of PCA components for 85% variance: {n_components}")
#%% md
# Apply PCA with the determined number of components
#%%
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
print(f"Reduced dimensions from {X_scaled.shape[1]} to {X_pca.shape[1]}")
#%% md
# <h2>3.2 Clustering Methods
#%% md
# <h3>3.2.1 K-Means Clustering - Finding optimal number of clusters
#%%
# Elbow Method to find the optimal number of clusters
inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    # K-means with k clusters
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_pca)

    # Calculate inertia (for elbow method)
    inertia.append(kmeans.inertia_)

    # Calculate silhouette score
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_pca, labels))

    print(f"K={k}, Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_score(X_pca, labels):.3f}")

#%% md
# Figure 15: Elbow Method for K-Means
#%%
plt.figure(figsize=(12, 6))

# Plot the elbow curve
plt.plot(K_range, inertia, marker='o', linestyle='-', color=colors[3], linewidth=2, markersize=8)

# Add vertical line at k=4 (our chosen number)
plt.axvline(x=4, color='red', linestyle='--', label='Chosen k=4')

# Annotate the elbow point
plt.annotate('Elbow Point', xy=(4, inertia[2]),
             xytext=(4.5, inertia[2] + (max(inertia) - min(inertia))/5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12)

# Customize the plot
plt.title('Elbow Method for Optimal k in K-Means Clustering', fontsize=14, fontweight='bold')
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.xticks(K_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('figure15_kmeans_elbow.png', dpi=300, bbox_inches='tight')
plt.show()
#%% md
# Figure 16: Silhouette Analysis
#%%
plt.figure(figsize=(12, 6))

# Plot silhouette scores
plt.plot(K_range, silhouette_scores, marker='o', linestyle='-', color=colors[5], linewidth=2, markersize=8)

# Add vertical line at k=4
plt.axvline(x=4, color='red', linestyle='--', label='Chosen k=4')

# Annotate the maximum silhouette score
best_k = K_range[np.argmax(silhouette_scores)]
best_score = max(silhouette_scores)
plt.annotate(f'Highest Score: {best_score:.3f}',
             xy=(best_k, best_score),
             xytext=(best_k + 0.5, best_score - 0.02),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12)

# Customize the plot
plt.title('Silhouette Scores for Different Numbers of Clusters', fontsize=14, fontweight='bold')
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.xticks(K_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('figure16_silhouette_scores.png', dpi=300, bbox_inches='tight')
plt.show()
#%% md
#  K-Means with the optimal number of clusters (k=4)
#%%
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X_pca)

# Add cluster labels to the original data
cleaned_data['KMeans_Cluster'] = kmeans_labels
#%% md
# <h3>3.2.2 Hierarchical Clustering
#%%
# Hierarchical Clustering with Ward's method
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_pca)

# Add hierarchical cluster labels to the original data
cleaned_data['Hierarchical_Cluster'] = hierarchical_labels
#%% md
# Figure 17: Create dendrogram for hierarchical clustering
#%%
# Generate the linkage matrix
# Create a random sample for the dendrogram visualization
np.random.seed(42)
sample_size = min(1000, X_pca.shape[0])  # Sample 1000 points or less if dataset is smaller
sample_indices = np.random.choice(X_pca.shape[0], sample_size, replace=False)
X_sample = X_pca[sample_indices]

# Now generate the linkage matrix
Z = linkage(X_sample, method='ward')

# Create a more interpretable dendrogram
plt.figure(figsize=(14, 8))
plt.title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
plt.xlabel('Number of samples in cluster', fontsize=12)
plt.ylabel('Distance', fontsize=12)

# Plot dendrogram with truncation to make it more readable
dendrogram(
    Z,
    truncate_mode='lastp',  # Show only the last p merged clusters
    p=25,                   # Show only the last 25 merged clusters
    leaf_rotation=90.,      # Rotates the x axis labels
    leaf_font_size=10.,     # Font size for the x axis labels
    show_contracted=True,   # Show contracted nodes as a triangle
    color_threshold=Z[-(optimal_k-1), 2]  # Color threshold for the 4 clusters
)

# Draw horizontal line for 4 clusters cut
plt.axhline(y=Z[-(optimal_k-1), 2], color='red', linestyle='--',
            label=f'Cut for {optimal_k} clusters')
plt.legend()
plt.tight_layout()
plt.savefig('figure17_dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()
#%% md
# <h2>3.3 Cluster Evaluation and Interpretation
#%%
# Calculate Adjusted Rand Index to compare clustering methods
ari_score = adjusted_rand_score(kmeans_labels, hierarchical_labels)
print(f"Adjusted Rand Index between K-Means and Hierarchical clustering: {ari_score:.3f}")
#%% md
# Figure 18: t-SNE visualization of clusters
#%%
# Apply t-SNE for visualization in 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(X_pca)

# Create a DataFrame for visualization
tsne_df = pd.DataFrame({
    'x': X_tsne[:, 0],
    'y': X_tsne[:, 1],
    'cluster': kmeans_labels,
    'income_category': cleaned_data['IncomeCategory'].values
})

# Plot t-SNE with clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    tsne_df['x'], tsne_df['y'],
    c=tsne_df['cluster'],
    cmap='viridis',
    alpha=0.7,
    s=40,
    edgecolors='w',
    linewidth=0.5
)

# Add legend and customize the plot
plt.colorbar(scatter, label='Cluster')
plt.title('t-SNE Visualization of Developer Clusters', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('figure18_tsne_clusters.png', dpi=300, bbox_inches='tight')
plt.show()
#%% md
# <h2>3.4 Cluster Characteristics
# 
#%%
# cluster labels back to cleaned data for analysis
cleaned_data_with_clusters = cleaned_data.copy()
#%%
# Calculate key statistics for each cluster
cluster_stats = cleaned_data_with_clusters.groupby('KMeans_Cluster').agg({
    'YearsCodePro': 'mean',
    'ConvertedCompYearly': 'median',
    'IncomeCategory': lambda x: (x == 'High Income').mean() * 100  # Percentage of high income
}).reset_index()

# Rename columns for clarity
cluster_stats = cluster_stats.rename(columns={
    'YearsCodePro': 'Avg_Experience_Years',
    'ConvertedCompYearly': 'Median_Income',
    'IncomeCategory': 'High_Income_Percentage'
})

print("Cluster Statistics:")
print(cluster_stats)
#%%
# Create a more comprehensive cluster profile
def analyze_cluster_profiles(df, cluster_column='KMeans_Cluster'):
    """Generate detailed profiles for each cluster"""
    profiles = []

    # For each cluster
    for cluster_id in sorted(df[cluster_column].unique()):
        # Filter data for this cluster
        cluster_data = df[df[cluster_column] == cluster_id]
        cluster_size = len(cluster_data)
        cluster_percentage = cluster_size / len(df) * 100

        # Calculate key metrics
        avg_experience = cluster_data['YearsCodePro'].mean()
        median_income = cluster_data['ConvertedCompYearly'].median()
        high_income_pct = (cluster_data['IncomeCategory'] == 'High Income').mean() * 100

        # Education distribution
        if 'EdLevel' in df.columns:
            top_education = cluster_data['EdLevel'].value_counts().index[0]
        else:
            top_education = "Unknown"

        # Company size distribution
        if 'OrgSize' in df.columns:
            top_org_size = cluster_data['OrgSize'].value_counts().index[0]
        else:
            top_org_size = "Unknown"

        # Most common countries
        if 'Country' in df.columns:
            top_countries = cluster_data['Country'].value_counts().nlargest(3).index.tolist()
        else:
            top_countries = ["Unknown"]

        # Most common developer roles (for multi-response column)
        if 'DevType' in df.columns:
            if any(cluster_data['DevType'].str.contains(';', na=False)):
                # Split multi-response and count
                dev_types = []
                for types in cluster_data['DevType'].dropna():
                    dev_types.extend([t.strip() for t in types.split(';')])
                top_roles = pd.Series(dev_types).value_counts().nlargest(3).index.tolist()
            else:
                top_roles = cluster_data['DevType'].value_counts().nlargest(3).index.tolist()
        else:
            top_roles = ["Unknown"]

        # Build the profile dictionary
        profile = {
            'Cluster_ID': cluster_id,
            'Size': cluster_size,
            'Percentage': cluster_percentage,
            'Avg_Experience': avg_experience,
            'Median_Income': median_income,
            'High_Income_Percentage': high_income_pct,
            'Top_Education': top_education,
            'Top_Org_Size': top_org_size,
            'Top_Countries': top_countries,
            'Top_Roles': top_roles
        }

        profiles.append(profile)

    return pd.DataFrame(profiles)

# Generate profiles
cluster_profiles = analyze_cluster_profiles(cleaned_data_with_clusters)
print("\nDetailed Cluster Profiles:")
print(cluster_profiles)
#%% md
# Visualize cluster profiles - Experience vs. Income with High Income Percentage
#%%
plt.figure(figsize=(12, 8))

# Create scatter plot of clusters
scatter = plt.scatter(
    cluster_profiles['Avg_Experience'],
    cluster_profiles['Median_Income'],
    s=cluster_profiles['Percentage'] * 20,  # Size based on cluster percentage
    c=cluster_profiles['High_Income_Percentage'],  # Color based on high income percentage
    cmap='viridis',
    alpha=0.7,
    edgecolors='black'
)

# Add cluster labels
for i, profile in cluster_profiles.iterrows():
    plt.annotate(
        f"Cluster {int(profile['Cluster_ID'])}",
        xy=(profile['Avg_Experience'], profile['Median_Income']),
        xytext=(7, 0),
        textcoords='offset points',
        fontsize=12,
        fontweight='bold'
    )

# Customize the plot
plt.colorbar(scatter, label='High Income Percentage')
plt.title('Cluster Profiles: Experience vs. Income', fontsize=14, fontweight='bold')
plt.xlabel('Average Years of Professional Experience', fontsize=12)
plt.ylabel('Median Annual Income (USD)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('cluster_profiles_summary.png', dpi=300, bbox_inches='tight')
plt.show()
#%% md
# Create descriptive cluster names based on profiles
#%%
cluster_names = {
    0: "Early-Career Generalists",
    1: "Mid-Career Specialists",
    2: "Senior Technical Leaders",
    3: "Emerging Technology Specialists"
}

# Map names to the DataFrame
cleaned_data_with_clusters['Cluster_Name'] = cleaned_data_with_clusters['KMeans_Cluster'].map(cluster_names)

# Print percentage of each named cluster
cluster_distribution = cleaned_data_with_clusters['Cluster_Name'].value_counts(normalize=True) * 100
print("\nCluster Distribution:")
print(cluster_distribution)

# Save the updated DataFrame for use in classification models
cleaned_data_with_clusters.to_csv('cleaned_data_with_clusters.csv', index=False)
print("\nData with cluster assignments saved to 'cleaned_data_with_clusters.csv'")
#%% md
# <u><h1>4. Machine Learning for Classification and their Implementation
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import graphviz
from sklearn.tree import export_graphviz
import matplotlib.cm as cm
import joblib
import warnings
warnings.filterwarnings('ignore')
#%%
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
colors = sns.color_palette("viridis", 10)
#%%
# Load preprocessed data
print("Loading and preparing data...")
try:
    # Attempt to load already processed data if available
    data = pd.read_csv("cleaned_data_with_clusters.csv")
    print("Loaded preprocessed data successfully")
except:
    print("Could not load preprocessed data, please run data preprocessing first")
    exit()
#%%
# Quick data preparation for classification
print("Preparing data for classification...")
# Create target variable (high income = above median)
median_income = data['ConvertedCompYearly'].median()
data['HighIncome'] = (data['ConvertedCompYearly'] > median_income).astype(int)
print(f"Median income threshold: ${median_income:,.2f}")

#%%
features = []
#%%
numerical_features = ['YearsCodePro']
features.extend(numerical_features)
#%%
#  key categorical features
# Country - keep only top countries by count
top_countries = data['Country'].value_counts().nlargest(10).index
for country in top_countries:
    col_name = f'Country_{country}'
    data[col_name] = (data['Country'] == country).astype(int)
    features.append(col_name)

# Education level
if 'EdLevel' in data.columns:
    top_ed_levels = data['EdLevel'].value_counts().nlargest(5).index
    for level in top_ed_levels:
        col_name = f'EdLevel_{level}'
        data[col_name] = (data['EdLevel'] == level).astype(int)
        features.append(col_name)

# Organization size
if 'OrgSize' in data.columns:
    top_org_sizes = data['OrgSize'].value_counts().nlargest(5).index
    for size in top_org_sizes:
        col_name = f'OrgSize_{size}'
        data[col_name] = (data['OrgSize'] == size).astype(int)
        features.append(col_name)

# cluster assignments
if 'KMeans_Cluster' in data.columns:
    for cluster in range(4):  # Assuming 4 clusters from previous analysis
        col_name = f'Cluster_{cluster}'
        data[col_name] = (data['KMeans_Cluster'] == cluster).astype(int)
        features.append(col_name)

# DevType
if 'DevType' in data.columns:
    # Handle multi-response
    all_dev_types = []
    for types in data['DevType'].dropna():
        if isinstance(types, str) and ';' in types:
            all_dev_types.extend([t.strip() for t in types.split(';')])
        else:
            all_dev_types.append(str(types).strip())

    # Get top 5 developer types
    top_dev_types = pd.Series(all_dev_types).value_counts().nlargest(5).index

    for dev_type in top_dev_types:
        col_name = f'DevType_{dev_type}'
        if ';' in data['DevType'].iloc[0]:
            # Multi-response column
            data[col_name] = data['DevType'].apply(
                lambda x: 1 if isinstance(x, str) and dev_type in x.split(';') else 0
            )
        else:
            # Single response column
            data[col_name] = (data['DevType'] == dev_type).astype(int)
        features.append(col_name)

# Languages
if 'LanguageHaveWorkedWith' in data.columns:
    # Handle multi-response
    all_languages = []
    for langs in data['LanguageHaveWorkedWith'].dropna():
        if isinstance(langs, str) and ';' in langs:
            all_languages.extend([l.strip() for l in langs.split(';')])
        else:
            all_languages.append(str(langs).strip())

    # Get top 5 languages
    top_languages = pd.Series(all_languages).value_counts().nlargest(5).index

    for language in top_languages:
        col_name = f'Language_{language}'
        if ';' in data['LanguageHaveWorkedWith'].iloc[0]:
            # Multi-response column
            data[col_name] = data['LanguageHaveWorkedWith'].apply(
                lambda x: 1 if isinstance(x, str) and language in x.split(';') else 0
            )
        else:
            # Single response column
            data[col_name] = (data['LanguageHaveWorkedWith'] == language).astype(int)
        features.append(col_name)

print(f"Selected {len(features)} features for classification")
#%%
# Prepare data for modeling
X = data[features]
y = data['HighIncome']
#%%
# Standardize numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])
#%%
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

#%% md
# model training functions
#%%
def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train logistic regression with minimal hyperparameter tuning"""
    print("\nTraining Logistic Regression...")

    # Pre-selected parameters based on common good values
    # No grid search to save time
    log_reg = LogisticRegression(
        C=0.1,  # Stronger regularization
        class_weight='balanced',
        solver='saga',  # Good for imbalanced data
        max_iter=1000,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )

    # Fit the model
    log_reg.fit(X_train, y_train)

    # Quick validation
    val_pred_proba = log_reg.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred_proba)

    print(f"Logistic Regression validation AUC: {val_auc:.4f}")
    return log_reg

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train random forest with minimal hyperparameter tuning"""
    print("\nTraining Random Forest...")

    # Pre-selected parameters based on common good values
    # No grid search to save time
    rf_classifier = RandomForestClassifier(
        n_estimators=100,  # Fewer trees for speed
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )

    # Fit the model
    rf_classifier.fit(X_train, y_train)

    # Quick validation
    val_pred_proba = rf_classifier.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred_proba)

    print(f"Random Forest validation AUC: {val_auc:.4f}")
    return rf_classifier

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with minimal hyperparameter tuning"""
    print("\nTraining XGBoost...")

    # Pre-selected parameters
    xgb_classifier = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=3,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        use_label_encoder=False
    )

    # Fit the model without early_stopping_rounds
    xgb_classifier.fit(X_train, y_train)

    # Quick validation
    val_pred_proba = xgb_classifier.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred_proba)

    print(f"XGBoost validation AUC: {val_auc:.4f}")
    return xgb_classifier
#%% md
# Train the models
#%%
print("\n=== Training Machine Learning Models ===")
log_reg = train_logistic_regression(X_train, y_train, X_val, y_val)
rf_classifier = train_random_forest(X_train, y_train, X_val, y_val)
xgb_classifier = train_xgboost(X_train, y_train, X_val, y_val)

# Create and train ensemble models
print("\n=== Creating Ensemble Models ===")

#%% md
# Voting Classifier
#%%
print("Training Voting Classifier...")
voting_clf = VotingClassifier(
    estimators=[
        ('logistic', log_reg),
        ('random_forest', rf_classifier),
        ('xgboost', xgb_classifier)
    ],
    voting='soft',  # Use probabilities
    weights=[1, 2, 2]  # Weight models based on performance
)
voting_clf.fit(X_train, y_train)
#%%
# Evaluate on validation set
val_pred_proba = voting_clf.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred_proba)
print(f"Voting Classifier validation AUC: {val_auc:.4f}")
#%% md
# classification workflow diagram
#%%
def create_classification_workflow_diagram():
    """Create a diagram illustrating the machine learning classification workflow"""
    plt.figure(figsize=(12, 8))

    # Box sizes and positions
    box_width = 0.3
    box_height = 0.1

    # Define colors
    data_color = '#D4F1F9'  # Light blue
    preproc_color = '#B3E5FC'  # Slightly darker blue
    model_color = '#81D4FA'  # Medium blue
    eval_color = '#4FC3F7'  # Darker blue

    # Data preparation boxes
    plt.fill_between([0.1, 0.9], [0.85, 0.85], [1.0, 1.0], color=data_color, alpha=0.6)
    plt.text(0.5, 0.95, 'Data Preparation', ha='center', va='center', fontsize=14, fontweight='bold')
    plt.text(0.5, 0.9, 'Feature Selection, Data Splitting, Feature Scaling', ha='center', va='center', fontsize=12)

    # Processing arrows
    plt.arrow(0.5, 0.85, 0, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')

    # Model Training boxes
    plt.fill_between([0.1, 0.9], [0.55, 0.55], [0.8, 0.8], color=preproc_color, alpha=0.6)
    plt.text(0.5, 0.75, 'Model Training', ha='center', va='center', fontsize=14, fontweight='bold')

    # Individual models
    plt.fill_between([0.15, 0.35], [0.65, 0.65], [0.7, 0.7], color=model_color, alpha=0.8)
    plt.text(0.25, 0.675, 'Logistic Regression', ha='center', va='center', fontsize=10)

    plt.fill_between([0.4, 0.6], [0.65, 0.65], [0.7, 0.7], color=model_color, alpha=0.8)
    plt.text(0.5, 0.675, 'Random Forest', ha='center', va='center', fontsize=10)

    plt.fill_between([0.65, 0.85], [0.65, 0.65], [0.7, 0.7], color=model_color, alpha=0.8)
    plt.text(0.75, 0.675, 'XGBoost', ha='center', va='center', fontsize=10)

    # Parameter selection
    plt.text(0.5, 0.625, 'Model Parameters & Optimization', ha='center', va='center', fontsize=12)

    # Processing arrows
    plt.arrow(0.5, 0.55, 0, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')

    # Ensemble Learning boxes
    plt.fill_between([0.1, 0.9], [0.3, 0.3], [0.5, 0.5], color=model_color, alpha=0.8)
    plt.text(0.5, 0.45, 'Ensemble Learning', ha='center', va='center', fontsize=14, fontweight='bold')

    plt.fill_between([0.3, 0.7], [0.35, 0.35], [0.4, 0.4], color=eval_color, alpha=0.8)
    plt.text(0.5, 0.375, 'Voting Classifier', ha='center', va='center', fontsize=10)

    # Processing arrows
    plt.arrow(0.5, 0.3, 0, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')

    # Evaluation boxes
    plt.fill_between([0.1, 0.9], [0.05, 0.05], [0.25, 0.25], color=eval_color, alpha=0.6)
    plt.text(0.5, 0.2, 'Model Evaluation', ha='center', va='center', fontsize=14, fontweight='bold')
    plt.text(0.5, 0.15, 'Accuracy, Precision, Recall, F1-Score, ROC-AUC', ha='center', va='center', fontsize=12)
    plt.text(0.5, 0.1, 'Final Model Selection', ha='center', va='center', fontsize=12)

    # Remove axes and set title
    plt.axis('off')
    plt.title('Machine Learning Classification Workflow', fontsize=16, fontweight='bold', pad=20)

    # Save the figure
    plt.tight_layout()
    plt.savefig('classification_workflow_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
#%% md
# Feature importance analysis
#%%
def analyze_feature_importance():
    """Analyze and visualize feature importance from different models"""
    print("\n=== Feature Importance Analysis ===")

    # Get feature importance from each model
    importances = {}

    # From logistic regression
    if hasattr(log_reg, 'coef_'):
        importances['Logistic Regression'] = np.abs(log_reg.coef_[0])

    # From random forest
    if hasattr(rf_classifier, 'feature_importances_'):
        importances['Random Forest'] = rf_classifier.feature_importances_

    # From XGBoost
    if hasattr(xgb_classifier, 'feature_importances_'):
        importances['XGBoost'] = xgb_classifier.feature_importances_

    # Create DataFrame with all importance values
    importance_df = pd.DataFrame({name: imp for name, imp in importances.items()})
    importance_df['Feature'] = features

    # Normalize importance values
    for name in importances.keys():
        importance_df[name] = importance_df[name] / importance_df[name].sum()

    # Calculate average importance and sort
    importance_df['Average'] = importance_df[[name for name in importances.keys()]].mean(axis=1)
    importance_df = importance_df.sort_values('Average', ascending=False)

    # Print top 10 features
    print("Top 10 Most Important Features:")
    print(importance_df[['Feature', 'Average']].head(10))

    return importance_df
#%%
# Save models for later evaluation
def save_models():
    """Save trained models for later use"""
    print("\n=== Saving Models ===")
    joblib.dump(log_reg, 'logistic_regression_model.pkl')
    joblib.dump(rf_classifier, 'random_forest_model.pkl')
    joblib.dump(xgb_classifier, 'xgboost_model.pkl')
    joblib.dump(voting_clf, 'voting_classifier_model.pkl')
    print("All models saved successfully")

#%%
# Execute the workflow
if __name__ == "__main__":
    # Feature importance analysis
    importance_df = analyze_feature_importance()

    # Create workflow diagram
    create_classification_workflow_diagram()

    # Save models
    save_models()

    print("\n=== Classification Model Training Complete ===")
    print(f"Models trained on {X_train.shape[0]} samples with {len(features)} features")
    print("Ready for evaluation (Section 5)")
#%% md
# <h1> logistic regression
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import validation_curve
import graphviz
from sklearn.tree import export_graphviz
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')
#%%
# Set consistent styling
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
colors = sns.color_palette("viridis", 10)
#%%
def plot_logistic_regression_coefficients(log_reg_model, feature_names, top_n=15):
    """
    Plot the top coefficients from a logistic regression model
    """
    # Get coefficients from the model
    coef = log_reg_model.coef_[0]

    # Create DataFrame with feature names and coefficients
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef
    })

    # Sort by absolute value of coefficient
    coef_df['AbsCoef'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('AbsCoef', ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(
        y=np.arange(len(coef_df)),
        width=coef_df['Coefficient'],
        color=[colors[1] if c > 0 else colors[6] for c in coef_df['Coefficient']]
    )

    # Add feature names as y-tick labels
    plt.yticks(np.arange(len(coef_df)), coef_df['Feature'])

    # Add a vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # Add value labels to the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x_pos = width + 0.01 if width > 0 else width - 0.01
        ha = 'left' if width > 0 else 'right'
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}', ha=ha, va='center')

    # Add titles and labels
    plt.title('Top Coefficients in Logistic Regression Model', fontsize=14, fontweight='bold')
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('logistic_regression_coefficients.png', dpi=300, bbox_inches='tight')
    plt.show()
#%%
def plot_logistic_regression_decision_boundary(X_train, y_train, log_reg_model):
    """
    Plot the decision boundary of a logistic regression model using PCA to reduce to 2D
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Reduce to 2 dimensions using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Train a new logistic regression model on the PCA-transformed data
    log_reg_pca = LogisticRegression(C=log_reg_model.C, solver=log_reg_model.solver,
                                     max_iter=1000, random_state=42)
    log_reg_pca.fit(X_pca, y_train)

    # Create a meshgrid to plot the decision boundary
    h = 0.02  # Step size in the mesh
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict probabilities for each point in the meshgrid
    Z = log_reg_pca.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Create plot
    plt.figure(figsize=(12, 10))

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, cmap=cm.viridis, alpha=0.8)

    # Plot the training points
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train,
               cmap=cm.RdBu, edgecolors='k', s=80, alpha=0.7)

    # Add legend, title and labels
    plt.legend(*scatter.legend_elements(), title="Income Class")
    plt.title('Logistic Regression Decision Boundary (PCA-reduced features)',
              fontsize=14, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('logistic_regression_decision_boundary.png', dpi=300, bbox_inches='tight')
    plt.show()
#%%
def plot_regularization_path(X_train, y_train, X_val, y_val):
    """
    Plot the impact of different regularization strengths on model performance
    """
    # Range of regularization strengths
    C_values = np.logspace(-3, 3, 15)

    # Lists to store validation scores
    auc_scores = []

    # Train models with different C values and evaluate
    for C in C_values:
        log_reg = LogisticRegression(C=C, solver='liblinear', max_iter=1000, random_state=42)
        log_reg.fit(X_train, y_train)

        # Get predictions and calculate AUC
        y_val_pred_proba = log_reg.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_pred_proba)
        auc_scores.append(auc)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(C_values, auc_scores, marker='o', linestyle='-', color=colors[3], linewidth=2)

    # Find best C value and highlight it
    best_C_idx = np.argmax(auc_scores)
    best_C = C_values[best_C_idx]
    best_auc = auc_scores[best_C_idx]

    # Highlight best point
    plt.plot(best_C, best_auc, 'ro', markersize=10)
    plt.annotate(f'Best C: {best_C:.3f}\nAUC: {best_auc:.3f}',
                xy=(best_C, best_auc), xytext=(best_C*5, best_auc-0.02),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    # Add titles and labels
    plt.title('Logistic Regression Regularization Path', fontsize=14, fontweight='bold')
    plt.xlabel('Regularization Parameter (C)', fontsize=12)
    plt.ylabel('ROC AUC Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('logistic_regression_regularization_path.png', dpi=300, bbox_inches='tight')
    plt.show()
#%% md
# <h1>Random Forest Visulisations
#%%
def plot_random_forest_feature_importance(rf_model, feature_names, top_n=15):

    # Get feature importances
    importances = rf_model.feature_importances_

    # Create DataFrame with feature names and importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort by importance and get top N
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(
        y=np.arange(len(importance_df)),
        width=importance_df['Importance'],
        color=colors[2]
    )

    # Add feature names as y-tick labels
    plt.yticks(np.arange(len(importance_df)), importance_df['Feature'])

    # Add value labels to the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}', ha='left', va='center')

    # Add titles and labels
    plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
#%%
def plot_tree_depth_vs_performance(X_train, y_train, X_val, y_val):
    """
    Plot how Random Forest performance changes with different tree depths
    """
    # Range of tree depths to test
    max_depths = np.arange(1, 26, 2)

    # Lists to store validation scores
    auc_scores = []

    # Train models with different depths and evaluate
    for depth in max_depths:
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=depth, random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # Get predictions and calculate AUC
        y_val_pred_proba = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_pred_proba)
        auc_scores.append(auc)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(max_depths, auc_scores, marker='o', linestyle='-', color=colors[4], linewidth=2)

    # Find best depth and highlight it
    best_depth_idx = np.argmax(auc_scores)
    best_depth = max_depths[best_depth_idx]
    best_auc = auc_scores[best_depth_idx]

    # Highlight best point
    plt.plot(best_depth, best_auc, 'ro', markersize=10)
    plt.annotate(f'Best Depth: {best_depth}\nAUC: {best_auc:.3f}',
                xy=(best_depth, best_auc), xytext=(best_depth+5, best_auc-0.02),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    # Add titles and labels
    plt.title('Random Forest Performance vs Tree Depth', fontsize=14, fontweight='bold')
    plt.xlabel('Maximum Tree Depth', fontsize=12)
    plt.ylabel('ROC AUC Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('random_forest_depth_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
#%%
def plot_n_estimators_vs_performance(X_train, y_train, X_val, y_val):
    """
    Plot how Random Forest performance changes with different numbers of trees
    """
    # Range of n_estimators to test
    n_estimators_range = np.arange(10, 301, 20)

    # Lists to store validation scores
    auc_scores = []

    # Train models with different numbers of trees and evaluate
    for n_estimators in n_estimators_range:
        rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=15, random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # Get predictions and calculate AUC
        y_val_pred_proba = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_pred_proba)
        auc_scores.append(auc)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, auc_scores, marker='o', linestyle='-', color=colors[5], linewidth=2)

    # Find best n_estimators and highlight it
    best_n_idx = np.argmax(auc_scores)
    best_n = n_estimators_range[best_n_idx]
    best_auc = auc_scores[best_n_idx]

    # Highlight best point
    plt.plot(best_n, best_auc, 'ro', markersize=10)
    plt.annotate(f'Best Trees: {best_n}\nAUC: {best_auc:.3f}',
                xy=(best_n, best_auc), xytext=(best_n+50, best_auc-0.02),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    # Add titles and labels
    plt.title('Random Forest Performance vs Number of Trees', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Trees', fontsize=12)
    plt.ylabel('ROC AUC Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('random_forest_n_estimators_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
#%%
def visualize_single_tree(rf_model, feature_names, output_file='single_tree_visualization.png'):
    """
    Visualize a single decision tree from the Random Forest
    """
    try:
        # Extract a single tree from the forest (just the first one for simplicity)
        tree_idx = 0
        tree = rf_model.estimators_[tree_idx]

        # Export the tree as a DOT file
        dot_data = export_graphviz(
            tree,
            feature_names=feature_names,
            filled=True,
            rounded=True,
            special_characters=True,
            max_depth=3  # Limit depth for readability
        )

        # Convert to PNG using graphviz
        graph = graphviz.Source(dot_data)
        graph.render(output_file.replace('.png', ''), format='png')

        print(f"Tree visualization saved to {output_file}")

    except Exception as e:
        print(f"Error visualizing tree: {e}")
        print("Make sure graphviz is installed on your system.")
        print("Try: pip install graphviz, and install the Graphviz software.")
#%%
# Main function to generate all visualizations
def generate_model_visualizations(X_train, y_train, X_val, y_val, log_reg, rf_classifier):
    """
    Generate all visualizations for Logistic Regression and Random Forest models
    """
    # Get feature names
    feature_names = X_train.columns

    print("Generating Logistic Regression visualizations...")

    # Logistic Regression visualizations
    plot_logistic_regression_coefficients(log_reg, feature_names)
    plot_logistic_regression_decision_boundary(X_train, y_train, log_reg)
    plot_regularization_path(X_train, y_train, X_val, y_val)

    print("Generating Random Forest visualizations...")

    # Random Forest visualizations
    plot_random_forest_feature_importance(rf_classifier, feature_names)
    plot_tree_depth_vs_performance(X_train, y_train, X_val, y_val)
    plot_n_estimators_vs_performance(X_train, y_train, X_val, y_val)

    # Single tree visualization (requires graphviz)
    try:
        visualize_single_tree(rf_classifier, feature_names)
    except:
        print("Skipping single tree visualization (graphviz may not be installed)")

    print("All visualizations completed!")
#%%
# After training your models
generate_model_visualizations(X_train, y_train, X_val, y_val, log_reg, rf_classifier)
#%% md
# <u><h1> 5 Evaluating Machine Learning Models
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, roc_curve,
                            confusion_matrix, classification_report)
import joblib
import warnings
warnings.filterwarnings('ignore')
#%%
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
colors = sns.color_palette("viridis", 10)

#%%
# Load the data and models
try:
    # Try to load the preprocessed data
    data = pd.read_csv("cleaned_data_with_clusters.csv")

    # Load saved models
    log_reg = joblib.load('logistic_regression_model.pkl')
    rf_classifier = joblib.load('random_forest_model.pkl')
    xgb_classifier = joblib.load('xgboost_model.pkl')
    voting_clf = joblib.load('voting_classifier_model.pkl')

    print("Loaded data and models successfully")
except:
    print("Error loading data or models. Make sure to run the model training code first.")
    raise
#%%
# Prepare data for evaluation
# Create target variable
median_income = data['ConvertedCompYearly'].median()
data['HighIncome'] = (data['ConvertedCompYearly'] > median_income).astype(int)

# Define features
features = []

# Include numerical features
numerical_features = ['YearsCodePro']
features.extend(numerical_features)
#%%
# Include key categorical features
# Country - keep only top countries by count
top_countries = data['Country'].value_counts().nlargest(10).index
for country in top_countries:
    col_name = f'Country_{country}'
    data[col_name] = (data['Country'] == country).astype(int)
    features.append(col_name)

# Education level
if 'EdLevel' in data.columns:
    top_ed_levels = data['EdLevel'].value_counts().nlargest(5).index
    for level in top_ed_levels:
        col_name = f'EdLevel_{level}'
        data[col_name] = (data['EdLevel'] == level).astype(int)
        features.append(col_name)

# Organization size
if 'OrgSize' in data.columns:
    top_org_sizes = data['OrgSize'].value_counts().nlargest(5).index
    for size in top_org_sizes:
        col_name = f'OrgSize_{size}'
        data[col_name] = (data['OrgSize'] == size).astype(int)
        features.append(col_name)

# Cluster assignments
if 'KMeans_Cluster' in data.columns:
    for cluster in range(4):  # Assuming 4 clusters from previous analysis
        col_name = f'Cluster_{cluster}'
        data[col_name] = (data['KMeans_Cluster'] == cluster).astype(int)
        features.append(col_name)

# DevType
if 'DevType' in data.columns and isinstance(data['DevType'].iloc[0], str):
    # Handle multi-response
    all_dev_types = []
    for types in data['DevType'].dropna():
        if ';' in types:
            all_dev_types.extend([t.strip() for t in types.split(';')])
        else:
            all_dev_types.append(types.strip())

    # Get top 5 developer types
    top_dev_types = pd.Series(all_dev_types).value_counts().nlargest(5).index

    for dev_type in top_dev_types:
        col_name = f'DevType_{dev_type}'
        if ';' in data['DevType'].iloc[0]:
            # Multi-response column
            data[col_name] = data['DevType'].apply(
                lambda x: 1 if isinstance(x, str) and dev_type in x.split(';') else 0
            )
        else:
            # Single response column
            data[col_name] = (data['DevType'] == dev_type).astype(int)
        features.append(col_name)

# Languages
if 'LanguageHaveWorkedWith' in data.columns and isinstance(data['LanguageHaveWorkedWith'].iloc[0], str):
    # Handle multi-response
    all_languages = []
    for langs in data['LanguageHaveWorkedWith'].dropna():
        if ';' in langs:
            all_languages.extend([l.strip() for l in langs.split(';')])
        else:
            all_languages.append(langs.strip())

    # Get top 5 languages
    top_languages = pd.Series(all_languages).value_counts().nlargest(5).index

    for language in top_languages:
        col_name = f'Language_{language}'
        if ';' in data['LanguageHaveWorkedWith'].iloc[0]:
            # Multi-response column
            data[col_name] = data['LanguageHaveWorkedWith'].apply(
                lambda x: 1 if isinstance(x, str) and language in x.split(';') else 0
            )
        else:
            # Single response column
            data[col_name] = (data['LanguageHaveWorkedWith'] == language).astype(int)
        features.append(col_name)

# Prepare data for modeling
X = data[features]
y = data['HighIncome']
#%%
# Standardize numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#%% md
# 1. Model Performance Comparison
#%%
def plot_model_performance_comparison():
    """
    Plot a comparison of different metrics across all models
    """
    # Dictionary to store evaluation results
    models = {
        'Logistic Regression': log_reg,
        'Random Forest': rf_classifier,
        'XGBoost': xgb_classifier,
        'Voting Classifier': voting_clf
    }

    # Calculate metrics for each model
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    results = pd.DataFrame(index=metrics)

    for model_name, model in models.items():
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        # Store results
        results[model_name] = [acc, prec, rec, f1, auc]

    # Create a bar plot
    plt.figure(figsize=(14, 8))

    # Set up bar positions
    bar_width = 0.18
    r1 = np.arange(len(metrics))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    # Create bars
    plt.bar(r1, results['Logistic Regression'], width=bar_width, label='Logistic Regression', color=colors[0])
    plt.bar(r2, results['Random Forest'], width=bar_width, label='Random Forest', color=colors[2])
    plt.bar(r3, results['XGBoost'], width=bar_width, label='XGBoost', color=colors[4])
    plt.bar(r4, results['Voting Classifier'], width=bar_width, label='Voting Classifier', color=colors[6])

    # Add labels and legend
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks([r + bar_width*1.5 for r in range(len(metrics))], metrics)
    plt.legend()

    # Add value labels to bars
    for i, model_name in enumerate(models.keys()):
        r = [r1, r2, r3, r4][i]
        for j, value in enumerate(results[model_name]):
            plt.text(r[j], value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print the results table
    print("Model Performance Comparison:")
    print(results.round(3))

    return results
#%% md
# 2. Confusion Matrix Visualization
#%%
def plot_confusion_matrices():
    """
    Plot confusion matrices for all models
    """
    # Dictionary to store models
    models = {
        'Logistic Regression': log_reg,
        'Random Forest': rf_classifier,
        'XGBoost': xgb_classifier,
        'Voting Classifier': voting_clf
    }

    # Set up the subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # For each model, create a confusion matrix plot
    for i, (model_name, model) in enumerate(models.items()):
        # Get predictions
        y_pred = model.predict(X_test)

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i])

        # Calculate metrics for title
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Add labels
        axes[i].set_xlabel('Predicted Label', fontsize=10)
        axes[i].set_ylabel('True Label', fontsize=10)
        axes[i].set_title(f'{model_name}\nAccuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}',
                        fontsize=12, fontweight='bold')

        # Set tick labels
        axes[i].set_xticks([0.5, 1.5])
        axes[i].set_yticks([0.5, 1.5])
        axes[i].set_xticklabels(['Low Income', 'High Income'])
        axes[i].set_yticklabels(['Low Income', 'High Income'])

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
#%% md
# 3. ROC Curves
#%%
def plot_roc_curves():
    """
    Plot ROC curves for all models
    """
    # Dictionary to store models
    models = {
        'Logistic Regression': log_reg,
        'Random Forest': rf_classifier,
        'XGBoost': xgb_classifier,
        'Voting Classifier': voting_clf
    }

    # Create plot
    plt.figure(figsize=(12, 8))

    # Plot diagonal line for random classifier
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')

    # For each model, plot ROC curve
    for i, (model_name, model) in enumerate(models.items()):
        # Get predicted probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

        # Calculate AUC
        auc = roc_auc_score(y_test, y_pred_proba)

        # Plot ROC curve
        plt.plot(fpr, tpr, color=colors[i*2], label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)

    # Add labels and legend
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

#%% md
# 4. Feature Importance Comparison
#%%
def plot_feature_importance_comparison():
    """
    Plot feature importance comparison across models
    """
    # Get feature importances from each model
    feature_names = X.columns

    # For Logistic Regression
    log_reg_coef = np.abs(log_reg.coef_[0])
    log_reg_imp = pd.Series(log_reg_coef, index=feature_names).sort_values(ascending=False)

    # For Random Forest
    rf_imp = pd.Series(rf_classifier.feature_importances_, index=feature_names).sort_values(ascending=False)

    # For XGBoost
    xgb_imp = pd.Series(xgb_classifier.feature_importances_, index=feature_names).sort_values(ascending=False)

    # Combine and normalize
    importances = pd.DataFrame({
        'Logistic Regression': log_reg_imp / log_reg_imp.sum(),
        'Random Forest': rf_imp / rf_imp.sum(),
        'XGBoost': xgb_imp / xgb_imp.sum()
    })

    # Calculate average importance
    importances['Average'] = importances.mean(axis=1)

    # Sort by average importance
    importances = importances.sort_values('Average', ascending=False)

    # Get top 10 features
    top_features = importances.head(10)

    # Create plot
    plt.figure(figsize=(14, 8))

    # Create heatmap
    sns.heatmap(top_features.T, annot=True, fmt='.3f', cmap='viridis', cbar=True)

    # Add labels
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Models', fontsize=12, fontweight='bold')
    plt.title('Feature Importance Comparison (Top 10 Features)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print importance table
    print("Top 10 Feature Importance Comparison:")
    print(top_features.round(3))

    return top_features
#%% md
# Run all visualizations and evaluations
# 
#%%
def run_all_evaluations():
    print("Generating model performance comparison...")
    results = plot_model_performance_comparison()

    print("\nGenerating confusion matrices...")
    plot_confusion_matrices()

    print("\nGenerating ROC curves...")
    plot_roc_curves()

    print("\nGenerating feature importance comparison...")
    top_features = plot_feature_importance_comparison()

    print("\nAll evaluation visualizations completed!")

    return results, top_features
#%% md
# Execute all evaluations
#%%
if __name__ == "__main__":
    results, top_features = run_all_evaluations()