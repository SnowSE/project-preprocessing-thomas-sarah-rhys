import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('young-people-survey-responses.csv')

#################################################
# Part 1: Data Loading and Initial Exploration
#################################################

# 1. Dataset Overview
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())

# 2. Initial Data Quality
# Table of null value counts
print(df.isnull().sum())

# Value counts for important categorical variables
print("\n--- Village - town value counts ---")
print(df["Village - town"].value_counts())

print("\n--- Internet usage value counts ---")
print(df["Internet usage"].value_counts())

# Target Distribution (Loneliness)
plt.figure(figsize=(8, 5))
sns.countplot(x='Loneliness', data=df, color='skyblue')
plt.title('Target Distribution: Loneliness Levels', fontsize=12)
plt.xlabel('Loneliness Level', fontsize=10)
plt.ylabel('Count', fontsize=10)
sns.despine()
plt.savefig('figures/initial-exploration/target_distribution.png')


#################################################
# Part 2: Exploratory Data Analysis (EDA)
#################################################

# Drop specified columns
df = df.drop(columns=['Unnamed: 0', 'Music', "Parents' advice", 'Techno', 'Siblings'])
# Key Numeric Variables: Movies, History, Mathematics, Pets, Spiders, Loneliness, Finances, Age
# Key Categorical Variables: Internet usage, Gender, Village - town
key_numeric = ['Movies', 'History', 'Mathematics', 'Pets', 'Spiders', 'Loneliness', 'Finances', 'Age']
key_categorical = ['Internet usage', 'Gender', 'Village - town']


# 1. Univariate Analysis
# Numerical Histograms to show 
for col in key_numeric:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, data=df, color='skyblue')
    plt.title(f'{col} Distribution', fontsize=12)
    plt.xlabel(col, fontsize=10)
    plt.ylabel('Count', fontsize=10)
    sns.despine()
    plt.savefig(f'figures/eda/numeric_{col.lower()}_distribution.png')
    plt.close()

# Categorical histograms to show frequency distribution
for col in key_categorical:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, data=df, color='coral')
    plt.title(f'{col} Distribution', fontsize=12)
    plt.xlabel(col, fontsize=10)
    plt.ylabel('Count', fontsize=10)
    sns.despine()
    plt.savefig(f'figures/eda/categorical_{col.lower().replace(" ", "_").replace(" - ", "_")}_distribution.png')
    plt.close()


# 2. Multivariate Analysis
# Box Plots grouped by Loneliness Level
plot_cols = [col for col in key_numeric if col != 'Loneliness']
rows_of_figs = (len(plot_cols) + 2) // 3
fig, axes = plt.subplots(rows_of_figs, 3, figsize=(18, rows_of_figs * 4))
axes = axes.flatten()
for idx, col in enumerate(plot_cols):
    sns.boxplot(x='Loneliness', y=col, data=df, ax=axes[idx], hue='Loneliness', palette='Set2', legend=False)
    axes[idx].set_title(f'Box Plot: {col} by Loneliness', fontsize=10)
    axes[idx].set_xlabel('Loneliness Level', fontsize=9)
    axes[idx].set_ylabel(col, fontsize=9)
plt.tight_layout()
plt.savefig('figures/eda/boxplots_by_loneliness.png', dpi=300, bbox_inches='tight')
plt.close()

# Bar Charts: Frequency Distributions by Loneliness Level
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, col in enumerate(key_categorical):
    sns.countplot(x=col, hue='Loneliness', data=df, ax=axes[idx], palette='Set2')
    axes[idx].set_title(f'{col} by Loneliness Level', fontsize=11)
    axes[idx].set_xlabel(col, fontsize=10)
    axes[idx].set_ylabel('Count', fontsize=10)
    axes[idx].legend(title='Loneliness', fontsize=8, title_fontsize=9)
plt.tight_layout()
plt.savefig('figures/eda/categorical_by_loneliness.png', dpi=300, bbox_inches='tight')
plt.close()


# 3. Correlation Analysis
# Correlation Heatmap
plt.figure(figsize=(20, 16))
numeric_cols = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_cols.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap of Numeric Variables', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('figures/eda/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()


#################################################
# Part 3: Data Preprocessing
#################################################


#################################################
# Part 4: Cross Validation (Train-Test Split)
#################################################


#################################################
# Part 5: Feature Scaling
#################################################


#################################################
# Part 7: Summary and Reflection
#################################################

