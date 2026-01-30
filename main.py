import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('young-people-survey-responses.csv')

#################################################
# Part 1: Data Loading and Initial Exploration
#################################################
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())

# Target Distribution (Loneliness)
print("\n--- Village - town value counts ---")
print(df["Village - town"].value_counts())

print("\n--- Internet usage value counts ---")
print(df["Internet usage"].value_counts())


# plt.figure(figsize=(8, 5))
# sns.countplot(x='Loneliness', data=df, color='skyblue')
# plt.title('Target Distribution: Loneliness Levels', fontsize=12)
# plt.xlabel('Loneliness Level', fontsize=10)
# plt.ylabel('Count', fontsize=10)
# sns.despine()
# plt.savefig('figures/target_distribution.png')


#################################################
# Part 2: Exploratory Data Analysis (EDA)
#################################################

# # pairplot with jitter
# df_viz = df.select_dtypes(include=[np.number]).drop(columns=['Unnamed: 0'])
# df_viz += np.random.uniform(-0.5, 0.5, df_viz.shape)
# sns.pairplot(data=df_viz)
# plt.savefig('figures/pairplot.png')

# # Pair grid with regplot and kde plot
# sns.PairGrid(data=df).map_upper(sns.regplot).map_lower(sns.kdeplot).map_diag(sns.histplot)
# plt.savefig('figures/kde_histplot_pairplot.png')

# # Correlation Heatmap
# plt.figure(figsize=(20, 16))
# numeric_cols = df.select_dtypes(include=[np.number]).drop(columns=['Unnamed: 0'])
# correlation_matrix = numeric_cols.corr()
# sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
#             center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
# plt.title('Correlation Heatmap of Numeric Variables', fontsize=14, pad=20)
# plt.tight_layout()
# plt.savefig('figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
# plt.close()

# Box Plots grouped by Loneliness Level
numeric_cols = df.select_dtypes(include=[np.number]).drop(columns=['Unnamed: 0'])
n_cols = len(numeric_cols.columns)
n_rows = (n_cols + 2) // 3
fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 4))
axes = axes.flatten()
for idx, col in enumerate(numeric_cols.columns):
    if col != 'Loneliness':
        sns.boxplot(x='Loneliness', y=col, data=df, ax=axes[idx], hue='Loneliness', palette='Set2', legend=False)
        axes[idx].set_title(f'Box Plot: {col} by Loneliness', fontsize=10)
        axes[idx].set_xlabel('Loneliness Level', fontsize=9)
        axes[idx].set_ylabel(col, fontsize=9)
plt.tight_layout()
plt.savefig('figures/boxplots_by_loneliness.png', dpi=300, bbox_inches='tight')
plt.close()

# Violin Plots grouped by Loneliness Level
fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 4))
axes = axes.flatten()
for idx, col in enumerate(numeric_cols.columns):
    if col != 'Loneliness':
        sns.violinplot(x='Loneliness', y=col, data=df, ax=axes[idx], hue='Loneliness', palette='Set3', legend=False)
        axes[idx].set_title(f'Violin Plot: {col} by Loneliness', fontsize=10)
        axes[idx].set_xlabel('Loneliness Level', fontsize=9)
        axes[idx].set_ylabel(col, fontsize=9)
plt.tight_layout()
plt.savefig('figures/violinplots_by_loneliness.png', dpi=300, bbox_inches='tight')
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

