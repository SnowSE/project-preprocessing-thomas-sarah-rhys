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


plt.figure(figsize=(8, 5))
sns.countplot(x='Loneliness', data=df, color='skyblue')
plt.title('Target Distribution: Loneliness Levels', fontsize=12)
plt.xlabel('Loneliness Level', fontsize=10)
plt.ylabel('Count', fontsize=10)
sns.despine()
plt.savefig('target_distribution.png')


#################################################
# Part 2: Exploratory Data Analysis (EDA)
#################################################
# Select numeric columns, drop ID, and add jitter
df_viz = df.select_dtypes(include=[np.number]).drop(columns=['Unnamed: 0'])
df_viz += np.random.uniform(-0.5, 0.5, df_viz.shape)
# sns.pairplot(data=df_viz)
# plt.savefig('pairplot.png')

sns.PairGrid(data=df_viz).map_upper(sns.scatterplot).map_lower(sns.kdeplot).map_diag(sns.histplot)
plt.savefig('kde_histplot_pairplot.png')

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

