import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('young-people-survey-responses.csv')

print(df.shape)
print(df.head())
print(df.info())
print(df.describe())