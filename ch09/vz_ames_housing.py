import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix


columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
df = pd.read_csv('../dataset/AmesHousing.txt', sep='\t', usecols=columns)
df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
df.dropna(axis=0, inplace=True)

scatterplotmatrix(df.values, figsize=(12, 10), names=df.columns, alpha=0.5)
plt.tight_layout()
plt.show()

print()