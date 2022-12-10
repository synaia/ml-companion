import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
import seaborn as sb

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
df = pd.read_csv('../dataset/AmesHousing.txt', sep='\t', usecols=columns)
df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
df.dropna(axis=0, inplace=True)

cm = np.corrcoef(df.values.T)
hm = heatmap(cm, row_names=df.columns, column_names=df.columns)
plt.tight_layout()
plt.show()

# plt.figure()
# sb.heatmap(df.corr(), annot=True)
# plt.show()
