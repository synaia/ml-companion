import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
import seaborn as sb
from help.helper import get_housingdata

X, y, df = get_housingdata()

cm = np.corrcoef(df.values.T)
hm = heatmap(cm, row_names=df.columns, column_names=df.columns)
plt.tight_layout()
plt.show()

# plt.figure()
# sb.heatmap(df.corr(), annot=True)
# plt.show()
