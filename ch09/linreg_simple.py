import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def lin_reg_plot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolors='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)


columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
df = pd.read_csv('../dataset/AmesHousing.txt', sep='\t', usecols=columns)
df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
df.dropna(axis=0, inplace=True)

X, y = df[['Gr Liv Area']].values, df['SalePrice'].values

slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print(f'Slope: {slr.coef_[0]:0.3f}')
print(f'Intercept: {slr.intercept_:0.3f}')

lin_reg_plot(X, y, slr)
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Scale price in U.S. dollars')
plt.tight_layout()
plt.show()
