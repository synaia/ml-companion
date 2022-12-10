import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from help.helper import lin_reg_plot, get_housingdata

X, y, _ = get_housingdata()
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
