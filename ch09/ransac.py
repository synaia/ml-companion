import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from help.helper import get_housingdata

X, y, _ = get_housingdata()


def mean_absolute_deviation(data):
    return np.mean(np.abs(data - np.mean(data)))


mad = mean_absolute_deviation(y)
ransac = RANSACRegressor(
    LinearRegression(),
    max_trials=100,   # this is the default value.
    min_samples=0.95,
    residual_threshold=None, # scikit-learn uses MAD <<Median Absolute Deviation>>
    random_state=123
    )
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_x = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_x[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c='steelblue', edgecolors='white', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='limegreen', edgecolors='white', marker='s', label='Outliers')
plt.plot(line_x, line_y_ransac, color='black', lw=2)
plt.xlabel('Living area above grownd in square feet')
plt.ylabel('Sale price in U.S. dollars')
plt.legend(loc='upper left', title=f'MAD: {mad:0.3f}')
plt.tight_layout()
plt.show()


