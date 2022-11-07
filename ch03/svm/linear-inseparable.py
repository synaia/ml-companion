import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)
# np.random.randn from standard normal dist. mean zero 0, var 1
# so; if x1 and x2 are opossite (centralized mean 0), like (x1:-0.58 and x2: 1.2)
# then this is True or yi = 1 in logical_xor condition.
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, 0)
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='royalblue', marker='s', label='Class 1')
plt.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], c='tomato', marker='o', label='Class 2')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')
plt.tight_layout()
plt.show()