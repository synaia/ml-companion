from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.0**c, multi_class='ovr')