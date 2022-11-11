from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from help.helper import get_class_grades_csv, get_ae_credit_card

df = get_class_grades_csv()
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
# replace of each NaN with column meann
imputed_data = imr.transform(df.values)
print()

# replace of each NaN with column mean (pandas version)
df.fillna(df.mean(), inplace=True)

# most_frequent: useful for categorical features values.
imr = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imr = imr.fit(df.values)
# replace of each NaN with column mean
imputed_data = imr.transform(df.values)
print()


ccdf = get_ae_credit_card()
some_encoder = LabelEncoder()
z = some_encoder.fit_transform(ccdf['selfemp'].values)
print(z)