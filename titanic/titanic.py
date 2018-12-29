import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

train = pd.read_csv("train.csv", index_col='PassengerId')
y = train.pop('Survived').values
X = train.drop(['Name', 'Ticket', 'Cabin'], axis=1)

cat = ['Pclass', 'Sex', 'Embarked']
cat_si_step = ('cat_si', SimpleImputer(strategy='most_frequent'))
cat_ohe_step = ('cat_ohe',
                OneHotEncoder(sparse=False, handle_unknown='ignore'))
cat_pipe = Pipeline([cat_si_step, cat_ohe_step])
age_si_step = ('age_si', SimpleImputer())
age_mm_step = ('age_mm', MinMaxScaler())
age_pipe = Pipeline([age_si_step, age_mm_step])
num_si_step = ('num_si', SimpleImputer())
num_ss_step = ('num_ss', StandardScaler())
num_pipe = Pipeline([num_si_step, num_ss_step])
ct = ColumnTransformer([('cat_pipe', cat_pipe, cat),
                        ('age_pipe', age_pipe, ['Age'])],
                        remainder=num_pipe,
                        n_jobs=-1)
print(ct.fit_transform(X))

