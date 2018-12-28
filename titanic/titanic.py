import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

train = pd.read_csv("train.csv", index_col='PassengerId')
y = train.pop('Survived').values
X = train.drop(['Name', 'Ticket', 'Cabin'], axis=1)

is_num = np.array([dt is not np.dtype('O') for dt in train.dtypes])
num_si_step = ('num_si', SimpleImputer())
ss_step = ('ss', StandardScaler())
num_pl = Pipeline([num_si_step, ss_step])
cat_si_step = ('cat_si', SimpleImputer(strategy='most_frequent'))
ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
cat_pl = Pipeline([cat_si_step, ohe_step])
ct = ColumnTransformer([('num_pl', num_pl, is_num),
                        ('cat_pl', cat_pl, ~is_num)])

