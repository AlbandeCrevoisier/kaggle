import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

train = pd.read_csv("train.csv", index_col='PassengerId')
y = train.pop('Survived').values
X = train.drop(['Name', 'Ticket', 'Cabin'], axis=1)

cat = ['Pclass', 'Sex', 'Embarked']
cat_si_step = ('si', SimpleImputer(strategy='most_frequent'))
cat_ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
cat_pipe = Pipeline([cat_si_step, cat_ohe_step])
age_si_step = ('si', SimpleImputer())
age_mm_step = ('mm', MinMaxScaler())
age_pipe = Pipeline([age_si_step, age_mm_step])
num_si_step = ('si', SimpleImputer())
num_ss_step = ('ss', StandardScaler())
num_pipe = Pipeline([num_si_step, num_ss_step])
ct = ColumnTransformer([('cat', cat_pipe, cat),
                        ('age', age_pipe, ['Age'])],
                        remainder=num_pipe,
                        n_jobs=-1)

clf_pipe = Pipeline([('ct', ct),
                    ('clf', RandomForestClassifier(n_estimators=100,
                                                   n_jobs=-1))])
clf_pipe.fit(X, y)

print(cross_val_score(clf_pipe, X, y, cv=10).mean())

test = pd.read_csv("test.csv", index_col='PassengerId')
test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

p = pd.DataFrame({'PassengerId': test.index,
                  'Survived': clf_pipe.predict(test)})
p.to_csv("sub.csv", index=False)

