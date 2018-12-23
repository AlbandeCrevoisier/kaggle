import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv("train.csv", index_col='PassengerId')

X = data.drop(['Survived', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'],
    axis=1)
for c in X:
    X[c].fillna(X[c].mean(), inplace = True)
y = data['Survived']

test = pd.read_csv("test.csv", index_col='PassengerId')
test = test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
for c in test:
    test[c].fillna(test[c].mean(), inplace = True)

clf = RandomForestClassifier().fit(X, y)
pred = clf.predict(test)
p = pd.DataFrame(pred, columns=['Survived'])
pd.concat([pd.DataFrame(test.index), p], axis=1).to_csv("sub.csv", index=False)
