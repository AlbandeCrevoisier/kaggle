#!/usr/bin/python
""" Digit recognizer """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score


dataset = pd.read_csv("train.csv")

y = dataset.pop('label')
X = dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

clf = make_pipeline(StandardScaler(), MLPClassifier((20, 20, 20, 20)))

print(cross_val_score(clf, X_train, y_train, cv=5))

clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

test = pd.read_csv("test.csv")
p = pd.DataFrame({'ImageId': test.index + 1, 'Label': clf.predict(test)})
p.to_csv("sub.csv", index=False)
