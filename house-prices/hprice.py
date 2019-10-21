import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_score


train = pd.read_csv("train.csv")

numerical_features = [
    'LotFrontage',
    'LotArea',
    'YearBuilt',
    'YearRemodAdd',
    'TotalBsmtSF',
    '1stFlrSF',
    '2ndFlrSF',
    'GarageArea',
    'OpenPorchSF',
    'EnclosedPorch',
    'PoolArea']
categorical_features = [
    'Street',
    'OverallQual',
    'OverallCond',
    'CentralAir',
    'SaleType',
    'SaleCondition']

X = train[['SalePrice'] + numerical_features + categorical_features].dropna()
y = X.pop('SalePrice').values

num_pipe = make_pipeline(
    SimpleImputer(),
    StandardScaler())
cat_pipe = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(sparse=False, handle_unknown='ignore'))

ct = make_column_transformer(
    (num_pipe, numerical_features),
    (cat_pipe, categorical_features),
    n_jobs=-1)
clf = make_pipeline(ct, BayesianRidge())

print(cross_val_score(clf, X, y, cv=5).mean())
clf.fit(X, y)

test = pd.read_csv("test.csv", index_col='Id')

p = pd.DataFrame(
    {'SalePrice': clf.predict(test)},
    test.index)
p.to_csv("sub.csv")
