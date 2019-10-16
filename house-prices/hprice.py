import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import BayesianRidge


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

#TODO pipe to handle categorical_features

br = BayesianRidge()
print(cross_val_score(br, X, y, cv=5).mean())
br.fit(X, y)

test = pd.read_csv("test.csv", index_col='Id')
test = test[numerical_features + categorical_features].fillna(test.mean())

p = pd.DataFrame(
    {'SalePrice': br.predict(test)},
    test.index)
p.to_csv("sub.csv")
