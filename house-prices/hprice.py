import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_csv("train.csv")

numerical_features = ['LotFrontage',
                      'LotArea',
                      'YearBuilt',
                      'YearRemodAdd',
                      'MasVnrArea',
                      'BsmtFinSF1',
                      'BsmtFinSF2',
                      'BsmtUnfSF',
                      'TotalBsmtSF',
                      '1stFlrSF',
                      '2ndFlrSF',
                      'LowQualFinSF',
                      'GrLivArea',
                      'GarageYrBlt',
                      'GarageArea',
                      'WoodDeckSF',
                      'OpenPorchSF',
                      'EnclosedPorch',
                      '3SsnPorch',
                      'ScreenPorch',
                      'PoolArea',
                      'MiscVal']
X = train[['SalePrice'] + numerical_features].dropna()
y = X.pop('SalePrice').values

br = BayesianRidge()
print(cross_val_score(br, X, y, cv=10).mean())
br.fit(X, y)

gbrt = GradientBoostingRegressor()
print(cross_val_score(gbrt, X, y, cv=10).mean())
gbrt.fit(X, y)

test = pd.read_csv("test.csv", index_col='Id')
test = test[numerical_features].fillna(test.mean())

p = pd.DataFrame({'SalePrice': br.predict(test)},
                 test.index)
p.to_csv("sub.csv")
