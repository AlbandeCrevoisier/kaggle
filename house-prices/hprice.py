import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
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

gbrt = GradientBoostingRegressor()
gbrt.fit(X, y)

test = pd.read_csv("test.csv", index_col='Id')
test = test[numerical_features].fillna(test.mean())

p = pd.DataFrame({'Id': test.index,
                  'SalePrice': gbrt.predict(test)})
p.to_csv("sub.csv", index=False)
