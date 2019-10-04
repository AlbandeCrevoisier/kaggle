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

gbr = GradientBoostingRegressor()
print(cross_val_score(gbr, X, y, cv=10).mean())
