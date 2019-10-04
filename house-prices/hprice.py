import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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
