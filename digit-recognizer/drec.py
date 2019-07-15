#!/usr/bin/python
""" Digit recognizer """
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import maptplotlib.pyplot as plt
import seaborn as sns
sns.set()
