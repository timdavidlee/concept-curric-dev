"""python -m ink_curriculum.projects.hotel_reservations"""
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

from ink_curriculum.projects.funcs import (
    normalize_column_names,
    make_linear_preprocessor,
    make_tree_preprocessor,
    make_tree_model,
    make_linear_model,
)


def main():
    df = pd.read_csv("/Users/timlee/Documents/data/vehicle-insurance-claim-fraud/fraud_oracle.csv")
    normalize_column_names(df)