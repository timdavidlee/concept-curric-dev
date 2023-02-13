from typing import List

import pandas as pd
from caseconverter import snakecase
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def normalize_column_names(df: pd.DataFrame):
    df.columns = [snakecase(c) for c in df.columns]


def make_linear_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """For linear + NN models:

    1. categorical inputs must be changed into 1-hot encoded columns
    2. numeric inputs must be normalized by mean + std dev

    Args:
        df:
            the raw input data
        numeric_cols:
            the numeric columns that need to be standardized around 1
        categorical_cols:
            the categorical columns that need to be converted to 1 hot encoding

    Returns:
        ColumnTransformer: can be used to transform the data before fitting to a model
    """

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", RobustScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def make_tree_preprocessor(categorical_cols: List[str]):
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor

