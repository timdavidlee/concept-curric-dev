from typing import List, Any

import pandas as pd
from caseconverter import snakecase
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


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
        ],
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
        ],
        remainder="drop",
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
        ],
        remainder="passthrough"
    )
    return preprocessor


def make_linear_model(preprocessor: ColumnTransformer):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def make_tree_model(preprocessor: ColumnTransformer):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier()),
    ])


def batchify(input_list: List[Any], batch_size: int = 20):
    count = len(input_list)
    for j in range(0, count, batch_size):
        yield input_list[j: j + batch_size]
