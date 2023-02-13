""" python -m ink_curriculum.projects.ecommerce_shipping"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate

from xgboost import XGBClassifier

from ink_curriculum.datasets.util import normalize_column_names


def make_linear_model(preprocessor: Pipeline):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression()),
    ])


def make_tree_model(preprocessor: Pipeline):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier()),
    ])


def main():
    categorical_feature_cols = [
        "warehouse_block", "mode_of_shipment"
    ]
    numeric_feature_cols = [
        "cost_of_the_product", "prior_purchases", "discount_offered", "weight_in_gms"
    ]

    df = pd.read_csv("/Users/timlee/Documents/data/ecommerce-shipping-data/Train.csv", index_col="ID")
    normalize_column_names(df)
    target = "reachedon_time_yn"

    x = df.drop(target, axis=1)
    y = df[target].values

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
            ("num", numeric_transformer, numeric_feature_cols),
            ("cat", categorical_transformer, categorical_feature_cols),
        ]
    )

    classifiers = [
        ("log_reg", make_linear_model(preprocessor)),
        ("tree", make_tree_model(preprocessor)),
    ]

    for name, clf in classifiers:
        scores = cross_validate(clf, x, y)
        print(name, np.round(scores["test_score"], 3))


if __name__ == "__main__":
    main()
