"""python -m ink_curriculum.projects.ecommerce_shipping"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate

from ink_curriculum.projects.funcs import (
    normalize_column_names, 
    make_linear_preprocessor,
    make_tree_preprocessor,
    make_tree_model,
    make_linear_model,
)


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

    x = df[categorical_feature_cols + numeric_feature_cols]
    y = df[target].values

    linear_preprocessor = make_linear_preprocessor(categorical_feature_cols, numeric_feature_cols)
    tree_preprocessor = make_tree_preprocessor(categorical_feature_cols)

    classifiers = [
        ("log_reg", make_linear_model(linear_preprocessor)),
        ("tree", make_tree_model(tree_preprocessor)),
    ]

    for name, clf in classifiers:
        scores = cross_validate(clf, x, y, scoring="roc_auc")
        print(name, np.round(scores["test_score"], 3))


if __name__ == "__main__":
    main()
