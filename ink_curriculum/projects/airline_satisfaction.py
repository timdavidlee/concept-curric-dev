"""python -m ink_curriculum.projects.airline_satisfaction"""
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate

from ink_curriculum.projects.funcs import (
    normalize_column_names,
    make_linear_preprocessor,
    make_tree_preprocessor,
    make_linear_model,
    make_tree_model
)


def main():
    filename = "/Users/timlee/Documents/data/airline-passenger-satisfaction/train.csv"
    df = pd.read_csv(filename, index_col="id")
    normalize_column_names(df)
    df = df.drop(["unnamed_0", "satisfaction"], axis=1)

    df["customer_type_binary"] = (df["customer_type"] == "Loyal Customer") * 1.

    x = df.drop(["customer_type", "customer_type_binary"], axis=1)
    y = df["customer_type_binary"].values

    categorical_feature_cols = [
        "gender", "type_of_travel", "class"
    ]

    numeric_feature_cols = [
        "age",
        "flight_distance",
        "inflight_wifi_service",
        "departure_arrival_time_convenient",
        "ease_of_online_booking",
        "gate_location",
        "food_and_drink",
        "seat_comfort",
        "inflight_entertainment",
        "on_board_service",
    ]

    linear_preprocessor = make_linear_preprocessor(numeric_feature_cols, categorical_feature_cols)
    tree_preprocessor = make_tree_preprocessor(categorical_cols=categorical_feature_cols)

    classifiers = [
        ("log_reg", make_linear_model(linear_preprocessor)),
        ("tree", make_tree_model(tree_preprocessor)),
    ]

    for name, clf in classifiers:
        scores = cross_validate(clf, x, y, scoring="roc_auc")
        print(name, "AUC:", np.round(scores["test_score"], 3))



if __name__ == "__main__":
    main()
