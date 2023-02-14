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


def main_classification():
    filename = "/Users/timlee/Documents/data/hotel-reservations-cancellation/Hotel Reservations.csv"

    df = pd.read_csv(filename, encoding="unicode_escape")
    normalize_column_names(df)
    df = df.drop(["booking_id"], axis=1)

    df["booking_status_binary"] = (df["booking_status"] == "Canceled") * 1.

    categorical_feature_cols = [
        "type_of_meal_plan", "room_type_reserved", "market_segment_type",
        "arrival_year", "arrival_month", "arrival_date",
    ]

    numeric_feature_cols = [
        "no_of_adults",
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "required_car_parking_space",
        "repeated_guest",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "avg_price_per_room",
        "no_of_special_requests",
    ]

    x = df[categorical_feature_cols + numeric_feature_cols]
    y = df["booking_status_binary"].values

    linear_preprocessor = make_linear_preprocessor(numeric_feature_cols, categorical_feature_cols)
    tree_preprocessor = make_tree_preprocessor(categorical_feature_cols)

    classifiers = [
        ("log_reg", make_linear_model(linear_preprocessor)),
        ("tree", make_tree_model(tree_preprocessor)),
    ]

    print("{:,}".format(y.mean()))

    for name, clf in classifiers:
        scores = cross_validate(clf, x, y, scoring="roc_auc")
        print(name, np.round(scores["test_score"], 3))


def main_regression():
    def make_linear_model(preprocessor: ColumnTransformer):
        return Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LinearRegression()),
        ])


    def make_tree_model(preprocessor: ColumnTransformer):
        return Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBRegressor()),
        ])

    filename = "/Users/timlee/Documents/data/hotel-reservations-cancellation/Hotel Reservations.csv"

    df = pd.read_csv(filename, encoding="unicode_escape")
    normalize_column_names(df)

    categorical_feature_cols = [
        "type_of_meal_plan", "room_type_reserved", "market_segment_type",
        "arrival_year", "arrival_month", "arrival_date",
    ]

    numeric_feature_cols = [
        "no_of_adults",
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "required_car_parking_space",
        "repeated_guest",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "no_of_special_requests",
    ]
    
    x = df[categorical_feature_cols + numeric_feature_cols]
    y = df["avg_price_per_room"].values

    linear_preprocessor = make_linear_preprocessor(numeric_feature_cols, categorical_feature_cols)
    tree_preprocessor = make_tree_preprocessor(categorical_feature_cols)

    classifiers = [
        ("log_reg", make_linear_model(linear_preprocessor)),
        ("tree", make_tree_model(tree_preprocessor)),
    ]

    for name, clf in classifiers:
        scores = cross_validate(clf, x, y, scoring="neg_root_mean_squared_error")
        print(name, np.round(scores["test_score"], 3))


if __name__ == "__main__":
    main_classification()
    main_regression()
