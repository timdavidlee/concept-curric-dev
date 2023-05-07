import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def load_events(
    csv_filename: str = "/Users/timlee/Documents/data/ecommerce-purchase-history/electronics-events.csv",
    feather_filename: str = "./electronics-events.feather",
    force: bool = False
):
    if os.path.exists(feather_filename) and (not force):
        logger.info("file exists, loading: {}".format(feather_filename))
        return pd.read_feather(feather_filename)

    events_df = pd.read_csv(
        csv_filename, parse_dates=["event_time",]
    ).sort_values(["user_session", "event_time"], ignore_index=True)

    events_df["event_time_epoch_sec"] = events_df["event_time"].astype(int) // 1_000_000_000
    events_df.to_feather(feather_filename)
    logger.info("saving to file, loading: {}".format(feather_filename))
    return events_df


def format_time(events_df: pd.DataFrame):
    events_df["total_session_ct"] = events_df.groupby("user_session").transform("size")
    events_df["next_event_time_epoch_sec"] = events_df.groupby("user_session")["event_time_epoch_sec"].shift(-1)
    events_df["time_spent"] = events_df["next_event_time_epoch_sec"] - events_df["event_time_epoch_sec"]
    events_df["time_spent"] = events_df["time_spent"].fillna(-1)
    events_df["dayofweek"] = events_df["event_time"].dt.dayofweek


def transform_y(events_df: pd.DataFrame, target_field: str, target_value: str):
    unique_user_session_id = events_df["user_session"].unique()
    unique_user_session_id = pd.Series(
        np.zeros(shape=unique_user_session_id.shape[0]),
        index=unique_user_session_id
    )
    mask = events_df[target_field] == target_value
    events_with_target_value = events_df.loc[mask, "user_session"].unique()
    unique_user_session_id[events_with_target_value] = 1
    return unique_user_session_id


def get_days_days_of_week_by_user_session(df: pd.DataFrame):
    holder = np.zeros(shape=(df.shape[0], 7))
    for j in range(7):
        m = df["dayofweek"] == j
        holder[m, j] = 1
    
    days_of_week = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    output_df = pd.DataFrame(holder, columns=days_of_week)
    nunique = df["user_session"].nunique()
    output_df["user_session"] = df["user_session"]
    out = output_df.groupby("user_session").max()
    assert out.shape[0] == nunique, Exception("{}, {}".format(out.shape[0], nunique))
    return out
    
    
def get_x(events_df: pd.DataFrame):
    mask_view_only = events_df["event_type"] == "view"
    return events_df[mask_view_only].groupby("user_session").agg(
        unique_product_views=pd.NamedAgg("product_id", "nunique"),
        unique_category_views=pd.NamedAgg("category_id", "nunique"),
        total_time_viewed_sec=pd.NamedAgg("time_spent", "sum"),
        max_price=pd.NamedAgg("price", "max"),
        min_price=pd.NamedAgg("price", "min"),
    )


def main():
    # history_df = pd.read_csv("/Users/timlee/Documents/data/ecommerce-purchase-history/electronics-purchase-history.csv")

    events_df = load_events(force=False)
    format_time(events_df)
    events_df = events_df.dropna(subset=["user_session"])
    events_df["category_code"] = events_df["category_code"].combine_first(events_df["category_id"])

    has_views = events_df["event_type"] == "view"
    user_sess_with_views = events_df.loc[has_views, "user_session"].unique()

    sessions_with_views = events_df["user_session"].isin(user_sess_with_views)
    events_df = events_df[sessions_with_views].reset_index(drop=True)


    # dow_encoder = OneHotEncoder()
    # dow_encoder.fit_transform(events_df[["event_time"]])

    user_session_start = events_df.groupby("user_session")["event_time_epoch_sec"].min().sort_values()
    val_pct = 0.15
    total_session_ct = user_session_start.shape[0]
    val_ct = int(user_session_start.shape[0] * val_pct)

    trn_sessions = user_session_start.head(total_session_ct - val_ct).index
    val_sessions = user_session_start.tail(val_ct).index

    mask_trn = events_df["user_session"].isin(trn_sessions)
    mask_val = events_df["user_session"].isin(val_sessions)

    trn_df = events_df[mask_trn].reset_index().copy()
    val_df = events_df[mask_val].reset_index().copy()


    y_trn = transform_y(trn_df, target_field="event_type", target_value="cart")
    y_val = transform_y(val_df, target_field="event_type", target_value="cart")

    x_trn_dow = get_days_days_of_week_by_user_session(trn_df)
    x_val_dow = get_days_days_of_week_by_user_session(val_df)

    scaler = StandardScaler()
    x_trn_feat = get_x(trn_df)
    vals = scaler.fit_transform(x_trn_feat)
    x_trn_feat = pd.DataFrame(vals, index=x_trn_feat.index, columns=x_trn_feat.columns)

    x_val_feat = get_x(val_df)
    vals = scaler.transform(x_val_feat)
    x_val_feat = pd.DataFrame(vals, index=x_val_feat.index, columns=x_val_feat.columns)


    x_trn_all = pd.concat([x_trn_dow, x_trn_feat], axis=1)
    x_val_all = pd.concat([x_val_dow, x_val_feat], axis=1)

    model = LogisticRegression(penalty="l2", solver="liblinear", max_iter=6_000)
    model.fit(x_trn_all, y_trn)

    y_trn_probs = model.predict(x_trn_all)
    y_val_probs = model.predict(x_val_all)


    score = roc_auc_score(y_trn, y_trn_probs)
    print(score)

    score = roc_auc_score(y_val, y_val_probs)
    print(score)

    model = RandomForestClassifier(n_estimators=100, n_jobs=8, min_samples_leaf=5, max_depth=40)
    model.fit(x_trn_all, y_trn)

    y_trn_probs = model.predict(x_trn_all)
    y_val_probs = model.predict(x_val_all)


    score = roc_auc_score(y_trn, y_trn_probs)
    print(score)

    score = roc_auc_score(y_val, y_val_probs)
    print(score)


if __name__ == "__main__":
    main()