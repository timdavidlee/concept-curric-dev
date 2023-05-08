"""
python -m ink_curriculum.projects.credit_card_fraud
"""
import os
import pandas as pd
from caseconverter import snakecase
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

pd.set_option("display.max_rows", 500)

"""
kaggle dataset:
    https://www.kaggle.com/datasets/mishra5001/credit-card
"""


feature_cols = [
    "NAME_CONTRACT_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "FLAG_MOBIL",
    "FLAG_WORK_PHONE",
    "FLAG_EMP_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
    "ORGANIZATION_TYPE",
]


binary_feats = [
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]


cat_feats = [
    "NAME_CONTRACT_TYPE",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]

cont_feats = [
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "CNT_FAM_MEMBERS",
]


def load_data():
    feather_file = "credit-card-fraud-app.feather"
    if os.path.exists(feather_file):
        return pd.read_feather(feather_file)

    df = pd.read_csv("/Users/timlee/Documents/data/credit-card-fraud/application_data.csv")
    df.to_feather(feather_file)
    return df


def load_schema_metadata():
    file = "/Users/timlee/Documents/data/credit-card-fraud/columns_description.csv"
    df = pd.read_csv(file, encoding="latin1")
    mask = df["Table"] == "application_data"
    lkup = {k: v for k, v in df.loc[mask, ["Row", "Description"]].values}
    return lkup


def encode_binary_feats(df):
    df_binary = df[binary_feats].reset_index(drop=True).copy()
    for col in binary_feats:
        df_binary[col] = df_binary[col].map(lambda x: 1 if x in (1, "Y", "y") else 0)
    return df_binary


def encode_cat_feats(df, encoder: OneHotEncoder = None):
    input_df = df[cat_feats].copy()
    for col in cat_feats:
        unique_vals = input_df[col].astype(str).fillna("").unique()
        clean_mapping = {uv: snakecase(uv) for uv in unique_vals}
        input_df[col] = input_df[col].fillna("").map(clean_mapping)

    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit(input_df)

    df_cat = encoder.transform(input_df)
    flat_cols = ["{}|{}".format(feat, c) for feat, subset in zip(cat_feats, encoder.categories_) for c in subset]
    df_cat = pd.DataFrame(df_cat, columns=flat_cols)
    return df_cat, encoder


def encode_cont_feats(df, scaler: StandardScaler = None):
    input_df = df[cont_feats].copy()

    # fill blanks with mean value
    for col in cont_feats:
        mean = input_df[col].mean()
        input_df[col] = input_df[col].fillna(mean)

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(input_df)

    df_cont = scaler.transform(input_df)
    df_cont = pd.DataFrame(df_cont, columns=cont_feats)
    return df_cont, scaler


def main():
    df = load_data()
    # schema_dict = load_schema_metadata()

    print("fraud: {:.04f} %, {}, {}".format(
        df["TARGET"].mean(),
        df["TARGET"].sum(),
        df["TARGET"].shape[0]
    ))

    y = df["TARGET"].values

    trn_df, val_df, y_trn, y_val = train_test_split(df, y, stratify=y, test_size=0.30)

    trn_binary = encode_binary_feats(trn_df)
    trn_cat, encoder = encode_cat_feats(trn_df)
    trn_cont, scaler = encode_cont_feats(trn_df)

    val_binary = encode_binary_feats(val_df)
    val_cat, _ = encode_cat_feats(val_df, encoder)
    val_cont, _ = encode_cont_feats(val_df, scaler)

    x_trn = pd.concat([trn_binary, trn_cat, trn_cont], axis=1)
    x_val = pd.concat([val_binary, val_cat, val_cont], axis=1)
    print("train + validation sizes: {} {}".format(x_trn.shape, x_val.shape))

    print("===== logreg =====")
    lin_model = LogisticRegression(max_iter=5_000, verbose=1)
    lin_model.fit(x_trn, y_trn)

    y_val_probs = lin_model.predict_proba(x_val)[:, 1]

    auc_score = roc_auc_score(y_val, y_val_probs)
    print("auc", auc_score)

    threshold = 0.4
    y_val_pred = (y_val_probs > threshold) * 1.
    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()

    score_labels = {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }
    print(score_labels)

    print("==== RandomForest =====")
    tree_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight="balanced")
    tree_model.fit(x_trn, y_trn)

    y_val_probs = tree_model.predict_proba(x_val)[:, 1]

    auc_score = roc_auc_score(y_val, y_val_probs)
    print("auc", auc_score)

    threshold = 0.4
    y_val_pred = (y_val_probs > threshold) * 1.
    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()

    score_labels = {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }
    print(score_labels)


if __name__ == "__main__":
    main()
