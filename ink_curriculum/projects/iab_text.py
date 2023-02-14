"""python -m ink_curriculum.projects.iab_text"""
from typing import Tuple

import numpy as np
import pandas as pd
from caseconverter import snakecase

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from xgboost import XGBClassifier

import tensorflow_hub as tf_hub
import tensorflow_text
from ink_curriculum.projects.funcs import batchify


def load_base_df(filename: str):
    with open(filename, "r") as f:
        lines = f.read().split("\n")

    columns = [
        "size",
        "image_id",
        "domain",
        "category",
        "title",
        "description",
        "keywords",
    ]

    list_of_dicts = []
    for li in lines:
        list_of_dicts.append(
            {col:li for col, li in zip(columns, li.split(","))}
        )

    df = pd.DataFrame.from_dict(list_of_dicts)
    for c in columns:
        df[c] = df[c].str.replace("\"", "")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning the dataframe"""
    df = df[df["description"].str.len() > 0].reset_index(drop=True)
    df.insert(0, "id", df.index.values)
    return df


def extract_categories(
    df: pd.DataFrame,
    minfreq: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pull out categories + text

    Args:
        df: pd.DataFrame
        minfreq: int

    Returns:
        df: the downfiltered base frame
        cat_df: 
    """
    cat_df = df[["id", "category"]].copy()
    cat_df["category"] = cat_df["category"].str.split(r"/").map(lambda x: [snakecase(c) for c in x if len(c) > 0])
    cat_df = cat_df.explode("category")
    
    minfreq = 5
    counts = cat_df["category"].value_counts()
    mask = counts > minfreq
    keepcats = counts[mask].index

    cat_df = cat_df[cat_df["category"].isin(keepcats)].reset_index(drop=True)
    
    remaining_ids = cat_df["id"].unique()
    df = df[df["id"].isin(remaining_ids)].drop("category", axis=1).reset_index(drop=True)
    return df, cat_df



def single_topic_detector(df, cat_df, category: str = "web_design_development"):
    """Try some of the following top topics

        business_industrial       3063
        arts_entertainment        2543
        shopping                  2367
        internet_telecom          1988
        hobbies_leisure           1737
        computers_electronics     1391
        people_society            1369
        home_garden               1319
        web_services              1269
        apparel                   1124
        health                    1063
        web_design_development    1033
        jobs_education             962
        online_communities         951
    """

    cat_df["target"] = (cat_df["category"] == category) * 1.0
    labels = cat_df.groupby("id")["target"].max()

    x = df[["description"]].copy()
    y = labels

    features = ColumnTransformer([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2)), "description") 
    ], remainder="passthrough")

    linear_pipe = Pipeline(steps=[
        ("preprocess", features),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    
    tree_pipe = Pipeline(steps=[
        ("preprocess", features),
        ("model", XGBClassifier(n_estimators=100, max_depth=5)),
    ])
    
    classifiers = [
        ("log_reg", linear_pipe),
        ("tree", tree_pipe),
    ]
    
    print(category)
    print("{}% | {:,}".format(y.mean(), y.sum()))

    for name, clf in classifiers:
        scores = cross_validate(clf, x, y, scoring="roc_auc")
        print(name, np.round(scores["test_score"], 3))

        
def single_topic_detector_pretrained(df: pd.DataFrame, cat_df: pd.DataFrame, topic: str):
    """Using TF pretrained model to encode the text

    Args:
        df: features text + id
        cat_df: the labels
        topic: the single topic we are predicting
    """
    cat_df["target"] = (cat_df["category"] == topic) * 1.0
    labels = cat_df.groupby("id")["target"].max()

    x = df[["description"]].copy()
    y = labels

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    encoder = tf_hub.load(module_url)

    collector = []
    batch_size = 200
    counter = 0
    for subset in batchify(x["description"].tolist(), batch_size):
        collector.append(
            pd.DataFrame(
                encoder(subset).numpy()
            )
        )
        counter += batch_size
        print("{} / {}".format(counter, x.shape[0]))

    x_tf_vec = pd.concat(collector, axis=0)

    hub_pipeline = Pipeline(steps=[
        ("model", LogisticRegression(max_iter=1000)),
    ])
    scores = cross_validate(hub_pipeline, x_tf_vec, y, scoring="roc_auc")
    print("for topic: {}".format(topic))
    print("pretrained+logreg", np.round(scores["test_score"], 3))


def main():
    filename = "/Users/timlee/Documents/data/iab-text-website/training_data_en.csv"
    df = load_base_df(filename)
    df = preprocess(df)
    df, cat_df = extract_categories(df)
    single_topic_detector(df, cat_df, "web_design_development")
    single_topic_detector(df, cat_df, "arts_entertainment")
    single_topic_detector(df, cat_df, "jobs_education")
    single_topic_detector_pretrained(df, cat_df, "web_design_development")


if __name__ == "__main__":
    main()
