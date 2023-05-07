from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import shap

TELCO_CHURN_CSV_FILE = "/Users/timlee/Documents/data/telco-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MASSDROP_CSV_FILE = "/Users/timlee/Documents/data/massdrop-catalog/massdrop-products.csv"
OLIST_PRODUCTS_FILE = "/Users/timlee/Documents/data/olist/olist_products_dataset.csv"
OLIST_CATEGORY_FILE = "/Users/timlee/Documents/data/olist/product_category_name_translation.csv"

# churn_data = pd.read_csv(CSV_FILE)
# massdrop = pd.read_csv(MASSDROP_CSV_FILE)


FEAT_COLS = [
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]


def load_data(target: str = "housewares"):
    olist_products = pd.read_csv(OLIST_PRODUCTS_FILE)
    olist_product_name_mapping = pd.read_csv(OLIST_CATEGORY_FILE)

    merged = olist_products.merge(
        olist_product_name_mapping,
        how="left",
        on="product_category_name"
    )
    merged = merged.dropna(subset=FEAT_COLS, how="any")
    
    
    y = (merged["product_category_name_english"] == "housewares") * 1.
    y = y.values
    x = merged[FEAT_COLS]

    return x, y


def main():
    x, y = load_data()

    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x)


    models = {
        "logreg": LogisticRegression(max_iter=5_000),
        "decisiontree": DecisionTreeClassifier(max_depth=6),
        "boostedstree": LGBMClassifier(max_depth=6)
    }

    kfold = StratifiedKFold(n_splits=5)

    counter = 0
    score_collector = defaultdict(list)
    for trn_inds, val_inds in kfold.split(x, y):
        counter += 1
        x_trn = x_norm[trn_inds]
        y_trn = y[trn_inds]
        x_val = x_norm[val_inds]
        y_val = y[val_inds]

        for name, m in models.items():
            m.fit(x_trn, y_trn)
            y_val_pred = m.predict_proba(x_val)[:, 1]
            score = roc_auc_score(y_val, y_val_pred)
            score_collector[name].append(score)
            print("[{}]: {:.04f} - {}".format(counter, score, name))
            
    for ky, score_set in score_collector.items():
        print("{}:\t{:.03f}".format(ky, np.mean(score_set)))


    mm = models["decisiontree"]
    feat_df = pd.DataFrame({
        "feat_imp": mm.feature_importances_,
        "feat_name": x.columns
    }).sort_values("feat_imp", ascending=False)
    feat_df

    explainer = shap.TreeExplainer(mm)
    shap_values = explainer.shap_values(x_val, y=y_val)
    shap.summary_plot(shap_values, x)


if __name__ == "__main__":
    main()
