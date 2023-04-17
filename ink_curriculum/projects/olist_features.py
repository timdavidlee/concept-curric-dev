import os
import time
from typing import Union
import pandas as pd
from pathlib import Path
from functools import wraps

PathLike = Union[Path, str]


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"{total_time:.4f} seconds:\t{func.__name__}")
        return result
    return timeit_wrapper


def save_feather(df: pd.DataFrame, filename: PathLike):
    df.to_feather(str(filename))
    print("saved: {} to {}".format(df.shape[0], str(filename)))


class OListDataset:

    _CUSTOMERS_CSV = "olist_customers_dataset.csv"
    _ORDERS_CSV = "olist_orders_dataset.csv"
    _ORDER_ITEMS_CSV = "olist_order_items_dataset.csv"
    _PAYMENTS_CSV = "olist_order_payments_dataset.csv"
    _PRODUCTS_CSV = "olist_products_dataset.csv"
    _SELLERS_CSV = "olist_sellers_dataset.csv"

    _DATASETS = {
        "customers": _CUSTOMERS_CSV,
        "orders": _ORDERS_CSV,
        "order_items": _ORDER_ITEMS_CSV,
        "payments": _PAYMENTS_CSV,
        "products": _PRODUCTS_CSV,
        "sellers": _SELLERS_CSV,
    }

    def __init__(self, csv_dir: PathLike):
        self.csv_dir = Path(csv_dir)
        self.unified_order_items_df = self._make_unified_order_items_df()
        self.unified_orders_df = self._make_unified_orders_df()

    def _make_unified_orders_df(self):
        df_orders = self.get_dataset("orders")
        df_customers = self.get_dataset("customers")
        return df_orders.merge(
            df_customers,
            how="left",
            on="customer_id"
        )

    def _make_unified_order_items_df(self):
        df_order_items = self.get_dataset("order_items")
        df_products = self.get_dataset("products")
        df_sellers = self.get_dataset("sellers")

        return df_order_items.merge(
            df_products,
            how="left",
            on="product_id"
        ).merge(
            df_sellers,
            how="left",
            on="seller_id"
        )

    def get_dataset(self, key: str):
        if key not in self._DATASETS:
            raise ValueError(
                "{} not found, must be one of these: {}".format(
                    key, self._DATASETS.keys()
                )
            )

        filepath = self.csv_dir / self._DATASETS[key]
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                "missing file, expecting: {}".format(filepath)
            )
        df = pd.read_csv(filepath)
        if key == "orders":
            self._format_orders_df(df)
        return df

    @staticmethod
    def _format_orders_df(df_orders: pd.DataFrame) -> pd.DataFrame:
        date_fields = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
        for f in date_fields:
            df_orders[f] = pd.to_datetime(df_orders[f])


@timeit
def create_order_features(
    df_orders: pd.DataFrame,
    output_dir: PathLike
) -> list:
    """Creates estimated deliver lookup by days"""
    df_orders["estimated_delivery"] = \
        df_orders["order_estimated_delivery_date"] - \
        df_orders["order_purchase_timestamp"]

    df_orders["estimated_delivery"] = df_orders["estimated_delivery"].dt.days
    output = df_orders[
        ["order_id", "customer_id", "estimated_delivery"]
    ].copy()

    savefile = output_dir / "order2estimated_delivery.feather"
    save_feather(output, savefile)
    return [savefile]


@timeit
def create_order2customer_state(
    df_orders: pd.DataFrame,
    df_customers: pd.DataFrame,
    output_dir: PathLike
):
    merged_df = df_orders.merge(
        df_customers[["customer_id", "customer_state"]],
        how="left",
        on="customer_id"
    )

    savefile = output_dir / "order2customer_state.feather"
    save_feather(
        merged_df[["order_id", "customer_state"]],
        savefile
    )
    return [savefile]


@timeit
def create_order2category_distribution(
    df_order_items: pd.DataFrame,
    df_products: pd.DataFrame,
    output_dir: PathLike,
):
    merged_df = df_order_items.merge(
        df_products[
            ["product_id", "product_category_name", "product_weight_g"]
        ],
        how="left",
        on="product_id"
    )

    agg_df = (
        merged_df
        .groupby(["order_id", "product_category_name"], as_index=False)
        .agg(
            item_ct=pd.NamedAgg("order_item_id", "size"),
            item_weight_g=pd.NamedAgg("product_weight_g", "sum"),
        )
    )

    agg_df1 = agg_df.pivot_table(
        index="order_id",
        columns="product_category_name",
        values="item_ct",
        aggfunc="sum",
        fill_value=0
    )
    agg_df1.columns = ["ct_{}".format(c) for c in agg_df1.columns]
    agg_df1["ct_total"] = agg_df1.sum(axis=1)
    agg_df1 = agg_df1.reset_index()

    agg_df2 = agg_df.pivot_table(
        index="order_id",
        columns="product_category_name",
        values="item_weight_g",
        aggfunc="sum",
        fill_value=0
    )
    agg_df2.columns = ["wt_{}".format(c) for c in agg_df2.columns]
    agg_df2["wt_total"] = agg_df2.sum(axis=1)
    agg_df2 = agg_df2.reset_index()

    savefile1 = output_dir / "order2category_count.feather"
    savefile2 = output_dir / "order2category_weight_g.feather"
    save_feather(agg_df1, savefile1)
    save_feather(agg_df2, savefile2)
    return [savefile1, savefile2]


@timeit
def create_order2price_and_freight(
    df_order_items: pd.DataFrame,
    output_dir: PathLike
):
    agg_df = df_order_items.groupby("order_id").agg(
        item_ct=pd.NamedAgg("order_item_id", "size"),
        total_price=pd.NamedAgg("price", "sum"),
        total_freight=pd.NamedAgg("freight_value", "sum")
    ).reset_index()
    savefile = output_dir / "order2price_and_freight.feather"
    save_feather(agg_df, savefile)
    return [savefile]


class FeatureFactory:
    def __init__(
        self,
        olist_dataset: OListDataset,
        cache_dir: PathLike
    ):
        self.olist_dataset = olist_dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_dataset(self, key: str):
        return self.olist_dataset.get_dataset(key)

    def make_features(self):
        fls = []
        fls += create_order_features(
            self._get_dataset("orders"),
            self.cache_dir,
        )

        fls += create_order2category_distribution(
            self._get_dataset("order_items"),
            self._get_dataset("products"),
            self.cache_dir,
        )
        fls += create_order2price_and_freight(
            self._get_dataset("order_items"),
            self.cache_dir,
        )

        fls += create_order2customer_state(
            self._get_dataset("orders"),
            self._get_dataset("customers"),
            self.cache_dir,
        )
