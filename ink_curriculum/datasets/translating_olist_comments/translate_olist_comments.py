"""python -m ink_curriculum.datasets.translate_olist_comments"""
import os
import json
import logging
import ujson
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional

import pandas as pd
from google.cloud import translate_v2
from tqdm import tqdm

API_KEY_FILE = (
    "/Users/timlee/Dropbox/keys/google-cloud-translator-project"
    "/google-translate-api-key.txt"
)

PROJECT_ID = "translator"
PARENT = "projects/{}".format(PROJECT_ID)

logger = logging.getLogger("olist-translation")
strm = logging.StreamHandler()
fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
strm.setFormatter(fmt)
logger.addHandler(strm)
logger.setLevel(logging.INFO)

def get_google_json_key_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def get_google_api_key(path: str):
    with open(path, "r") as f:
        return f.read()
    
def translate_text(
    text: str,
    target_language_code: str,    
    api_key: str,
    source_language_code: Optional[str] = None,
):
    """Translate a string message

    Args:
        text: some sort of foreign language
        target_language_code: should be 2-letter `en`
        api_key: should be the google-api-key
    """
    client = translate_v2.Client(
        client_options=dict(
            api_key=api_key,
            quota_project_id=PROJECT_ID
        )
    )
    
    response = client.translate(
        text,
        target_language=target_language_code,
        source_language=source_language_code,
    )

    return response


def translate_msg(
    raw_msg: dict,
    output_dir: Path,
    api_key: str,
    target_language_code: str = "en",
    source_language_code: str = "pt",
):
    """
    Args:
        raw_msg: the raw message
        output_dir: the location where to save the english message
        
    Sample of input:
        {
            "review_id": "e629c32c18b7bfb4ba0e0958b1458f1f",
            "order_id": "a83bbd7210fc35296b93c16a48be49b8",
            "review_score": 5,
            "review_comment_title": "",
            "review_comment_message": "",
            "review_creation_date": "2017-07-18 00:00:00",
            "review_answer_timestamp": "2017-07-21 18:58:21",
            "review_comment_title_eng": "",
            "review_comment_message_eng": ""
        }
    """
    filename = str(output_dir / "{}.json".format(raw_msg["review_id"]))
    if os.path.exists(filename):
        return

    for ky in ["review_comment_title", "review_comment_message"]:
        if len(raw_msg[ky]) < 4:
            raw_msg["{}_eng".format(ky)] = ""
        else:
            raw_msg["{}_eng".format(ky)] = translate_text(
                raw_msg[ky],
                target_language_code=target_language_code,
                api_key=api_key,
            )

    with open(filename, "w") as f:
        f.write(ujson.dumps(raw_msg))


def main():
    api_key = get_google_api_key(API_KEY_FILE)

    file = "/Users/timlee/Documents/data/olist/olist_order_reviews_dataset.csv"
    df = pd.read_csv(file)
    logger.info("loading data from {:,}: {}".format(df.shape[0], file))
    list_of_dicts = df.fillna("").to_dict("records")
    output_dir = Path("./translated_msgs")
    output_dir.mkdir(parents=True, exist_ok=True)

    single_trans_func = partial(
        translate_msg,
        output_dir=output_dir,
        api_key=api_key,
        target_language_code="en",
        source_language_code="pt",
    )

    # list_of_dicts = list_of_dicts[:3000]

    # generally a little less than the number of cores
    logger.info("starting translation")
    # with ThreadPoolExecutor(6) as pool:
    #     list(tqdm(
    #         pool.map(
    #             single_trans_func,
    #             list_of_dicts,
    #         ),
    #         total=len(list_of_dicts)
    #     ))
    for datadict in tqdm(list_of_dicts):
        single_trans_func(datadict)


if __name__ == "__main__":
    main()
