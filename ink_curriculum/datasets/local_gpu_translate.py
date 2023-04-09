"""python -m ink_curriculum.datasets.local_gpu_translate"""
import os
from copy import deepcopy
from pathlib import Path
from typing import List

import ujson
import pandas as pd
import numpy as np
from transformers import pipeline

ckpt = 'Narrativa/mbart-large-50-finetuned-opus-pt-en-translation'
TITLE_COL = "review_comment_title"
MESSAGE_COL = "review_comment_message"

def load_src_data():
    path = "~/olist_order_reviews_dataset.csv"
    df = pd.read_csv(path, low_memory=False)
    df[TITLE_COL] = df[TITLE_COL].fillna("")
    df[MESSAGE_COL] = df[MESSAGE_COL].fillna("")
    print(df.shape[0])
    list_of_jsons = df.to_dict("records")
    return list_of_jsons


def main(list_of_dicts: List[dict], outdir: Path):
    print("loading model")
    pipe = pipeline("translation", model=ckpt)
    print("model loading done ...")

    total_rec_ct = len(list_of_dicts)
    for k, datadict in enumerate(list_of_dicts):
        review_id = datadict["review_id"]
        outfile = outdir / "{}.json".format(review_id)
        if os.path.exists(outfile):
            print("file already found: {}".format(outfile))
            continue

        outdata = deepcopy(datadict)
        try:
            converted_ct = 0
            for ky in [TITLE_COL, MESSAGE_COL]:
                
                translated_ky = "{}_eng".format(ky)
                raw_val = outdata.get(ky)
                if len(raw_val) < 4:
                    outdata[translated_ky] = ""
                else:
                    converted_ct += 1
                    outdata[translated_ky] = pipe(
                        raw_val,
                        truncation=True,
                        max_length=512,
                        src_lang="pt_XX",
                        tgt_lang="en_XX"
                    )
            with open(str(outfile), "w") as f:
                ujson.dump(outdata, f)
                print("saved: [{:,} / {:} ] {}".format(total_rec_ct, k + 1, outfile))

            # if converted_ct > 0:
            #     print("translated" + ujson.dumps(outdata, indent=2))

        except Exception as e:
            print(ujson.dumps(datadict, indent=2))
            raise e

    
if __name__ == "__main__":
    # main()
    data = load_src_data()
    outdir = Path("/home/tlee/translated_olist/")
    outdir.mkdir(parents=True, exist_ok=True)
    main(data, outdir)