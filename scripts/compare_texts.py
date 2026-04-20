import pandas as pd
import hashlib

aide = pd.read_csv("data/raw/aide/AIDE_train_essays.csv")
daigt = pd.read_csv("data/raw/daigt_v2/train_v2_drcat_02.csv")

def hash_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

aide["h"] = aide["text"].astype(str).map(hash_text)
daigt["h"] = daigt["text"].astype(str).map(hash_text)

overlap = set(aide["h"]) & set(daigt["h"])
print("Exact text overlaps:", len(overlap))

# show the overlapping rows in each dataset
print("AIDE overlapping rows:")
print(aide[aide["h"].isin(overlap)][["id","prompt_id","generated"]])

print("DAIGT overlapping rows:")
print(daigt[daigt["h"].isin(overlap)][["source","label"]].head(20))
