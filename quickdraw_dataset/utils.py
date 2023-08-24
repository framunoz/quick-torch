from enum import Enum
from pathlib import Path

import requests

_CACHE_PATH = Path(".", ".quickdraw_cache")
_CACHE_PATH.mkdir(parents=True, exist_ok=True)
_CAT_PATH = _CACHE_PATH / "categories.txt"

# Download categories.txt if it does not exist
if not _CAT_PATH.exists():
    url = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
    res = requests.get(url)
    with _CAT_PATH.open("w") as f:
        f.write(res.text)
    res.close()

# Create auxiliary lists for the enumerator
with _CAT_PATH.open("r") as f:
    _CATEGORIES = [
        (cat[:-1].upper().replace(" ", "_").replace("-", "_"), cat[:-1]) for cat in f
    ]
    _INT_LABEL = {value: label for label, (_, value) in enumerate(_CATEGORIES)}

"""Enumerates the possible categories of QuickDraw"""
Category = Enum("Category", _CATEGORIES)


def _query(self):
    return self.value.replace(" ", "%20")


def _label(self):
    return _INT_LABEL[self.value]


def _str_(self):
    return self.value


Category.query = property(_query)
Category.label = property(_label)
Category.__str__ = _str_
