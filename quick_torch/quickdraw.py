import io
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import ndjson
import numpy as np
import requests
from PIL import Image
from torchvision.datasets.vision import VisionDataset

from .utils import Category

_CATEGORY_T = Category | str
_LABEL = {cat: i for i, cat in enumerate(Category)}


class QuickDraw(VisionDataset):
    """`QuickDraw <https://quickdraw.withgoogle.com/data>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``QuickDraw/<category>/data``, ``QuickDraw/<category>/data_recognized``
            and  ``QuickDraw/<category>/data_not_recognized`` exist.
        category (Category, str, optional): The specific category to use. It is an element of ``Category`` enumerator.
        recognized (bool, optional): Wheter the draw was recognized or not.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    ndjson_url = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
    numpy_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

    def __init__(
        self,
        root: str,
        category: _CATEGORY_T | Sequence[_CATEGORY_T] = "face",
        recognized: bool = None,
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        # self.category: list[Category]
        # match category:
        #     case None:
        #         self.category = [cat for cat in Category]
        #     case Category(category) as cat:
        #         self.category = [cat]
        #     case _:
        #         self.category = [Category(cat) for cat in category]
            
        self.category: Category = Category(category)
        self.recognized: bool = recognized

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data = self._load_data().reshape(-1, 28, 28)

    @property
    def folder(self) -> Path:
        return Path(self.root, self.__class__.__name__, self.category.value)

    def _check_exists(self) -> bool:
        return self.folder.exists()

    def _check_all_files_exists(self) -> bool:
        files_name = [
            f"data{recog}.npy" for recog in ["", "_recognized", "_not_recognized"]
        ]
        return all([(self.folder / name).exists() for name in files_name])

    def download(self):
        """Download the QuickDraw data if it doesn't exist already."""
        if self._check_all_files_exists():
            return

        # Create path of directories
        self.folder.mkdir(parents=True, exist_ok=True)
        # Download npy file
        url = self.numpy_url + self.category.query + ".npy"
        try:
            print(f"Downloading {url}")
            response = requests.get(url)
            response.raise_for_status()
            data = np.load(io.BytesIO(response.content))
            np.save(self.folder / f"data.npy", data)
        except Exception as error:
            print(f"Failed to download (trying next):\n{error}")
        finally:
            print()

        # Download ndjson file
        url = self.ndjson_url + self.category.query + ".ndjson"
        try:
            print(f"Downloading {url}")
            response = requests.get(url)
            items = response.json(cls=ndjson.Decoder)
            recognized = []
            for item in items:
                recognized.append(item["recognized"])
            recognized = np.array(recognized)
            data_recognized = data[recognized]
            data_not_recognized = data[~recognized]
            np.save(self.folder / f"data_recognized.npy", data_recognized)
            np.save(self.folder / f"data_not_recognized.npy", data_not_recognized)
        except Exception as error:
            print(f"Failed to download (trying next):\n{error}")
        finally:
            print()

    def _load_data(self):
        name_file = "data"
        match self.recognized:
            case True:
                name_file += "_recognized"
            case False:
                name_file += "_not_recognized"
        name_file += ".npy"

        return np.load(self.folder / name_file)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.fromarray(self.data[index], mode="L")
        target = _LABEL[self.category]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
