import io
import math
import typing as t
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data.dataset import ConcatDataset, Dataset
from torchvision.datasets.vision import VisionDataset

from .utils import Category

__all__ = ["QuickDraw"]

_CATEGORY_T = Category | str
_LABEL = {category: i for i, category in enumerate(Category)}


def _create_list_categories(
    categories: _CATEGORY_T | t.Iterable[_CATEGORY_T],
) -> list[Category]:
    if categories is None:
        to_return = list(Category)
    elif isinstance(categories, (str, Category)):
        to_return = [Category(categories)]
    elif isinstance(categories, t.Iterable):
        to_return = [Category(category) for category in categories]
    else:
        raise TypeError("Please provide a category or an iterable of categories.")
    return to_return


class QuickDraw(VisionDataset):
    """`QuickDraw <https://quickdraw.withgoogle.com/data>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``QuickDraw/<category>/data``, ``QuickDraw/<category>/data_recognized``
            and  ``QuickDraw/<category>/data_not_recognized`` exist.
        categories (Category, str, list, optional): The specific category to use. It is an element of ``Category`` enumerator.
        train (bool, optional): If True, uses the train data, otherwise uses the test data. If None, use all data.
            Use the train_percentage parameter to decide the proportion of data for the split.
        train_percentage (float): The proportion of data for the split.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        recognized (bool, optional): If true, uses the recognized data. If None, use all data.
        max_items_per_class (int, optional): The maximum number of images per category. If None, use all data.
        seed (int, optional): Sets the seed for the division. Used for the train_percentage parameter.
            Default is 12722028422223837445.
    """

    ndjson_url = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
    numpy_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

    def __init__(
        self,
        root: str,
        categories: t.Optional[_CATEGORY_T | t.Sequence[_CATEGORY_T]] = "face",
        train: t.Optional[bool] = None,
        train_percentage: float = 0.9,
        transform: t.Optional[t.Callable] = None,
        target_transform: t.Optional[t.Callable] = None,
        download: bool = False,
        recognized: t.Optional[bool] = None,
        max_items_per_class: t.Optional[int] = None,
        seed: t.Optional[int] = 12722028422223837445,
    ) -> None:
        super().__init__(
            root=root, transform=transform, target_transform=target_transform
        )

        self.categories: list[Category] = _create_list_categories(categories)
        self.train = train
        self.train_percentage = train_percentage
        self.recognized: bool = recognized
        self.max_items_per_class = max_items_per_class
        self.seed = seed or 12722028422223837445

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data, self.targets = self._load_data()

    def __getitem__(self, index: int) -> t.Tuple[t.Any, t.Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.fromarray(self.data[index], mode="L")
        target = self.targets[index]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def copy(self):
        """Return a copy of the dataset."""
        return QuickDraw(
            root=self.root,
            categories=self.categories,
            train=self.train,
            train_percentage=self.train_percentage,
            transform=self.transform,
            target_transform=self.target_transform,
            download=False,
            recognized=self.recognized,
            max_items_per_class=self.max_items_per_class,
            seed=self.seed,
        )

    def __add__(self, other: Dataset) -> t.Union[ConcatDataset, Dataset]:
        if not isinstance(other, self.__class__):
            return super().__add__(other)

        categories = list(set(self.categories + other.categories))
        data = np.concatenate((self.data, other.data), axis=0)
        targets = np.concatenate((self.targets, other.targets), axis=0)
        to_return = QuickDraw(
            root=self.root,
            categories=categories,
            train=self.train,
            train_percentage=self.train_percentage,
            transform=self.transform,
            target_transform=self.target_transform,
            download=False,
            recognized=self.recognized,
            max_items_per_class=self.max_items_per_class,
            seed=self.seed,
        )
        to_return.data = data
        to_return.targets = targets

        return to_return

    def _get_indice_n_val(self, train_percentage, seed):
        train_percentage = train_percentage or self.train_percentage
        seed = seed or self.seed

        generator = torch.manual_seed(seed)
        n_data = self.data.shape[0]
        indices = torch.randperm(n_data, generator=generator).tolist()
        n_val = math.floor(len(indices) * train_percentage)
        return indices, n_val

    def get_test_data(self, train_percentage=None, seed=None):
        """Return a copy of the dataset with the test data."""
        if self.train is False:
            return self.copy()

        if self.train is True:
            raise ValueError(
                "The train parameter is set to True. You can't get the test data."
            )

        indices, n_val = self._get_indice_n_val(train_percentage, seed)

        data = self.data[indices[n_val:]]
        targets = self.targets[indices[n_val:]]

        self_copy = self.copy()
        self_copy.data = data
        self_copy.targets = targets

        return self_copy

    def get_train_data(self, train_percentage=None, seed=None):
        """Return a copy of the dataset with the train data."""
        if self.train is True:
            return self.copy()

        if self.train is False:
            raise ValueError(
                "The train parameter is set to False. You can't get the train data."
            )

        indices, n_val = self._get_indice_n_val(train_percentage, seed)

        data = self.data[indices[:n_val]]
        targets = self.targets[indices[:n_val]]

        self_copy = self.copy()
        self_copy.data = data
        self_copy.targets = targets

        return self_copy

    @property
    def folders(self) -> list[Path]:
        """List of all category folders."""
        return [
            Path(self.root, self.__class__.__name__, cat.value)
            for cat in self.categories
        ]

    def _check_exists(self) -> bool:
        """Check that all the respective folders exist."""
        return all([folder.exists() for folder in self.folders])

    def _check_files_exists(self, folder: Path) -> bool:
        """Check that all the necessary files exist in that folder."""
        files_name = ["data.npy"]
        if self.recognized is not None:
            files_name += ["data_recognized.npy", "data_not_recognized.npy"]
        files = [(folder / name).exists() for name in files_name]
        return all(files)

    def _check_all_files_exist(self) -> bool:
        """Check that all files exist in their respective folders."""
        files = [self._check_files_exists(folder) for folder in self.folders]
        return all(files)

    def download(self):
        """Download the QuickDraw data if it doesn't exist already."""
        if self._check_all_files_exist():
            return

        # Create path of directories
        for folder in self.folders:
            folder.mkdir(parents=True, exist_ok=True)

        # Create list of urls
        urls_npy = [
            self.numpy_url + category.query + ".npy" for category in self.categories
        ]
        urls_ndjson = [
            self.ndjson_url + category.query + ".ndjson" for category in self.categories
        ]

        for url_npy, url_ndjson, folder in zip(urls_npy, urls_ndjson, self.folders):
            data = None

            # Download npy file
            if not (folder / "data.npy").exists():
                try:
                    print(f"Downloading {url_npy}")
                    response = requests.get(url_npy)
                    response.raise_for_status()
                    data = np.load(io.BytesIO(response.content))
                    np.save(folder / "data.npy", data)

                except Exception as error:
                    print(f"Failed to download (trying next):\n{error}")

                finally:
                    print()

            # Download ndjson file
            if (
                self.recognized is not None
                and not (folder / "data_recognized.npy").exists()
                and not (folder / "data_not_recognized.npy").exists()
            ):
                try:
                    import jsonlines

                    print(f"Downloading {url_ndjson}")
                    response = requests.get(url_ndjson)
                    items = jsonlines.Reader(io.BytesIO(response.content))

                    recognized = []
                    for item in items:
                        recognized.append(item["recognized"])
                    recognized = np.array(recognized)

                    if data is None:
                        data = np.load(folder / "data.npy")

                    data_recognized = data[recognized]
                    data_not_recognized = data[~recognized]

                    np.save(folder / "data_recognized.npy", data_recognized)
                    np.save(folder / "data_not_recognized.npy", data_not_recognized)

                except Exception as error:
                    print(f"Failed to download (trying next):\n{error}")

                finally:
                    print()

    def _load_data(self):
        X = np.empty([0, 784], dtype=np.uint8)
        y = np.empty([0], dtype=np.int16)

        for folder, category in zip(self.folders, self.categories):
            name_file = "data"
            if self.recognized is None:
                pass
            elif self.recognized:
                name_file += "_recognized"
            else:
                name_file += "_not_recognized"
            name_file += ".npy"

            data = np.load(folder / name_file)

            if self.train is not None:
                generator = torch.manual_seed(self.seed)
                n_data = data.shape[0]
                indices = torch.randperm(n_data, generator=generator).tolist()
                n_val = math.floor(len(indices) * self.train_percentage)
                if self.train:
                    data = data[indices[:n_val]]
                else:
                    data = data[indices[n_val:]]

            if self.max_items_per_class:
                data = data[: self.max_items_per_class, :]

            labels = np.full(data.shape[0], _LABEL[category])

            X = np.concatenate((X, data), axis=0)
            y = np.append(y, labels)

        X = X.reshape(-1, 28, 28)

        return X, y
