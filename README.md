# Quick, Torch!

Quick, Torch! is a simple package that provides a ["Quick, Draw!"](https://github.com/googlecreativelab/quickdraw-dataset) dataset using the abstract class `VisionDataset`, provided by `torchvision` API. This package mirrors the [MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) dataset provided in torchvision.

You can install this package with
```
pip install quick-torch --upgrade
```

# Example
Here are a simple example of usage:
```python
from quick_torch import QuickDraw
import torchvision.transforms as T


ds = QuickDraw(
    root="dataset", 
    categories="face", 
    download=True, 
    transform=T.Resize((128, 128))
)
print(f"{len(ds) = }")
first_data = ds[0]
first_data

>>> Downloading https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/face.npy
>>> 
>>> len(ds) = 161666
>>> (<PIL.Image.Image image mode=L size=128x128>, 108)
```

For more examples, please refer to the notebook [example.ipynb](https://github.com/framunoz/quick-torch/blob/main/example.ipynb)

# How to cite?

If you use this code in your work, please reference it as follows:
```
@software{munoz2023quicktorch,
  author  = {Mu\~{n}oz, Francisco},
  license = {MIT},
  month   = {10},
  title   = {{quick-torch}},
  url     = {https://github.com/framunoz/quick-torch},
  version = {1.0.4},
  year    = {2023}
}
```

# References
This work was mainly inspired by the following repositories:
- [MNIST from torchvision](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py#L19)
- [The dataset of this notebook](https://github.com/nateraw/quickdraw-pytorch/blob/main/quickdraw.ipynb)
