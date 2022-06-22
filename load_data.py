import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchgeo.datasets import BigEarthNet
from config import get_args


# load train data from sentinel 2
def load_data(data_root, batch_size=64, subset=False, peat_only=True):
    train_data = BigEarthNet(root=data_root,
                             split='train',
                             bands='s2',
                             num_classes=43,
                             transforms=None,
                             download=False)

    if subset:
        # this is for testing the network
        sub_inds = list(range(128))
        train_data = Subset(train_data, sub_inds)

    dataloader = DataLoader(train_data,
                            batch_size=batch_size,
                            shuffle=True)

    return train_data, dataloader

# re-writing class for BigEarthNet modified dataset
# https://torchgeo.readthedocs.io/en/latest/_modules/torchgeo/datasets/bigearthnet.html#BigEarthNet.__getitem__

import glob
import json
import os
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from torch import Tensor

from torchgeo.datasets import VisionDataset

def sort_sentinel2_bands(x: str) -> str:
    """Sort Sentinel-2 band files in the correct order."""
    x = os.path.basename(x).split("_")[-1]
    x = os.path.splitext(x)[0]
    if x == "B8A":
        x = "B08A"
    return x

class BigEarthNetModified(VisionDataset):
    """Modified version of the BigEarthNet dataset.
    Using a class inspired by Microsoft's here
    https://torchgeo.readthedocs.io/en/latest/_modules/torchgeo/datasets/bigearthnet.html#BigEarthNet.__getitem__

    The `BigEarthNet <https://bigearth.net/>`_
    dataset is a dataset for multilabel remote sensing image scene classification.

    In the modified version, I add the ability to load
    only the classes and bands I care about.

    Dataset features:

    * 590,326 patches from 125 Sentinel-1 and Sentinel-2 tiles
    * Imagery from tiles in Europe between Jun 2017 - May 2018
    * 12 spectral bands with 10-60 m per pixel resolution (base 120x120 px)
    * 2 synthetic aperture radar bands (120x120 px)
    * 43 or 19 scene classes from the 2018 CORINE Land Cover database (CLC 2018)

    Dataset format:

    * images are composed of multiple single channel geotiffs
    * labels are multiclass, stored in a single json file per image
    * mapping of Sentinel-1 to Sentinel-2 patches are within Sentinel-1 json files
    * Sentinel-1 bands: (VV, VH)
    * Sentinel-2 bands: (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    * All bands: (VV, VH, B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    * Sentinel-2 bands are of different spatial resolutions and upsampled to 10m

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS.2019.8900532

    """

    class_sets = {
        7: {'Land principally occupied by agriculture, with significant areas of natural vegetation': 20,
                  'Mixed forest': 22,
                  'Moors and heathland': 23,
                  'Pastures': 27,
                  'Peatbogs': 28,
                  'Salt marshes': 34,
                  'Water courses': 42},
        1: {'Peatbogs': 28},
    }

    image_size = (120, 120)

    splits_metadata = {
        "train": {
            "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/train.csv?inline=false",
            # noqa: E501
            "filename": "bigearthnet-train.csv",
            "md5": "623e501b38ab7b12fe44f0083c00986d",
        },
        "val": {
            "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/val.csv?inline=false",
            # noqa: E501
            "filename": "bigearthnet-val.csv",
            "md5": "22efe8ed9cbd71fa10742ff7df2b7978",
        },
        "test": {
            "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/test.csv?inline=false",
            # noqa: E501
            "filename": "bigearthnet-test.csv",
            "md5": "697fb90677e30571b9ac7699b7e5b432",
        },
    }
    metadata = {
        "s2": {
            "url": "https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz",
            "md5": "5a64e9ce38deb036a435a7b59494924c",
            "filename": "BigEarthNet-S2-v1.0.tar.gz",
            "directory": "BigEarthNet-v1.0",
        },
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        n_channels: str = "all",   # or three
        peat_only: bool = True,    # or seven
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
    ) -> None:

        assert split in self.splits_metadata
        self.root = root
        self.split = split
        self.n_channels = n_channels
        self.peat_only = peat_only
        self.transforms = transforms
        if peat_only:
            self.class2idx = {c: i for i, c in enumerate(self.class_sets[1])}
        else:
            self.class2idx = {c: i for i, c in enumerate(self.class_sets[7])}
        self.num_classes = len(self.class2idx.items())
        self.folders = self._load_folders()

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(index)
        label = self._load_target(index)
        sample: Dict[str, Tensor] = {"image": image, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.folders)


    def _load_folders(self) -> List[Dict[str, str]]:
        """Load folder paths.

        Returns:
            list of dicts of s1 and s2 folder paths
        """
        filename = self.splits_metadata[self.split]["filename"]
        dir_s2 = self.metadata["s2"]["directory"]

        with open(os.path.join(self.root, filename)) as f:
            lines = f.read().strip().splitlines()
            pairs = [line.split(",") for line in lines]

        # addition of condition for filtering folders to just ones with labels we want
        # for now can add the additional condition of only doing this when we know
        # that we are reducing the labels desired
        if self.peat_only:
            folders = [
                {
                    "s2": os.path.join(self.root, dir_s2, pair[0]),
                }
                for pair in pairs if self._refine_folders(pair)
            ]
        else:
            folders = [
                {
                    "s2": os.path.join(self.root, dir_s2, pair[0]),
                }
                for pair in pairs
            ]

        return folders

    def _refine_folders(self, pair):
        """Filters to only retain folders meeting class conditions.

        :return:
        """

        dir_s2 = self.metadata["s2"]["directory"]
        lab_file = pair[0] + '_labels_metadata.json'
        fp = os.path.join(self.root, dir_s2, pair[0], lab_file)
        with open(fp, 'r') as f:
            jsonfile = json.loads(f.read())
        if np.isin(list(self.class2idx.keys()), jsonfile["labels"]):
            return True
        else:
            return False

    def _load_paths(self, index: int) -> List[str]:
        """Load paths to band files.

        Args:
            index: index to return

        Returns:
            list of file paths
        """
        folder = self.folders[index]["s2"]
        paths = glob.glob(os.path.join(folder, "*.tif"))
        paths = sorted(paths, key=sort_sentinel2_bands)
        return paths

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        paths = self._load_paths(index)
        images = []
        for path in paths:
            # Bands are of different spatial resolutions
            # Resample to (120, 120)
            with rasterio.open(path) as dataset:
                array = dataset.read(
                    indexes=1,
                    out_shape=self.image_size,
                    out_dtype="int32",
                    resampling=Resampling.bilinear,
                )
                images.append(array)
        arrays: "np.typing.NDArray[np.int_]" = np.stack(images, axis=0)
        tensor = torch.from_numpy(arrays)
        return tensor

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target label
        """
        folder = self.folders[index]["s2"]
        path = glob.glob(os.path.join(folder, "*.json"))[0]
        with open(path) as f:
            labels = json.load(f)["labels"]

        # labels -> indices
        indices = [self.class2idx[label] for label in labels]
        target = torch.zeros(self.num_classes, dtype=torch.long)
        target[indices] = 1
        return target


if __name__ == "__main__":

    args = get_args()

    train_subset = BigEarthNetModified(root=args['data_root'])

    train_data, dataloader = load_data(data_root=args['data_root'])
    train_dict = next(iter(dataloader))
    print(f"Feature batch shape: {train_dict['image'].size()}")
    print(f"Labels batch shape: {train_dict['label'].size()}")
    img = train_dict['image'][0].squeeze()[1, :, :]
    label = train_dict['label'][0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
