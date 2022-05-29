import os, os.path
import json
import shutil
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchgeo.datasets import BigEarthNet, stack_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler

data_root = 'C:/Users/amy_c/Documents/Python/geo_gans/ben_data'

# load train data from sentinel 2
train_data = BigEarthNet(root=data_root,
                         split='train',
                         bands='s2',
                         num_classes=43,
                         transforms=None,
                         download=False)

classes_wanted = {'Land principally occupied by agriculture, with significant areas of natural vegetation': 20,
                  'Mixed forest': 22,
                  'Moors and heathland': 23,
                  'Pastures': 27,
                  'Peatbogs': 28,
                  'Salt marshes': 34,
                  'Water courses': 42}

# for every folder in the bigearthnet data
# go through and open the json file
# if the label is not in the string dictionary
# delete the whole folder
# path joining version for other paths
relevant_dirs = []
DIR = 'ben_data/BigEarthNet-v1.0/'
for im_folder in os.listdir(DIR):
    mdjson_str = DIR + im_folder + "/" + im_folder + '_labels_metadata.json'
    with open(mdjson_str, 'r') as f:
        jsonfile = json.loads(f.read())
    for label in jsonfile["labels"]:
        if label in classes_wanted.keys():
            relevant_dirs.append(im_folder)
            break

for im_folder in os.listdir(DIR):
    if im_folder not in relevant_dirs:
        shutil.rmtree(DIR+im_folder, ignore_errors=True)
        print(f"Removed {DIR+im_folder}")
train_csv=pd.read_csv("ben_data/bigearthnet-train_old.csv", header=None)
print(len(train_csv))
train_csv = train_csv[train_csv[0].isin(relevant_dirs)]
print(len(train_csv))
train_csv.to_csv("ben_data/bigearthnet-train.csv")