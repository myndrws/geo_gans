# exploring the data to use in the GANs
# understanding effects of normalisation etc.

from load_data import load_data
from config import args
from matplotlib import pyplot as plt
import numpy as np

import torchvision.utils as vutils
from torchvision import transforms
from load_data import BigEarthNetModified
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import FiveCrop, Lambda, PILToTensor
import torch

_, dataloader = load_data(data_root=args.data_root,
                          batch_size=args.batch_size,
                          subset=True)

# only need to run this block once - otherwise in args
args.data_mean = torch.tensor(args.data_mean, dtype=torch.float64)
args.data_std = torch.tensor(args.data_std, dtype=torch.float64)
mean = args.data_mean
std = args.data_std

# if never run before, this is the code to get the mean
# and standard deviation of whole set - need to make sure
# not just read in a subset too.
# mean = 0.
# std = 0.
# nb_samples = 0.
# for data in dataloader:
#     ims = data["image"]
#     batch_samples = ims.size(0)
#     ims = ims.view(batch_samples, ims.size(1), -1)
#     mean += ims.mean(2).sum(0)
#     std += ims.std(2).sum(0)
#     nb_samples += batch_samples
#
# mean /= nb_samples
# std /= nb_samples
#
# print(mean, std)

real_batch = next(iter(dataloader))
real_batch_cpu = real_batch['image'].to("cpu").float()

#################################################
# visualise data for viewing not modelling
#################################################

# here we normalise by the max pixel value to view the data
# but this is not how data is passed in to the modelling; instead
# it's normalised by mean and std. deviation, which makes it look strange
# to view but would make it easier for the machine to find distinguishing features.
def visualise_batch(batch, with_max_normalisaton=True):
    plt.figure(figsize=(15, 15), frameon=False)
    plt.axis("off")
    plt.title("Real Images")
    normalising_constants = batch.amax()
    pltimgs = batch / normalising_constants if with_max_normalisaton else batch
    grid_pltimgs = vutils.make_grid(pltimgs)
    t_grid = np.transpose(grid_pltimgs)
    plt.imshow(t_grid)
    plt.show()

visualise_batch(batch=real_batch_cpu, with_max_normalisaton=True)

########################################################
# show how a single image would be seen by the model
########################################################

# normalising the data with the mean and standard deviation
# rather than the max - this is to highlight the relevant pixels
# in the image rather than display it nicely - two different things.
# then add five crop to this - this is so that all the images will be
# 64 * 64 and we get the most data we can from them

# select a sample from the batch
img = real_batch_cpu[0, :, :, :]

transform_norm = transforms.Compose([
    transforms.Normalize(mean, std)
])

# get normalized image
img_normalized = transform_norm(img)

# permute to match the desired dimension format
img_normalized = img_normalized.permute(2, 1, 0).numpy()
plt.imshow(img_normalized)
plt.show()

# add fivecropping to the transforms
transform_norm_crop = transforms.Compose([
    transforms.Normalize(mean, std),
    transforms.FiveCrop(size=(64, 64))
])

# transform image
(top_left, top_right, bottom_left, bottom_right, center) = transform_norm_crop(img)

# permute to match the desired dimension format
for image in [top_left, top_right, bottom_left, bottom_right, center]:
    plt.imshow(image.permute(2, 1, 0).numpy())
    plt.show()

###########################################################
# apply transforms to a whole batch as POC for dataloader
###########################################################

# first try out dataset and dataloading separately
# define a transforms on whole set
im_transforms = transforms.Compose([
    transforms.Normalize(mean, std),
    FiveCrop(size=(64, 64)),  # this is a list of PIL Images
    Lambda(lambda crops: torch.stack(crops))  # returns a 4D tensor
])

# load in with transforms
train_data_full = BigEarthNetModified(root=args.data_root,
                                      split="train",
                                      n_channels=args.n_channels,
                                      peat_only=True,
                                      transforms=im_transforms)

# load in only a subset
train_data_subset = Subset(train_data_full, list(range(args.batch_size)))
dataloader2 = DataLoader(train_data_subset, batch_size=args.batch_size, shuffle=True)

real_batch = next(iter(dataloader2))

# from fivecrop documentation - can replicate within modelling
input = real_batch["image"]  # input is a 5d tensor, target is 2d
bs, ncrops, c, h, w = input.size()
result = input.view(-1, c, h, w) # fuse batch size and ncrops
result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops

real_batch_cpu = result.to("cpu").float()
visualise_batch(batch=real_batch_cpu, with_max_normalisaton=True)
