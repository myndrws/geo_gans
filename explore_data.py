# exploring the data to use in the GANs
# understanding effects of normalisation etc.

from load_data import load_data
from config import args
from matplotlib import pyplot as plt
import numpy as np
import torchvision.utils as vutils

train_data, dataloader = load_data(data_root=args.data_root,
                                   batch_size=args.batch_size,
                                   subset=True)

real_batch = next(iter(dataloader))
real_batch_cpu = real_batch['image'].to("cpu").float()

#################################################
# visualise data for viewing not modelling
#################################################

# here we normalise by the max pixel value to view the data
# but this is not how data is passed in to the modelling; instead
# it's normalised by mean and std. deviation, which makes it look strange
# to view but would make it easier for the machine to find distinguishing features.
plt.figure(figsize=(15, 15), frameon=False)
plt.axis("off")
plt.title("Real Images")
normalising_constants = real_batch_cpu.amax()
pltimgs = real_batch_cpu / normalising_constants
grid_pltimgs = vutils.make_grid(pltimgs)
t_grid = np.transpose(grid_pltimgs)
plt.imshow(t_grid)
plt.show()

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
from torchvision import transforms

mean, std = img.mean([1, 2]), img.std([1, 2])
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

