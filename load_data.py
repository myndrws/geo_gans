
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchgeo.datasets import BigEarthNet

data_root = 'C:/Users/amy_c/Documents/Python/geo_gans/ben_data'
batch_size = 64

# load train data from sentinel 2
train_data = BigEarthNet(root=data_root,
                         split='train',
                         bands='s2',
                         num_classes=43,
                         transforms=None,
                         download=False)

dataloader = DataLoader(train_data,
                        batch_size=batch_size,
                        shuffle=True)

if __name__ == "__main__":

    train_dict = next(iter(dataloader))
    print(f"Feature batch shape: {train_dict['image'].size()}")
    print(f"Labels batch shape: {train_dict['label'].size()}")
    img = train_dict['image'][0].squeeze()[1,:,:]
    label = train_dict['label'][0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
