import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchgeo.datasets import BigEarthNet
from config import get_args


# load train data from sentinel 2
def load_data(data_root, batch_size=64, subset=False):
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


if __name__ == "__main__":
    args = get_args()
    train_data, dataloader = load_data(data_root=args['data_root'])
    train_dict = next(iter(dataloader))
    print(f"Feature batch shape: {train_dict['image'].size()}")
    print(f"Labels batch shape: {train_dict['label'].size()}")
    img = train_dict['image'][0].squeeze()[1, :, :]
    label = train_dict['label'][0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
