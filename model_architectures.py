# This code is taken from the PyTorch DCGAN tutorial
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import torch.nn as nn

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, args):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(args.z_gen_dims, args.fmap_gen_dims * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.fmap_gen_dims * 8),
            nn.ReLU(True),
            # state size. (args.fmap_gen_dims*8) x 4 x 4
            nn.ConvTranspose2d(args.fmap_gen_dims * 8, args.fmap_gen_dims * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.fmap_gen_dims * 4),
            nn.ReLU(True),
            # state size. (args.fmap_gen_dims*4) x 8 x 8
            nn.ConvTranspose2d(args.fmap_gen_dims * 4, args.fmap_gen_dims * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.fmap_gen_dims * 2),
            nn.ReLU(True),
            # state size. (args.fmap_gen_dims*2) x 16 x 16
            nn.ConvTranspose2d(args.fmap_gen_dims * 2, args.fmap_gen_dims, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.fmap_gen_dims),
            nn.ReLU(True),
            # state size. (args.fmap_gen_dims) x 32 x 32
            nn.ConvTranspose2d(args.fmap_gen_dims, args.n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (args.n_channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu, args):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (args["n_channels"]) x 64 x 64
            nn.Conv2d(args.n_channels, args.fmap_disc_dims, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (args.fmap_disc_dims) x 32 x 32
            nn.Conv2d(args.fmap_disc_dims, args.fmap_disc_dims * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.fmap_disc_dims * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (args.fmap_disc_dims*2) x 16 x 16
            nn.Conv2d(args.fmap_disc_dims * 2, args.fmap_disc_dims * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.fmap_disc_dims * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (args.fmap_disc_dims*4) x 8 x 8
            nn.Conv2d(args.fmap_disc_dims * 4, args.fmap_disc_dims * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.fmap_disc_dims * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (args.fmap_disc_dims*8) x 4 x 4
            nn.Conv2d(args.fmap_disc_dims * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


if __name__ == "__main__":

    from torchinfo import summary
    from config import args

    # test input sizes
    test_d_size = (args.batch_size, args.n_channels, args.image_size, args.image_size)
    test_g_size = (args.batch_size, args.z_gen_dims, 1, 1)

    # Create the Discriminator
    netD = Discriminator(ngpu=0, args=args).to("cpu")
    netD.apply(weights_init)

    # print summary of model
    print(summary(netD, test_d_size))

    # Create the generator
    netG = Generator(ngpu=0, args=args).to("cpu")
    netG.apply(weights_init)

    # print summary of model
    print(summary(netG, test_g_size))