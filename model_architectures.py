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
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 7, 1, 1, bias=False),  # altered to have ksize=7 padding=1
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 7, 2, 0, bias=False),  # altered to have ksize=7 and padding=0
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 7, 1, 0, bias=False),  # altered to have ksize=7
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


if __name__ == "__main__":

    from torchinfo import summary

    # fix some args
    nchannels = 12
    batch_size = 64
    im_size = 120
    noise_dims = 100
    discrim_dims = gen_dims = 64

    # test input sizes
    test_d_size = (batch_size, nchannels, im_size, im_size)
    test_g_size = (batch_size, noise_dims, 1, 1)

    # Create the Discriminator
    netD = Discriminator(ngpu=0,
                         ndf=discrim_dims,
                         nc=nchannels).to("cpu")
    netD.apply(weights_init)

    # print summary of model
    print(summary(netD, test_d_size))

    # Create the generator
    netG = Generator(ngpu=0,
                     nz=noise_dims,
                     ngf=gen_dims,
                     nc=nchannels).to("cpu")
    netG.apply(weights_init)

    # print summary of model
    print(summary(netG, test_g_size))