# lots of this code taken from the PyTorch DCGAN tutorial
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.animation as animation
from IPython.display import HTML

from config import get_args
from load_data import load_data
from model_architectures import Generator, Discriminator, weights_init


def main(args):

    # print args to console
    print(f"Using args: {args}")
    print(datetime.now().strftime("%H:%M:%S"))

    # Set random seed for reproducibility
    manualSeed = 42
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # load data
    train_data, dataloader = load_data(data_root=args['data_root'],
                                       batch_size=args['batch_size'],
                                       subset=args['subset_data'])

    # set device based on args
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args["n_gpus"] > 0) else "cpu")

    # Create the generator
    netG = Generator(args['n_gpus'],
                     args['z_gen_dims'],
                     args['fmap_gen_dims'],
                     args['n_channels']).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (args['n_gpus'] > 1):
        netG = nn.DataParallel(netG, list(range(args['n_gpus'])))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = Discriminator(args['n_gpus'],
                         args['fmap_disc_dims'],
                         args['n_channels']).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (args['n_gpus'] > 1):
        netD = nn.DataParallel(netD, list(range(args['n_gpus'])))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, args['z_gen_dims'], 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args['lr'], betas=(args['adam_b1'], 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args['lr'], betas=(args['adam_b1'], 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # Training Loop
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(args['epochs']):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data['image'].to(device).float()
            b_size = real_cpu.size(0)  # size of the first dimension
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args['z_gen_dims'], 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args['epochs'], i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == args['epochs'] - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                # fake is sliced for visualisation using 16 images and rgb channels
                img_list.append(vutils.make_grid(fake[:16, 1:4, :, :], normalize=True))

            iters += 1

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("results/losses.png")
    plt.show()

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    # sliced for visualised using 16 images and rgb channels
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    pltimgs = real_batch['image'].to(device).float()[:16, 1:4, :, :]
    plt.imshow(np.transpose(vutils.make_grid(pltimgs, normalize=True).cpu()))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1]))
    plt.savefig("results/fake_vs_real_vis.png")
    plt.show()


if __name__ == "__main__":
    # get args
    args = get_args(bash_parser=False)

    # run main
    main(args)
