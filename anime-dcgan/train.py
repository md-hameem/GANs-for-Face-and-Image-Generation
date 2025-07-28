import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from models.dcgan import Generator, Discriminator
from tqdm import tqdm
import multiprocessing

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
image_size = 64
nz = 100         # Latent vector size
ngf = 64         # Generator feature maps
ndf = 64         # Discriminator feature maps
nc = 3           # Number of channels (RGB)
batch_size = 128
epochs = 20
lr = 0.0002
beta1 = 0.5

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

dataset = ImageFolder(root="data/", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Models
netG = Generator(nz, ngf, nc).to(device)
netD = Discriminator(nc, ndf).to(device)

# Loss and Optimizers
criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Fixed noise for visualizing progress
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Training Loop
    print("Starting Training...")
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for i, (real_images, _) in enumerate(loop):
            real_images = real_images.to(device)
            b_size = real_images.size(0)
            real_labels = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
            fake_labels = torch.full((b_size,), 0.0, dtype=torch.float, device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            netD.zero_grad()
            output = netD(real_images)
            lossD_real = criterion(output, real_labels)

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            output = netD(fake_images.detach())
            lossD_fake = criterion(output, fake_labels)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            # -----------------
            #  Train Generator
            # -----------------
            netG.zero_grad()
            output = netD(fake_images)
            lossG = criterion(output, real_labels)
            lossG.backward()
            optimizerG.step()

            loop.set_postfix(lossD=lossD.item(), lossG=lossG.item())

        # Save sample generated images
        with torch.no_grad():
            fake_samples = netG(fixed_noise).detach().cpu()
        save_image(fake_samples, f"outputs/fake_epoch_{epoch+1:03d}.png", normalize=True)

        # Optional: Save model checkpoints
        torch.save(netG.state_dict(), f"outputs/netG_epoch_{epoch+1}.pth")
        torch.save(netD.state_dict(), f"outputs/netD_epoch_{epoch+1}.pth")

    print("Training finished.")
