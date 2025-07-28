import os
import torch
import torchvision.utils as vutils
from models.dcgan import Generator

# Config
nz = 100             # Size of latent vector (must match training)
ngf = 64             # Generator feature map size (same as training)
nc = 3               # Number of channels (RGB)
num_images = 16      # Number of images to generate
output_path = "outputs/generated_faces.png"
checkpoint_path = "outputs/netG_epoch_100.pth"  # Change to your saved model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Generator
netG = Generator(nz, ngf, nc).to(device)
netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
netG.eval()

# Generate noise and images
noise = torch.randn(num_images, nz, 1, 1, device=device)
with torch.no_grad():
    fake_images = netG(noise).detach().cpu()

# Save as image grid
os.makedirs("outputs", exist_ok=True)
vutils.save_image(fake_images, output_path, normalize=True, nrow=4)

print(f"Generated images saved to {output_path}")
