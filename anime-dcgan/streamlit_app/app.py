import streamlit as st
import torch
import torchvision.utils as vutils
from PIL import Image
import numpy as np
from generator_utils import load_generator, generate_images

st.set_page_config(page_title="Anime Face Generator", layout="centered")
st.title("🎭 Anime Face Generator (DCGAN)")
st.caption("Generate random anime faces using a PyTorch DCGAN model.")

# Sidebar
st.sidebar.header("⚙️ Settings")
num_faces = st.sidebar.slider("Number of Faces", 1, 32, 8)
seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, value=42)

# Load the generator model
netG = load_generator("outputs/netG_epoch_100.pth")

# Generate button
if st.button("🎨 Generate Faces"):
    torch.manual_seed(seed)
    z = torch.randn(num_faces, 100, 1, 1)
    images = generate_images(netG, z)

    # Grid display
    grid = vutils.make_grid(images, nrow=min(8, num_faces), normalize=True)
    npimg = grid.mul(255).add(0.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    st.image(Image.fromarray(npimg), caption=f"{num_faces} Generated Anime Faces")
