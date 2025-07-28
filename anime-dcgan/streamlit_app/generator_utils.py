import torch
from models.dcgan import Generator

def load_generator(checkpoint_path, nz=100, ngf=64, nc=3, device="cpu"):
    model = Generator(nz, ngf, nc).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def generate_images(model, noise):
    with torch.no_grad():
        fake_images = model(noise)
    return fake_images.cpu()
