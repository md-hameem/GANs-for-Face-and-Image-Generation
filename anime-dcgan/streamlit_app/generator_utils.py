import sys
import os
import torch
from torchvision.utils import make_grid

# Add root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dcgan import Generator


def load_generator(model_path: str, nz: int = 100, ngf: int = 64, nc: int = 3) -> torch.nn.Module:
    """Load the trained Generator model from a .pth file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator(nz=nz, ngf=ngf, nc=nc).to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    return netG



def generate_images(model: torch.nn.Module, latent_vectors: torch.Tensor) -> torch.Tensor:
    """Generate images from the Generator given latent vectors z."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_vectors = latent_vectors.to(device)
    with torch.no_grad():
        fake_images = model(latent_vectors).detach().cpu()
    return fake_images
