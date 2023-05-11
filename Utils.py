# Utils.py
import os
import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.figure()
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
    
    
    
def save_tensor_images(image_tensor, path, num_images=1, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().to('cpu').view(-1, *size)
    image_grid2 = make_grid(image_unflat[:num_images], nrow=5)  



    image_grid2 = image_grid2.permute(1, 2, 0).numpy()
    im = Image.fromarray(image_grid2.astype(np.uint8))
    im.save(path)

    
def setup_logging(): 
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    os.makedirs(os.path.join("./results", 'fake'), exist_ok=True)
    os.makedirs(os.path.join("./results", 'real'), exist_ok=True)
    
    
    
def get_noise(n_samples, z_dim, device='cuda'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''

    return torch.randn(n_samples, z_dim, device=device)
    #Alternative: return torch.randn(n_samples, z_dim).to(device)

