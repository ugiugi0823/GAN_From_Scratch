#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import time 
from torch import nn
from tqdm.auto import tqdm

from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torch.utils.data import DataLoader


from Utils import *
from Model import *
from Loss import *
from Block import *

# Load MNIST dataset as tensors
dataloader = DataLoader(
    MNIST('.', download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True)






cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True # Whether the generator should be tested
gen_loss = False


gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)



setup_logging()

def train()
    for epoch in range(n_epochs):
      
        # Dataloader returns the batches
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
    
            # Flatten the batch of real images from the dataset
            real = real.view(cur_batch_size, -1).to(device)
    
            ### Update discriminator ###
            # Zero out the gradients before backpropagation
            disc_opt.zero_grad()
    
            # Calculate discriminator loss
            disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
    
            # Update gradients
            disc_loss.backward(retain_graph=True)
    
            # Update optimizer
            disc_opt.step()
    
            # For testing purposes, to keep track of the generator weights
            if test_generator:
                old_generator_weights = gen.gen[0][0].weight.detach().clone()
    
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
            gen_loss.backward()
            gen_opt.step()
            #### END CODE HERE ####
    
            # For testing purposes, to check that your code changes the generator weights
            if test_generator:
                assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
    
            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step
    
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step
    
            ### Visualization code ###
            if cur_step % display_step == 0 and cur_step > 0:
                
                print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
    
                # show_tensor_images
                show_tensor_images(fake)
                show_tensor_images(real)
    
                # save_tensor_images
                
                save_tensor_images(fake, f'/content/results/fake/fake_{cur_step}.png')
                save_tensor_images(real, f'/content/results/real/real_{cur_step}.png')
                
                
    
    
                
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1
            
            
if __name__ == '__main__':
    end = time.time() 
    
    train()
    print(time.time() -end)

