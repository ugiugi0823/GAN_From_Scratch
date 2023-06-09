# Train.py
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



def train():
    # Set your parameters
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 200
    z_dim = 64
    display_step = 500
    batch_size = 128
    lr = 0.00001
    device = 'cuda'

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

                
                print('fake',type(fake))
                print('fake',fake.shape)
                print('fake',fake.dtype)
                print(fake)

                print('real',type(real))
                print('real',real.shape)
                print('real',real.dtype)
                print(real)
    
                # show_tensor_images
                # show_tensor_images(fake)
                # show_tensor_images(real)
    
                # save_tensor_images
                
                save_tensor_images(fake, f'./results/fake/{cur_step}_fake.png')
                save_tensor_images(real, f'./results/real/{cur_step}_real.png')
                
                
    
    
                
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1
            
            
if __name__ == '__main__':
    setup_logging()
    end = time.time() 
    
    train()
    print(time.time() -end)

