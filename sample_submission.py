## Sample submission file for the VAE + Flow leaderboard challenge
## Author: Scott H. Hawley, Oct 6 2025 

# NOTE: This is a basic baseline submission (latent_dim=3, simple MLP)
# You should be able to significantly outperform these metrics!

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
import gdown
import os


class SimpleVAEModel(nn.Module):
    def __init__(self,
                 latent_dim=3,    # dimensionality of the latent space. bigger=less compression, better reconstruction
                 n_hid=[256,64],  # simple
                 act = nn.LeakyReLU,
                 ):
        super().__init__()
        self.encoder = nn.Sequential(
                nn.Linear(28 * 28, n_hid[0]),act(),
                nn.Linear(n_hid[0], n_hid[1]), act(),
                nn.Linear(n_hid[1], latent_dim*2), # *2 b/c mu, log_var
                )
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim, n_hid[1]), act(),
                nn.Linear(n_hid[1], n_hid[0]),  act(),
                nn.Linear(n_hid[0], 28 * 28),
                )
        self.latent_dim, self.n_hid, self.act = latent_dim, n_hid, act # save for possible use later

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten to (batch, 784)
        z = self.encoder(x)
        i_half = z.shape[-1]//2
        mu, log_var = z[:,:i_half],  z[:,i_half:]
        z_hat = mu + torch.randn_like(mu) * torch.exp(0.5*log_var)
        x_hat = self.decoder(z_hat)
        x_hat = x_hat.view(x_hat.size(0), 1, 28, 28)  # Reshape back for BCE loss
        return z, x_hat, mu, log_var, z_hat
    

class SimpleFlowModel(nn.Module):
    def __init__(self, latent_dim=3, n_hidden=32, n_layers=3, act=nn.LeakyReLU):
        super(SimpleFlowModel, self).__init__()
        self.latent_dim = latent_dim
        self.layers = nn.Sequential(
            nn.Linear(latent_dim+1, n_hidden), act(),
            *[nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers-1)],
            nn.Linear(n_hidden, latent_dim),)
    
    def forward(self, x, t, act=F.gelu):
        t = t.expand(x.size(0), 1)  # Ensure t has the correct dimensions
        x = torch.cat([x, t], dim=1)
        return self.layers(x)




### these next two are identical to blog post
@torch.no_grad()
def fwd_euler_step(model, current_points, current_t, dt):
    velocity = model(current_points, current_t)
    return current_points + velocity * dt 

@torch.no_grad()
def integrate_path(model, initial_points, step_fn=fwd_euler_step, n_steps=100,
                   save_trajectories=False, warp_fn=None):
    """this 'sampling' routine is primarily used for visualization."""
    device = next(model.parameters()).device
    current_points = initial_points.clone()
    ts =  torch.linspace(0,1,n_steps).to(device)
    if warp_fn: ts = warp_fn(ts)
    if save_trajectories: trajectories = [current_points]    
    for i in range(len(ts)-1):
        current_points = step_fn(model, current_points, ts[i], ts[i+1]-ts[i])
        if save_trajectories: trajectories.append(current_points)
    if save_trajectories: return current_points, torch.stack(trajectories).cpu()
    return current_points 
#####




class SubmissionInterface(nn.Module):
    """All teams must implement this for automated evaluation.
    When you subclass/implement these methods, replace the NotImplementedError."""
    
    def __init__(self):
        super().__init__()
        self.device = 'cpu' 
        self.latent_dim = 3
        self.load_vae()
        self.load_flow_model()
    
    def load_vae(self):
        """this completely specifies the vae model including configuration parameters,
           downloads/mounts the weights from Google Drive, automatically loads weights"""
        self.vae = SimpleVAEModel(latent_dim=self.latent_dim)
        vae_weights_file = 'downloaded_vae.safetensors'
        if not os.path.exists(vae_weights_file):
            safetensors_link = "https://drive.google.com/file/d/1N4VS3HKBrXnuQhiud1ruMTFZ5jbhsrIn/view?usp=sharing"
            gdown.download(safetensors_link, vae_weights_file, quiet=False, fuzzy=True)
        self.vae.load_state_dict(load_file(vae_weights_file))
        
    def load_flow_model(self):
        """this completely specifies the flow model including configuration parameters,
           downloads/mounts the weights from Google Drive, automatically loads weights"""
        self.flow_model = SimpleFlowModel(latent_dim=self.latent_dim)
        flow_weights_file = 'downloaded_flow.safetensors'
        if not os.path.exists(flow_weights_file):
            safetensors_link = "https://drive.google.com/file/d/13hlolKEc1QB6wA5M_sSgH8wfKR35fjc9/view?usp=sharing"
            gdown.download(safetensors_link, flow_weights_file, quiet=False, fuzzy=True)
        self.flow_model.load_state_dict(load_file(flow_weights_file))
    
    def generate_samples(self, n_samples:int, n_steps:int) -> torch.Tensor:
        z0 = torch.randn([n_samples, self.latent_dim]).to(self.device)
        z1 = integrate_path(self.flow_model, z0, n_steps=n_steps)
        gen_xhat = F.sigmoid(self.decode(z1).view(-1, 28, 28))
        return gen_xhat

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        # if your vae has linear layers, flatten first
        # if your vae has conv layers, comment out next line
        images = images.view(images.size(0), -1)  
        with torch.no_grad():
            z = self.vae.encoder(images.to(self.device))
            mu = z[:, :self.latent_dim]  # return only first half (mu)
            return mu
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decoder(latents)

    def to(self, device):
        self.device = device 
        self.vae.to(self.device)
        self.flow_model.to(self.device)
        return self 
    

# Sample usage: 
# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# mysub = SubmissionInterface().to(device) # loads vae and flow models
# xhat_gen = mysub.generate_samples(n_samples=10, n_steps=100)
