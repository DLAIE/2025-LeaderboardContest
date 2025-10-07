#! /usr/bin/env python3

# Evaluate the submission model on various metrics
# Work in progress! 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import importlib.util
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from safetensors.torch import load_file
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import argparse 
import gdown
import os


DATASET_INFO = {
    'name': 'MNIST',
    'num_classes': 10,
    'input_size': (28, 28),
    'num_channels': 1,
}

# For evaluating generated images: it's the code from the ResNet lesson! 
# https://github.com/drscotthawley/DLAIE/blob/main/2025/06b_SkipsResNetsUNets.ipynb
ACTIVATION = nn.SiLU()
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_skip=True, use_bn=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.use_skip = use_skip

    def forward(self, x):
        if self.use_skip: x0 = x
        out = ACTIVATION(self.bn1(self.conv1(x)))
        out = F.dropout(out, 0.4, training=self.training)
        out = self.bn2(self.conv2(out))
        if self.use_skip: out = out + x0
        return ACTIVATION(out)
    
class FlexibleCNN(nn.Module):
    def __init__(self, dataset_info=DATASET_INFO, base_channels=32, blocks_per_level=4, use_skips=False, use_bn=True):
        super().__init__()

        # Initial conv
        self.conv1 = nn.Conv2d(dataset_info['num_channels'], base_channels, 3, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(base_channels) if use_bn else nn.Identity()

        # Build levels dynamically
        self.levels = nn.ModuleList()
        channels = [base_channels, base_channels*2, base_channels*4]

        for level_idx, ch in enumerate(channels):
            level_blocks = nn.ModuleList()
            for block_idx in range(blocks_per_level):
                level_blocks.append(ResidualBlock(ch, use_skips, use_bn))
            self.levels.append(level_blocks)

        # Transition layers
        self.transitions = nn.ModuleList([
            nn.Conv2d(base_channels, base_channels*2, 1, bias=not use_bn),
            nn.Conv2d(base_channels*2, base_channels*4, 1, bias=not use_bn)
        ])

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels*4, dataset_info['num_classes'])

    def forward(self, x):
        x = ACTIVATION(self.bn1(self.conv1(x)))

        # Level 1 blocks
        for block in self.levels[0]:
            x = block(x)

        # Downsample + transition + Level 2 blocks
        x = F.avg_pool2d(x, 2)
        x = self.transitions[0](x)
        for block in self.levels[1]:
            x = block(x)

        # Downsample + transition + Level 3 blocks
        x = F.avg_pool2d(x, 2)
        x = self.transitions[1](x)
        for block in self.levels[2]:
            x = block(x)

        x = self.global_avg_pool(x)
        return self.fc(x.flatten(start_dim=1))

class ResNet(FlexibleCNN):
    def __init__(self, **kwargs):
        super().__init__(use_skips=True, use_bn=True, **kwargs)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a submission model.")
    parser.add_argument('--submission', type=str, default='sample_submission.py', help='Path to the submission file.')
    args = parser.parse_args()

    submission_file = args.submission
    print(f"Evaluating submission from file: {submission_file}")

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print("Using device:", device)

    # read in the submission file as a module
    spec = importlib.util.spec_from_file_location("submission", submission_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


    # instantiate the submission class
    submission = module.SubmissionInterface().to(device)
    submission.vae.eval()
    submission.flow_model.eval()


    # encode and decode testing data
    with torch.no_grad():
        batch_size = 256
        mnist_test = MNIST(root='./data', train=False, download=True, transform=T.ToTensor())
        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
        data_iter = iter(test_loader)
        images, labels = next(data_iter)
        images = images.to(device)
        recon = submission.decode(submission.encode(images)).view(-1, 28, 28).detach()
        if recon.max() > 1.0 or recon.min() < 0.0:
            print("WARNING: reconstructions out of [0,1] range, applying sigmoid.")
            recon = torch.sigmoid(recon)  # ensure in [0,1] range
        images, recon = images.cpu(), recon.cpu() # easier this way

    # what metrics will we use... (work in progress)
    metrics = {}

    # MSE of reconstructions
    mse = F.mse_loss(recon, images.view(-1, 28, 28))
    metrics['mse'] = mse.item()
    print(f"Reconstruction MSE (lower is better) ↓: {mse.item()}") 

    # SSIM of reconstructions
    ssim_total = 0.0
    for i in range(recon.shape[0]):
        ssim_total += ssim(recon[i].cpu().numpy(), images[i].view(28, 28).cpu().numpy(), data_range=1.0)
    ssim_avg = ssim_total / recon.shape[0]
    print(f"Reconstruction SSIM (higher is better) ↑: {ssim_avg}")
    metrics['ssim'] = ssim_avg

    # flow model generation
    gen_batch_size = 512 # bigger is better for statistical metrics
    with torch.no_grad():
        print("Generating samples...")
        samples = submission.generate_samples(n_samples=gen_batch_size, n_steps=100).to(device)
        if samples.max() > 1.0 or samples.min() < 0.0:
            print("WARNING: generated samples out of [0,1] range, applying sigmoid.")
            samples = torch.sigmoid(samples)  # ensure in [0,1] range

    # evaluate generated samples...

    # Use pretrained "deep" ResNet from lesson 06b to evaluate generated images 
    deep_resnet = ResNet(blocks_per_level=4).to(device)
    resnet_weights_file = 'downloaded_resnet.safetensors'
    if not os.path.exists(resnet_weights_file):
        shareable_link = "https://drive.google.com/file/d/1kW_wnq-J_41_ESyQUX1PJD9-vvbWbCQ8/view?usp=sharing"
        gdown.download(shareable_link, resnet_weights_file, quiet=False, fuzzy=True)
    deep_resnet.load_state_dict(load_file(resnet_weights_file))
    samples = samples.unsqueeze(1)  # add channel dim
    print("samples.shape:", samples.shape)
    logits = deep_resnet(samples)
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # add small value to avoid log(0)
    avg_entropy = entropy.mean().item()
    print(f"Avg. predictive entropy of generated samples (lower is better) ↓: {avg_entropy}")
    metrics['entropy'] = avg_entropy  

    # more stuff... (work in progress)  

    # statistical comparisons between distibutions of ground truth images & gen'd images:
    #  mean, std, kl divergence, wasserstein/sinkhorn distance, ...

    # FID scores (but FID is technically for ImageNet not MNIST, so maybe not the best metric here)

