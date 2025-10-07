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
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import argparse 

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

    # Evaluate reconstructions with MSE and SSIM
    mse = F.mse_loss(recon, images.view(-1, 28, 28))
    metrics['mse'] = mse.item()
    print(f"Reconstruction MSE ↓: {mse.item()}") 
    ssim_total = 0.0
    for i in range(recon.shape[0]):
        ssim_total += ssim(recon[i].cpu().numpy(), images[i].view(28, 28).cpu().numpy(), data_range=1.0)
    ssim_avg = ssim_total / recon.shape[0]
    print(f"Reconstruction SSIM ↑: {ssim_avg}")
    metrics['ssim'] = ssim_avg

    # flow model generation
    with torch.no_grad():
        samples = submission.generate_samples(n_samples=batch_size, n_steps=100)

    # TODO: evaluate generated samples in various ways - work in progress! 

