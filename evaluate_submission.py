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
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import argparse 
import gdown
import os
import time 
import csv
from contextlib import contextmanager
import tempfile
import shutil

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')


@contextmanager
def in_temp_submission_dir(submission_file):
    """Context manager that copies submission to temp dir and switches to it."""
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_submission = os.path.join(temp_dir, os.path.basename(submission_file))
        shutil.copy(submission_file, temp_submission)
        os.chdir(temp_dir)
        try:
            yield temp_submission
        finally:
            os.chdir(original_dir)


def get_submission(submission_file, device='cpu'):
    """Safely loads only SubmissionInterface, skipping executable code."""
    import ast
    
    with open(submission_file, 'r') as f:
        # python file may contain Jupyter shell commands and magics (!, %, %%); filter them
        lines = [line for line in f.readlines()  if not line.strip().startswith(('!', '%', '%%')) 
            and not ('google.colab' in line and 'import' in line)]
        source = ''.join(lines)
    
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom))]
    safe_module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(safe_module)
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'load_file': load_file, 'gdown': gdown, 'os': os}
    exec(compile(safe_module, submission_file, 'exec'), namespace)
    return namespace['SubmissionInterface']()



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



def setup_resnet(resnet_weights_file = 'downloaded_resnet.safetensors'): 
    deep_resnet = ResNet(blocks_per_level=4)
    deep_resnet.eval()
    if not os.path.exists(resnet_weights_file):
        print("Downloading resnet weights..") 
        shareable_link = "https://drive.google.com/file/d/1kW_wnq-J_41_ESyQUX1PJD9-vvbWbCQ8/view?usp=sharing"
        gdown.download(shareable_link, resnet_weights_file, quiet=False, fuzzy=True)
    deep_resnet.load_state_dict(load_file(resnet_weights_file))
    return deep_resnet


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print("Using device:", device)

    parser = argparse.ArgumentParser(description="Evaluate a submission model.")
    # we'll do teamname_submission.py.  so "sample" isn't really a team, it's just an example.
    parser.add_argument('--submission', type=str, default='sample_submission.py', help='Path to the submission file.')
    args = parser.parse_args()

    submission_file = args.submission
    print(f"Evaluating submission from file: {submission_file}")
    
    # instantiate the submission class
    with in_temp_submission_dir(submission_file) as temp_file:
        submission = get_submission(temp_file).to(device)
        submission.vae.eval()
        submission.flow_model.eval()

    metrics = {}
    try: 
        metrics['team'] = submission.info['team']
    except Exception as e:
        print("WARNING: No team name in submission.info, trying filename instead.")
        metrics['team'] = submission_file.split('_')[0]
    print(f"Team name: {metrics['team']}")
    metrics['names'] = submission.info.get('names', 'N/A')
    print(f"Team members: {metrics['names']}\n")



    # total number of parameters across vae and flow model 
    total_params = sum(p.numel() for p in submission.vae.parameters()) + sum(p.numel() for p in submission.flow_model.parameters())
    print(f"Total parameters in VAE + Flow Model: {total_params:,}\n")
    metrics['total_params'] = total_params

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


    # MSE of reconstructions
    mse = F.mse_loss(recon, images.view(-1, 28, 28))
    metrics['mse'] = mse.item()
    print(f"Reconstruction MSE (lower is better) ↓: {mse.item():.4f}") 

    from skimage.metrics import structural_similarity as ssim
    # SSIM of reconstructions
    ssim_total = 0.0
    for i in range(recon.shape[0]):
        ssim_total += ssim(recon[i].cpu().numpy(), images[i].view(28, 28).cpu().numpy(), data_range=1.0)
    ssim_avg = ssim_total / recon.shape[0]
    print(f"Reconstruction SSIM (higher is better) ↑: {ssim_avg:.4f}")
    metrics['ssim'] = ssim_avg

    # flow model generation
    gen_batch_size = min(2048, len(mnist_test))  # don't exceed test set size
    print(f"\nGenerating {gen_batch_size} samples from flow model...")
    with torch.no_grad():
        start = time.time()
        samples = submission.generate_samples(n_samples=gen_batch_size).to(device)
        if samples.max() > 1.0 or samples.min() < 0.0:   # note: me having to do this will incur for you a small time penalty ;-) 
            print("WARNING: generated samples out of [0,1] range, applying sigmoid.")
            samples = torch.sigmoid(samples)  # ensure in [0,1] range
        gen_time = time.time() - start
        metrics['gen_time'] = gen_time
        metrics['time_per_sample'] = gen_time / gen_batch_size
        print(f"Sample generation took {gen_time:.6f} seconds, {metrics['time_per_sample']*1000:.4f} ms/sample.")


    # evaluate generated samples...

    # Use pretrained "deep" ResNet from lesson 06b to evaluate generated images 
    deep_resnet = setup_resnet().to(device)

    samples = samples.unsqueeze(1)  # add channel dim
    logits = deep_resnet(samples)
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # add small value to avoid log(0)
    avg_entropy = entropy.mean().item()
    print(f"Avg. predictive entropy of gen'd samples (lower is better) ↓: {avg_entropy:.4f}")
    metrics['entropy'] = avg_entropy  


    # more stuff... (work in progress)  

    # statistical comparisons between distibutions of ground truth images & gen'd images:
    # get mnist test images, use random choice but without duplicate indices
    random_indices = np.random.choice(len(mnist_test), size=gen_batch_size, replace=False)
    real_images = torch.stack([mnist_test[i][0] for i in random_indices]).to(device)

    real_logits = deep_resnet(real_images)
    real_probs = F.softmax(real_logits, dim=1)
    real_entropy = -torch.sum(real_probs * torch.log(real_probs + 1e-8), dim=1)
    avg_real_entropy = real_entropy.mean().item()
    print(f"Avg. predictive entropy of real  samples (lower is better) ↓: {avg_real_entropy:.4f}")
    metrics['real_entropy'] = avg_real_entropy




    #  mean, std, kl divergence, wasserstein/sinkhorn distance, ...
    real_mean = real_images.mean().item()
    real_std = real_images.std().item()
    gen_mean = samples.mean().item()
    gen_std = samples.std().item()
    metrics['real_mean'] = real_mean
    metrics['real_std'] = real_std
    metrics['gen_mean'] = gen_mean
    metrics['gen_std'] = gen_std
    print(f"Gen'd images - mean: {gen_mean:.4f}, std: {gen_std:.4f}")
    print(f"Real  images - mean: {real_mean:.4f}, std: {real_std:.4f}")

    # Diversity: Compare predicted class distributions (real vs generated)
    
    real_preds = real_logits.argmax(dim=1)  # predicted classes
    gen_preds = logits.argmax(dim=1)

    # Count how many of each digit (0-9)
    real_class_counts = torch.bincount(real_preds, minlength=10).float() / len(real_preds)
    gen_class_counts = torch.bincount(gen_preds, minlength=10).float() / len(gen_preds)
    #print(f"Real class distribution: {real_class_counts.cpu().numpy()}")
    #print(f"Gen  class distribution: {gen_class_counts.cpu().numpy()}")
    print(f"Class Distribution Comparison:\n{'   Real':20s} {'  Generated':15s}")
    for i in range(10):
        real_val = real_class_counts[i].item()
        gen_val = gen_class_counts[i].item()
        real_bar = '█' * int(real_val * 50)
        gen_bar = '█' * int(gen_val * 50)
        print(f"{i}: {real_bar:10s} ({real_val:.3f})  {gen_bar:10s} ({gen_val:.3f})")

    kl_div = F.kl_div(gen_class_counts.log(), real_class_counts, reduction='sum').item()
    metrics['kl_div_classes'] = kl_div
    print(f"KL Divergence of class distributions (lower is better) ↓: {kl_div:.4f}\n")

    # Confidence comparison: how confident is the classifier on real vs generated?
    real_max_probs = real_probs.max(dim=1)[0]  # max prob for each image
    gen_max_probs = probs.max(dim=1)[0]

    real_avg_conf = real_max_probs.mean().item()
    gen_avg_conf = gen_max_probs.mean().item()

    metrics['real_confidence'] = real_avg_conf
    metrics['gen_confidence'] = gen_avg_conf
    print(f"Gen  images - avg classifier confidence (higher is better) ↑: {gen_avg_conf:.4f}")
    print(f"Real images - avg classifier confidence (higher is better) ↑: {real_avg_conf:.4f}")

    # FID scores (but FID is technically for ImageNet not MNIST, so maybe not the best metric here)

    from scipy.stats import wasserstein_distance

    # # Flatten images and compute 1D Wasserstein.... Nah! too slow
    # real_flat = real_images.flatten().cpu().numpy()
    # gen_flat = samples.flatten().cpu().numpy()
    # w_dist = wasserstein_distance(real_flat, gen_flat)
    # print(f"Wasserstein distance (lower is better) ↓: {w_dist}")


    print("\nSummary of main metrics:")
    print(f"{'Team':<15} | {'Params ↓':>10} {'GenT(s)↓':>10} {'ms/samp↓':>10} | {'MSE ↓':>8} {'SSIM ↑':>8} | {'Entropy↓':>10} {'KL Div↓':>10} {'Conf ↑':>8}")
    print("-" * 110)
    parts = [ 
        f"{metrics['team']:<15}",
        f"{metrics['total_params']:>10,} {metrics['gen_time']:>10.4f} {metrics['time_per_sample']*1000:>10.4f}",
        f"{metrics['mse']:>8.4f} {metrics['ssim']:>8.4f}",
        f"{metrics['entropy']:>10.4f} {metrics['kl_div_classes']:>10.4f} {metrics['gen_confidence']:>8.4f}"
    ]
    print(" | ".join(parts))

    time_stamp = time.strftime("%Y%m%d-%H%M%S")

    # append to submissions_full.csv file
    csv_file = 'submissions_full.csv'
    print("Appending results to", csv_file)
    write_header = not os.path.exists(csv_file)
    # with open(csv_file, 'a') as f:
    #     if write_header:
    #         f.write("team,names,total_params,gen_time,time_per_sample,mse,ssim,entropy,kl_div_classes,gen_confidence,real_confidence,real_entropy,real_mean,real_std,gen_mean,gen_std,time_stamp\n")
    #     f.write(f"{metrics['team']},{metrics['names']},{metrics['total_params']},{metrics['gen_time']:.6f},{metrics['time_per_sample']:.6f},{metrics['mse']:.6f},{metrics['ssim']:.6f},{metrics['entropy']:.6f},{metrics['kl_div_classes']:.6f},{metrics['gen_confidence']:.6f},{metrics['real_confidence']:.6f},{metrics['real_entropy']:.6f},{metrics['real_mean']:.6f},{metrics['real_std']:.6f},{metrics['gen_mean']:.6f},{metrics['gen_std']:.6f},{time_stamp}\n")
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0: 
            writer.writerow(['team','names','total_params','gen_time','time_per_sample','mse','ssim','entropy','kl_div_classes','gen_confidence','real_confidence','real_entropy','real_mean','real_std','gen_mean','gen_std','time_stamp'])
        writer.writerow([metrics['team'], metrics['names'], metrics['total_params'], metrics['gen_time'], metrics['time_per_sample'], metrics['mse'], metrics['ssim'], metrics['entropy'], metrics['kl_div_classes'], metrics['gen_confidence'], metrics['real_confidence'], metrics['real_entropy'], metrics['real_mean'], metrics['real_std'], metrics['gen_mean'], metrics['gen_std'], time_stamp])
