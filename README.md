# 2025-LeaderboardContest


*Deep Learning & AI Ethics, Fall 2025, Dr. Hawley, Belmont University*

# Synthesis Assignment: Latent Flow Matching Leaderboard Competition!

## Overview

It's time to put together everything we're learned so far, albeit in simplied, toy-model context.  

You are to build and evaluate a complete latent flow matching system combining a Variational Autoencoder (VAE) with a  (unconditional) flow matching model. This synthesis project integrates major concepts from the first part of the semester into a single end-to-end generative modeling pipeline.

## Objective

Create a system that generates MNIST-like images using flow matching in a latent space of a VAE. You train all models.

1. **VAE Component**: Compresses MNIST digits to a low-dimensional latent space and reconstructs them
2. **Flow Model Component**: Learns to generate samples in the latent space using flow matching
3. **End-to-End Generation**: Produces new digit images by sampling from the flow model and decoding through the VAE

## Team Structure

- Teams of 1 to 3 people. Only from our class. (Though see "Post-Competition Open Leaderboard" below)

## Technical Constraints

- **Dataset**: MNIST handwritten digits,  You may only train on the *training* subset. 
- **Latent Space**: Choose dimensionality from 3 to 49 (7x7). 
- **Framework**: PyTorch ("raw") or Lightning
- **Save/Load**: Must implement model checkpointing to Google Drive

## Evaluation Categories

Teams will be ranked across multiple categories using a ranking-sum system (lower total rank = better):

1. **VAE Reconstruction Quality**: MSE and/or SSIM between original and reconstructed images
2. **VAE Generation Consistency**: (Pretrained) Classifier confidence statistics on samples from latent space
3. **Flow Model Sample Quality**: Classifier evaluation of end-to-end generated images
4. **Flow Model Generative Diversity:** Genereate all digits, not just a few. 
5. **Model Efficiency**: Total number of parameters across VAE + flow model
6. **Training Performance**: Convergence speed and final loss values

You will perform your own analysis and document them as part of your results, yet the official leaderboard rankings will be computed using Dr. Hawley's measurement codes.

## Ranking System

- Each team receives a rank (1 = best, N = worst) for each category
- Final score = sum of ranks across all categories
- **Lowest total score wins**
- Ties broken by overall analysis quality

## Deliverables

1. **Working Code**: Jupyter notebook(s) with complete pipeline
2. **Documentation:** In the form of a README file, explaining how to execute all parts. 
3. **Model Weights**: Saved checkpoints for reproducibility, via shareable Google Drive link. **Use [safetensors](https://huggingface.co/docs/safetensors/en/index) format.**
4. **Results Analysis**: 
   - Performance across all evaluation metrics
   - Sample visualizations and latent space analysis
   - Discussion of architectural choices and trade-offs
5. **Generated Samples**: Representative outputs from your system
6. **Submission via GitHub Pull Request or Canvas (TBA):** For code, docs, and analysis. (You may use anonymized usernames if you want.) Format and Domain  Working on the system. Announced via update later.

## Timeline

- **Assignment Released**: Fri October 3
- **Submissions Begin**: Wed Oct 8 
- **Submissions End**: ~~Sat October 11 11:59 PM US Central time.~~  Sun October 12 5:30 PM US Central Time
- **Results Announced:** by Fri October 17.

## Resources

- Use LLMs for coding assistance
- Course materials on VAEs and flow matching
- Pre-trained MNIST classifier (from ConvNets or  ResNets lessons)
- Google Colab recommended for computational resources

## Further Rules

**You may not *train* on any images from the validation or testing subsets of MNIST**. You are on your honor to abide by this rule. As such, violation will count as **cheating** and a **violation of the University Honor Code**, and will be punished appropriately. (There are ways to check if you cheated.)

***Other than that, pretty much anything goes*,** though **see "Awards and Prizes" below** for certain incentives.You may use any model architecture you like (MLP, CNN, skip connections, Attention), and any trick you can think of: pretraining, data augmention, hyperparameter tuning, special optimization tricks,...all fine.  You don't have to stick purely to Colab for training, you can use whatever computer(s) you want, train for as long as you want.  Just make sure it runs on Colab too.

## Awards and Prizes

- **"Open Innovation" award**: "Anything goes" approach. Best rank overall.
- **"Sustainable AI" award**: MNIST-only data from scratch (no pretraining), with compute limits (must be entirely trainable on free Colab tier)
- **"Efficient Architecture" award**: Lowest parameter count while meeting quality thresholds

**Prizes include:**

- **One pair of WanB Signature AirPod Pros** that I've been waiting to use as a prize..
- Swag from the BDAIC, CoreWeave, and/or other organizations.
- / **TBD**.

## Tips for Success

- Start with a simple baseline and iterate
- Balance compression vs. reconstruction quality in VAE design
- Ensure your flow model trains stably
- Freeze your VAE, and pre-generate latent encodings of all images, train flow model on those.
- Generate diverse samples for robust evaluation
- Document your architectural decisions and trade-offs

**Remember**: This competition rewards well-rounded systems, not just optimization of individual metrics. Focus on understanding the trade-offs and building a coherent end-to-end pipeline.

## Post-Competition Open Leaderboard

While our official competition leaderboard will only be for class members (and a possible Honorary member), the official evaluation code and model weights will be published so that outside competitors may submit their results and codes via Pull Requests to an unofficial "self-reporting" leaderboard. (Up to some time like end of 2025).
