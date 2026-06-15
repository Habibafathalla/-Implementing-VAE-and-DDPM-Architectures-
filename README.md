# Generative Models on Fashion-MNIST: DDPM & VAE

A PyTorch implementation of two generative models trained on the Fashion-MNIST dataset:

- **DDPM** — Denoising Diffusion Probabilistic Model with a U-Net backbone and EMA
- **VAE** — Variational Autoencoder

Both models are evaluated using **FID** (Fréchet Inception Distance).

---



## Project Structure

```
.
├── ddpm-fashion-mnist.ipynb   # DDPM training, sampling, and evaluation
├── fashion-mnist-vae.ipynb    # VAE training and evaluation
└── Report documenting.md
```


## Environment

The notebooks were developed and run on **Kaggle** (Python 3, CUDA 12.1, T4/P100/A100 GPU).

### Python Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install einops timm tqdm scipy
```




