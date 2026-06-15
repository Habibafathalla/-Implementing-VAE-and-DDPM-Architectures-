# Generative Modeling on Fashion-MNIST: VAE vs. DDPM

## Introduction

Generative modeling is the task of learning a data distribution and sampling new examples from it.Two families of models have dominated the landscape: **Variational Autoencoders (VAEs)**, introduced by Kingma & Welling (2013), and **Denoising Diffusion Probabilistic Models (DDPMs)**, popularized by Ho et al. (2020). Both can generate realistic images from noise, yet they differ fundamentally in how they learn and how they generate.

---

## Theoretical Background

### Variational Autoencoder (VAE)
**The core idea — compress, then recreate.**

VAE has two parts:
- An Encoder that takes an image and compresses it into a small list of numbers, In our case, this is a vector of just 128 numbers, compared to the 784 pixels in the original image.
- A Decoder that takes any list of 128 numbers and turns it back into an image.

![VAE Arch](blob:https://markdownviewer.pages.dev/305b42c6-1483-41e7-9298-cdf2d1d7e373)

**Reparameterization Trick**:
A plain autoencoder (without the "variational") would only learn to reconstruct images it has already seen. The "variational" part adds a crucial twist: instead of mapping each image to a single fixed point, the encoder maps it to a **small region** in the compressed space and **described it is  by a mean(μ) and a standard deviation(σ)**. During training, the model samples a random point from that region and decodes it.

This forces the compressed space to be **smooth and continuous**. **Points that are close together in the 128 number space will produce similar looking images*. More importantly, we can now generate new images by picking a random point in this latent space (sampling from a standard normal distribution) and decoding it.

**The model is trained with two goals simultaneously:**

**Reconstruction quality** : the decoded image should look like the original.
**KL regularization** : the compressed regions should stay close to the center of the space, not spread out chaotically.

These two objectives are in tension and must be balanced. Together they form the **ELBO (Evidence Lower Bound)**, the VAE's training objective :

>Evidence Lower Bound (ELBO):
```
ELBO = Reconstruction Quality  −  KL Penalty
 ```

- The first term is the **reconstruction loss** which describe how well the decoder recovers the input. 
- The second term is the **KL divergence** which is a regularizer that pushes the learned posterior toward the prior, ensuring the latent space is compact and continuous.


### Denoising Diffusion Probabilistic Model (DDPM)
**The core idea — destroy, then learn to undo.**

For DDPM, instead of compressing images, it learns to reverse a destruction process. It takes a clear photo and slowly adding static (random noise) to it, step by step, over 1000 steps. By the end, the image is pure random noise.
This destruction process is called the **forward process**, and it is fixed and mathematical (no learning needed).

![DDPM Arch](blob:https://markdownviewer.pages.dev/e26d3698-acf1-4ad6-8f19-64fe7b1bcef4)

#### What the model learns.

The neural network (a U-Net) learns to **reverse** this process: given a noisy image at step t, predict what noise was added so it can be subtracted. 

>mean squared error objective
 ```
L = || true noise  −  predicted noise ||²
 ```

#### How generation works:

To generate a new image, we start with pure random noise (as if we were at step 1000) and run the reverse process 1000 times. At each step, the U-Net looks at the current noisy image and the current step number, predicts the noise component, and subtracts a small amount of it. After 1000 steps, we have a clean, realistic image that the model has never seen before.

The key detail is that the U-Net receives the **step number as input**,this tells it "how much noise is currently in the image" so it can calibrate how aggressively to denoise.

| Dimension | VAE | DDPM |
|-----------|-----|------|
| **Encoding** | Learned encoder network | Fixed Gaussian noise schedule |
| **Latent space** | Low-dimensional (128-D), structured | Same dimensionality as data, T=1000 levels |
| **Generation speed** | Single forward pass through decoder | T=1000 sequential denoising steps |
| **Training objective** | ELBO (reconstruction + KL) | Simplified noise prediction (MSE) |
| **Sample quality** | Fast but often blurry | Slow but high fidelity |
| **Latent interpretability** | Continuous, interpolatable space | No compact latent space |
| **Mode coverage** | Risk of mode dropping via KL | Better diversity via stochastic sampling |
---

## Dataset

**Fashion-MNIST** (Xiao et al., 2017) is a drop-in replacement for MNIST consisting of 70,000 grayscale images (60,000 train / 10,000 test) at 28×28 pixels, representing 10 clothing categories:

| Class | Label |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

**Preprocessing:** Images were normalized to `[-1, 1]` using `transforms.Normalize((0.5,), (0.5,))`. Both models share this preprocessing. Batch size was 64 for both models.

---

## VAE: Architecture and Implementation

The VAE was built entirely from scratch in PyTorch. It is composed of three main parts that work in sequence: an Encoder that compresses the image, a Latent Sampling step that introduces controlled randomness, and a Decoder that reconstructs the image from the compressed representation.

### Encoder

The encoder is the "compression" part of the VAE. Its job is to look at a full input image and squeeze it down into a much smaller set of numbers — but instead of producing a single fixed point, it outputs two vectors: a mean **μ** and a log-variance **log σ²**. Together, these define a small region (a Gaussian "cloud") in the latent space rather than a single point. This is what makes the VAE generative rather than just compressive: the region concept forces the space to stay smooth and continuous, so nearby points decode into similar-looking images.

In the encoder  a Convolutional Neural Network (CNN) was used rather than flattening the image first, because CNNs preserve spatial structure — they understand that pixels near each other are related. Strided convolutions (stride=2) are used to progressively halve the spatial resolution at each level, achieving the compression:

```
Input: (B, 1, 28, 28)
  → Conv2d(1→64, k=3, p=1)           # (B, 64, 28, 28)
  → ResBlock(64)
  → Conv2d(64→128, k=3, s=2, p=1)    # (B, 128, 14, 14)  [stride-2 downsample]
  → ResBlock(128)
  → Conv2d(128→256, k=3, s=2, p=1)   # (B, 256, 7, 7)    [stride-2 downsample]
  → ResBlock(256)
  → Flatten                           # (B, 256×7×7 = 12544)
  → Linear(12544 → 128)               # μ
  → Linear(12544 → 128)               # log σ²
```

Each `ResBlock` is a small sub-network inserted between the convolution layers. It applies: GroupNorm → SiLU → Conv3×3 → GroupNorm → SiLU → Conv3×3, and then adds the input back to the output (a skip connection):

```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
    def forward(self, x):
        return x + self.block(x)
```

### Reparameterization Trick

We have `μ` and `log σ²` from the encoder, but we need to actually sample a latent vector z from the distribution N(μ, σ²) to pass to the decoder. The problem is that sampling is a random operation gradients cannot flow through it, so training by backpropagation would be impossible.

The **reparameterization trick** solves this by factoring out the randomness into a separate variable `ε` that has nothing to do with the network

```
z = μ + σ · ε,    ε ~ N(0, I)
```
Now the only random part is `ε`, which is just an external noise input. The network parameters (`μ` and `log σ²`) are connected to z through simple multiplication and addition both fully differentiable so that gradients can flow cleanly from `z` back through the encoder.


```python
def reparameterize(mu, log_var):
    std     = torch.exp(0.5 * log_var)
    epsilon = torch.randn_like(std)
    return mu + std * epsilon
```

### Decoder

What it does: The decoder is the **"reconstruction** part. It takes the sampled latent vector `z` (128 numbers) and works backwards, progressively expanding it back to a full 28×28 image. It is the symmetric mirror of the encoder: where the encoder used **stride convolutions**  to shrink the spatial size, the decoder uses **transposed convolutions(ConvTranspose2d)**  to expand it back.
```
Input: z ∈ (B, 128)
  → Linear(128 → 256×7×7)            # (B, 12544)
  → Reshape                           # (B, 256, 7, 7)
  → ResBlock(256)
  → ConvTranspose2d(256→128, k=4, s=2, p=1)   # (B, 128, 14, 14)
  → ResBlock(128)
  → ConvTranspose2d(128→64, k=4, s=2, p=1)    # (B, 64, 28, 28)
  → ResBlock(64)
  → Conv2d(64 → 1, k=3, p=1)         # (B, 1, 28, 28)
  → Tanh                              # Output in [-1, 1]
```
`Tanh` at the output ensures the decoded image lives in the same `[-1, 1]` space as the normalized input, making the reconstruction loss well-defined.

### Generation
 works by skipping the encoder entirely: we sample `z ~ N(0, I)` directly and pass it through the decoder. Because the KL loss during training forced the latent space to look like `N(0, I)`, random samples from this distribution decode into plausible clothing images.

### Loss Function

The VAE has two competing objectives:
- Reconstruction: The decoded image should look like the original. We measure this with **MSE (mean squared error)** between the input x and its reconstruction x̂.
- Organized latent space: The encoder's output distribution should stay close to the standard normal `N(0, I)`. This is the **KL divergence term**. Without it, the encoder could learn to map each image to a tiny, isolated region of latent space — the space would have no structure, and random sampling from it would produce garbage.
- 
Together they form ELBO:

```
L_VAE = L_recon + β · L_KL
```

```python
def vae_loss(x, x_hat, mu, log_var, beta=1.0):
    recon_loss = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    kl_loss    = -0.5 * torch.mean(
        torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    )
    return (recon_loss + beta * kl_loss), recon_loss, kl_loss
```

### Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 100 (with early stopping, patience=7) |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 2e-4 |
| LR scheduler | CosineAnnealingLR (T_max=100, η_min=1e-5) |
| Latent dimension | 128 |
| β (KL weight) | 1.0 |
| Gradient clipping | max_norm = 1.0 |

A **cosine annealing** learning rate schedule was used to allow the model to fine-tune in later epochs. **Early stopping** (patience = 7 epochs on validation loss) prevented overfitting.

---

## DDPM: Architecture and Implementation

The DDPM is made up of four main building blocks that each have a distinct, well-defined role: a **Noise Scheduler** that controls how much noise is added at each step, **Sinusoidal Time Embeddings** that tell the model where it is in the noise schedule, a **U-Net** that predicts the noise to remove, and a **ResBlock** inside the U-Net that is the fundamental repeating unit connecting all of the above.

### 5.1 Noise Scheduler

The scheduler is the blueprint for the entire **forward (destruction) process**. It answers the question: "exactly how much noise should be in the image at step t?" It does this by precomputing two sets of values for all 1000 timesteps:

- `β(beta)`:The **variance** added at each individual step. It starts very small `β_1 = 1e-4` and grows linearly to `β_T = 0.02`, meaning early steps add tiny amounts of noise and later steps add larger amounts.
- `ᾱ(alpha-bar)`:The **cumulative noise** level up to step t. This is the product of all `(1 - β)` values from step 1 to step t. The key use of `ᾱ` is that it lets us jump directly to any noise level in a single calculation, meaning that we don't have to simulate 500 sequential noising steps to get `x_500`, we can compute it directly from `x_0` in one shot using the Diffusion Kernel: `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε`.

```python
class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps=1000):
        super().__init__()
        self.beta  = torch.linspace(1e-4, 0.02, num_time_steps)
        alpha      = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0)   # ᾱ_t
```
**Note**: The scheduler has no learnable parameters — it is a fixed, pre-chosen hyperparameter of the model, not trained.

### Sinusoidal Time Embeddings

The U-Net that predicts noise must handle images at all 1000 different noise levels. A heavily noised image at step 900 needs a very different prediction than a lightly noised image at step 10, yet both are just arrays of pixel values that look superficially similar. The model needs to be told **which step it is currently denoising**.

**Sinusoidal Time Embeddings** solve this by converting the integer timestep t into a rich vector of numbers using sine and cosine waves at different frequencies. Each timestep gets a unique vector. 

```python
class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps, embed_dim):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        )
        embeddings = torch.zeros(time_steps, embed_dim)
        embeddings[:, 0::2] = torch.sin(position * div)  
        embeddings[:, 1::2] = torch.cos(position * div)   
        self.register_buffer('embeddings', embeddings)
```
These Embeddings are then injected directly into every **ResBlock in the U-Net**, so every layer of the model is conditioned on the current noise level at all times.

### U-Net Architecture

The U-Net is the core neural network, the part that actually predicts the noise. It takes a noisy image and the current timestep as input and outputs a noise estimate of the same size as the image. 

##### The architecture has two paths with a bottleneck in the middle:
##### 
- **Encoder (downsampling)**: Progressively shrinks the spatial size of the feature maps using strided convolutions, allowing the model to build up abstract, global understanding of the image structure.
**Encoder path** (each level: 2× ResBlock → optional Attention → stride-2 Conv):

```
(B, 1, 28, 28)   →  in_conv  →  (B, 64, 28, 28)
Level 0: ResBlock×2 → Conv↓2 → (B, 128, 14, 14),  skip=(B, 64, 28, 28)
Level 1: ResBlock×2 → Attn  → Conv↓2 → (B, 256, 7, 7),  skip=(B, 128, 14, 14)
Level 2: ResBlock×2 → Conv↓2 → (B, 256, 3, 3),  skip=(B, 256, 7, 7)
```

- **Bottleneck** (ResBlock → Attention → ResBlock):
```
(B, 256, 3, 3) → ResBlock → Attention → ResBlock → (B, 256, 3, 3)
```
- **Decoder (upsampling)** : Progressively restores the spatial size using transposed convolutions, translating that abstract understanding back into a full-resolution noise prediction.
**Decoder path** (each level: BilinearUp + concat(skip) → ResBlock×2 → Attention):
```
Level 0: ↑2 + skip(256) → cat → (B, 512, 7, 7)   → ResBlock×2
Level 1: ↑2 + skip(128) → cat → (B, 256, 14, 14) → ResBlock×2 → Attn
Level 2: ↑2 + skip(64)  → cat → (B, 128, 28, 28) → ResBlock×2
Output: GroupNorm → SiLU → Conv1×1 → (B, 1, 28, 28)
```
 
- **Skip connections**: Each encoder level is directly connected to its corresponding decoder level via a concatenation. This passes fine spatial detail from the shallow layers to the decoder, compensating for information lost during downsampling. This is what makes the output spatially accurate rather than blurry.

![U-Net Architecture for ddpm](blob:https://markdownviewer.pages.dev/0fa32445-7178-40e4-8418-8cc840fa680c)



**ResBlock (Time-conditioned):**
 Its crucial property is that it receives both the image features `x` and the sinusoidal time embedding, and adds them together channel-wise before processing. This is how the timestep information flows into every layer of the network

```python
class ResBlock(nn.Module):
    def __init__(self, C, num_groups, dropout_prob, t_emb_dim):
        super().__init__()
        self.gnorm1 = nn.GroupNorm(num_groups, C)
        self.gnorm2 = nn.GroupNorm(num_groups, C)
        self.conv1  = nn.Conv2d(C, C, 3, padding=1)
        self.conv2  = nn.Conv2d(C, C, 3, padding=1)
        self.t_proj = nn.Linear(t_emb_dim, C)    # project time → channel dim

    def forward(self, x, t_emb):
        t = self.t_proj(F.silu(t_emb))[:, :, None, None]  # (B, C, 1, 1)
        r = self.conv1(F.silu(self.gnorm1(x)))
        r = r + t           # inject time conditioning
        r = self.dropout(r)
        r = self.conv2(F.silu(self.gnorm2(r)))
        return r + x        # residual connection
```

**Attention (applied at the 14×14 feature map (level 1))**
Self-attention is added at the middle resolution level (14×14). At this scale, the receptive field is large enough that global relationships in the image matter.

```python
class Attention(nn.Module):
    def __init__(self, C, num_heads, dropout_prob):
        super().__init__()
        self.proj1 = nn.Linear(C, C * 3)   # project to Q, K, V in one step
        self.proj2 = nn.Linear(C, C)        # output projection
        self.num_heads = num_heads

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')          # treat pixels as sequence tokens
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q, k, v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        return rearrange(self.proj2(x), 'b h w C -> b C h w')
```

### Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 100 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 2e-4 |
| EMA decay | 0.9999 |
| Timesteps (T) | 1000 |
| Noise schedule | Linear β: 1e-4 → 0.02 |

**Exponential Moving Average (EMA)**: After each optimizer step, a smoothed copy of the model weights is maintained using a weighted average (decay = 0.9999). The EMA model is used exclusively for sampling. Because it averages over many training steps rather than using the latest (potentially noisy) gradient update, EMA weights tend to produce significantly cleaner and more coherent generated images.

### Sampling (Reverse Process)

Sampling is the reverse of training. Instead of adding noise, we remove it 999 times, starting from pure Gaussian noise (as if we were at timestep 1000) and at each step we subtract a calibrated fraction of it, then repeat from t=999 down to t=1.

![Sampling (Reverse Process)](blob:https://markdownviewer.pages.dev/e1902a35-96e3-4fa9-b922-bc4e125f21cf)
---

## Experimental Setup

Both models were trained on the full Fashion-MNIST training split (60,000 images) for 100 epochs. Evaluation used the full 10,000-image test set for real features, and 10,000 generated samples for fake features.

**Evaluation pipeline:**

1. Load a pretrained InceptionV3 (`torchvision.models.inception_v3`, default weights).
2. Register a forward hook on `inception.avgpool` to extract 2048-dimensional features.
3. For real images: normalize from `[-1, 1]` to `[0, 1]`, expand 1→3 channels, bilinear-upsample to 299×299, pass through Inception.
4. For generated images: same channel expansion and upsample.
5. Compute FID using the matrix square root formula (`scipy.linalg.sqrtm`).
6. Compute Inception Score from the softmax class probabilities.

---

## Quantitative Results

### Fréchet Inception Distance (FID)

FID measures the Wasserstein-2 distance between the distribution of real and generated images in Inception feature space:

$$FID = \|\mu_r - \mu_f\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_f - 2(\Sigma_r \Sigma_f)^{1/2}\right)$$

Lower FID → generated distribution is closer to real data.

###  Results Summary

| Metric | VAE | DDPM | Better |
|--------|-----|------|--------|
| **FID ↓** | ~46.57 | ~6.73 | **DDPM** |
| **Training time / epoch** | ~45 s | ~3–4 min | **VAE** |
| **Sampling time (10k images)** | < 5 s | ~15–20 min | **VAE** |
| **Parameters** | ~3.5M | ~12M | **VAE** |

**Interpretation:**

The DDPM achieves substantially better FID, indicating its generated distribution is significantly closer to the real Fashion-MNIST distribution. This aligns with the broader literature showing diffusion models outperform VAEs on image quality metrics. The VAE's higher FID reflects its tendency to produce blurry samples due to the MSE reconstruction loss averaging over multiple modes of the posterior.

---

##  Qualitative Analysis

### **VAE Analysis** 

- **Diversity and class coverage** :The 16 samples span a wide range of categories — shoes (flat shoes, heeled boots, ankle boots), bags, pullovers, dresses, and coats are all present. 
The latent space is well-spread, and random samples from N(0, I) land in recognizable regions rather than producing noise or mode collapse.
- **Blurriness** : The 16 samples generated are generally resemble clothing silhouettes but lack sharp edges and fine textures. Items like trousers and bags tend to be cleaner; shirts and sneakers appear smeared. This is a well-known artifact of VAEs trained with MSE reconstruction loss, which optimizes for pixel-wise average rather than perceptual sharpness.

-  One image in the bottom row appears to have an overlapping artifact due to the fact that the latent space have Overlapping clusters 


| Latent Space | VAE generated samples
|--------|-----
| ![alt text](blob:https://markdownviewer.pages.dev/c9e22fd8-80f8-4349-b27b-a725735c0448)|![alt text](blob:https://markdownviewer.pages.dev/14836a59-c269-46c7-96b8-d19f678b8c79) 

### Reconstruction Fidelity (VAE)

**Fine detail and texture are lost** The logo on the sweatshirt clearly visible in the real image , it is completely absent in the reconstruction , replaced by a smooth grey rectangle. Similarly, the last shirt  becomes a plain in the reconstruction. The MSE loss treats all pixels equally and penalizes large structural errors more than fine textures, so the decoder learns to "average out" detail rather than commit to specific patterns.
![alt text](blob:https://markdownviewer.pages.dev/752d4143-f060-4059-9769-487aa97bc3e4)

### **DDPM Analysis** 

**Sharpness and structural detail** The main difference from the VAE samples is structure. Trouser legs, bag handles, shoe soles, and dress hemlines are often visually distinct. The iterative denoising process allows the model to commit to fine structural details without the averaging problem of the VAE.

**Texture and pattern variety** Within the same category, samples show meaningful variation. The upper-body garments in the top row range from a plain pullover , a zip-up jacket and short sleeves shirt, the bags vary in strap configuration and proportions. This intra-class diversity is a strength of the DDPM's stochastic reverse process 

| DDPM  | DDPM generated samples
|--------|-----
|![alt text](blob:https://markdownviewer.pages.dev/75a6f594-5cae-486f-864f-5f81765ae47b) |![alt text](blob:https://markdownviewer.pages.dev/a5fa4e1e-d1ac-444d-a220-0948c8da9508)

---


