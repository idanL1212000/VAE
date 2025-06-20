import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from model import ConvVAE


def loss_function(recon_x, x, mu, logvar, sigma_p=0.4):
    # MSE reconstruction loss
    MSE = nn.MSELoss(reduction='sum')(recon_x, x) / (x.size(0) * x.size(1) * x.size(2) * x.size(3))

    # KL Divergence: D_KL(q(z|x) || p(z))
    # For each latent dimension: 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
    KLD = 0.5 * torch.mean(mu.pow(2) + logvar.exp() - logvar - 1)

    return MSE + KLD, MSE, KLD


def plot_loss_curves(train_losses, train_mse_losses, train_kld_losses,
                     val_losses, val_mse_losses, val_kld_losses, epochs):
    """Plot training and validation loss curves in a single figure"""
    plt.figure(figsize=(10, 6))

    epochs_range = range(1, epochs + 1)

    # Plot all metrics in a single graph
    plt.plot(epochs_range, train_losses, 'b-', label='Train Total Loss')
    plt.plot(epochs_range, val_losses, 'b--', label='Val Total Loss')
    plt.plot(epochs_range, train_mse_losses, 'r-', label='Train MSE')
    plt.plot(epochs_range, val_mse_losses, 'r--', label='Val MSE')
    plt.plot(epochs_range, train_kld_losses, 'g-', label='Train KLD')
    plt.plot(epochs_range, val_kld_losses, 'g--', label='Val KLD')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Latent Optimization VAE Training and Validation Losses')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/latent_optimization/loss_curves.png')
    plt.close()


# Random seed for reproducibility
torch.manual_seed(5)

# Create directories for saving results
os.makedirs("results/latent_optimization", exist_ok=True)
os.makedirs("checkpoints/latent_optimization", exist_ok=True)

print("Starting Latent Optimization VAE training...")

# MNIST Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# Take a stratified subset of the training data, keeping only 20000 samples
train_targets = train_dataset.targets
train_idx, val_idx = train_test_split(range(len(train_targets)), train_size=18000, test_size=2000,
                                      stratify=train_targets)
train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
val_dataset = torch.utils.data.Subset(
    torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform), val_idx)

train_loader = DataLoader(train_dataset, batch_size=64,
                          shuffle=False)  # Important: Don't shuffle for latent optimization
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model parameters
latent_dim = 200
epochs = 30
learning_rate_decoder = 0.001  # Learning rate for the decoder
learning_rate_latent = 0.01  # Learning rate for the latent vectors (10x higher as recommended)
sigma_p = 0.4

model = ConvVAE(latent_dim=latent_dim)  # we only need the decoder part of the VAE
# Initialize latent parameters for each training example
# Important: We use mu and logvar (log of variance) for numerical stability
num_train = len(train_dataset)
latent_params = {
    'mu': torch.zeros(num_train, latent_dim, requires_grad=True),
    'logvar': torch.zeros(num_train, latent_dim, requires_grad=True)
}

# Initialize latent parameters for validation set
num_val = len(val_dataset)
val_latent_params = {
    'mu': torch.zeros(num_val, latent_dim, requires_grad=True),
    'logvar': torch.zeros(num_val, latent_dim, requires_grad=True)
}

optimizer_decoder = optim.Adam(model.parameters(), lr=learning_rate_decoder)
optimizer_latent = optim.Adam([latent_params['mu'], latent_params['logvar']], lr=learning_rate_latent)
optimizer_val_latent = optim.Adam([val_latent_params['mu'], val_latent_params['logvar']], lr=learning_rate_latent)

prior_dist = torch.distributions.Normal(0, 1)

train_losses = []
train_mse_losses = []
train_kld_losses = []
val_losses = []
val_mse_losses = []
val_kld_losses = []

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    train_mse = 0
    train_kld = 0

    for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} - Training")):
        batch_size = data.size(0)
        batch_start_idx = batch_idx * batch_size
        batch_end_idx = min((batch_idx + 1) * batch_size, num_train)

        batch_mu = latent_params['mu'][batch_start_idx:batch_end_idx]
        batch_logvar = latent_params['logvar'][batch_start_idx:batch_end_idx]

        optimizer_decoder.zero_grad()
        optimizer_latent.zero_grad()

        z = model.reparameterize(batch_mu, batch_logvar)
        recon_batch = model.decode(z)
        loss, mse, kld = loss_function(recon_batch, data, batch_mu, batch_logvar, sigma_p)

        loss.backward()

        optimizer_decoder.step()
        optimizer_latent.step()

        train_loss += loss.item()
        train_mse += mse.item()
        train_kld += kld.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_train_mse = train_mse / len(train_loader)
    avg_train_kld = train_kld / len(train_loader)

    train_losses.append(avg_train_loss)
    train_mse_losses.append(avg_train_mse)
    train_kld_losses.append(avg_train_kld)

    # Validation phase
    model.eval()
    val_loss = 0
    val_mse = 0
    val_kld = 0

    for batch_idx, (data, _) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} - Validation")):
        batch_size = data.size(0)
        batch_start_idx = batch_idx * batch_size
        batch_end_idx = min((batch_idx + 1) * batch_size, num_val)

        batch_mu = val_latent_params['mu'][batch_start_idx:batch_end_idx]
        batch_logvar = val_latent_params['logvar'][batch_start_idx:batch_end_idx]

        optimizer_val_latent.zero_grad()

        z = model.reparameterize(batch_mu, batch_logvar)
        recon_batch = model.decode(z)
        loss, mse, kld = loss_function(recon_batch, data, batch_mu, batch_logvar, sigma_p)

        loss.backward()
        optimizer_val_latent.step()

        val_loss += loss.item()
        val_mse += mse.item()
        val_kld += kld.item()

        val_loss += loss.item()
        val_mse += mse.item()
        val_kld += kld.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_mse = val_mse / len(val_loader)
    avg_val_kld = val_kld / len(val_loader)

    val_losses.append(avg_val_loss)
    val_mse_losses.append(avg_val_mse)
    val_kld_losses.append(avg_val_kld)

    print(f'Epoch {epoch}/{epochs}, '
          f'Train Loss: {avg_train_loss:.4f}, '
          f'Train MSE: {avg_train_mse:.4f}, '
          f'Train KLD: {avg_train_kld:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, '
          f'Val MSE: {avg_val_mse:.4f}, '
          f'Val KLD: {avg_val_kld:.4f}')

if epoch in [1, 5, 10, 20, 30]:
    torch.save(model.state_dict(), f'checkpoints/latent_optimization/vae_epoch_{epoch}.pth')

print("Training completed.")

plot_loss_curves(train_losses, train_mse_losses, train_kld_losses,
                 val_losses, val_mse_losses, val_kld_losses, epochs)