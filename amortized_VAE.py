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

    MSE = nn.MSELoss(reduction='sum')(recon_x, x) / (x.size(0) * x.size(1) * x.size(2) * x.size(3))

    # For each latent dimension: 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
    KLD = 0.5 * torch.mean(mu.pow(2) + logvar.exp() - logvar - 1)

    return MSE + KLD, MSE, KLD

#this function was created with halp of llm
def plot_loss_curves(train_losses, train_mse_losses, train_kld_losses,
                     val_losses, val_mse_losses, val_kld_losses, epochs):
    """Plot training and validation loss curves in a single figure"""
    plt.figure(figsize=(10, 6))

    epochs_range = range(1, epochs + 1)

    plt.plot(epochs_range, train_losses, 'b-', label='Train Total Loss')
    plt.plot(epochs_range, val_losses, 'b--', label='Val Total Loss')
    plt.plot(epochs_range, train_mse_losses, 'r-', label='Train MSE')
    plt.plot(epochs_range, val_mse_losses, 'r--', label='Val MSE')
    plt.plot(epochs_range, train_kld_losses, 'g-', label='Train KLD')
    plt.plot(epochs_range, val_kld_losses, 'g--', label='Val KLD')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('VAE Training and Validation Losses')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/amortized/loss_curves.png')
    plt.close()

# random seed for reproducibility
torch.manual_seed(5)

# Create directories for saving results
os.makedirs("results/amortized", exist_ok=True)
os.makedirs("checkpoints/amortized", exist_ok=True)

print("Starting VAE training and visualization...")

# MNIST Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# take a stratified subset of the training data, keeping only 20000 samples
train_targets = train_dataset.targets
train_idx, val_idx = train_test_split(range(len(train_targets)), train_size=18000, test_size=2000,
                                      stratify=train_targets)
train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
val_dataset = torch.utils.data.Subset(
    torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform), val_idx)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model
latent_dim = 200
epochs = 30
learning_rate = 0.001
sigma_p = 0.4
model = ConvVAE(latent_dim=latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss, mse, kld = loss_function(recon_batch, data, mu, logvar, sigma_p)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_mse += mse.item()
        train_kld += kld.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_train_mse = train_mse / len(train_loader)
    avg_train_kld = train_kld / len(train_loader)

    train_losses.append(avg_train_loss)
    train_mse_losses.append(avg_train_mse)
    train_kld_losses.append(avg_train_kld)

    # Validation
    model.eval()
    val_loss = 0
    val_mse = 0
    val_kld = 0

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} - Validation")):
            recon_batch, mu, logvar = model(data)
            loss, mse, kld = loss_function(recon_batch, data, mu, logvar, sigma_p)

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
        torch.save(model.state_dict(), f'checkpoints/amortized/vae_epoch_{epoch}.pth')

print("Training completed.")

plot_loss_curves(train_losses, train_mse_losses, train_kld_losses,
                 val_losses, val_mse_losses, val_kld_losses, epochs)