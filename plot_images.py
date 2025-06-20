#this page was made to work only for amortized
#and then given to claude with a prompt to make work with a string given at run time
#for image making from the given model
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import ConvVAE


def plot_reconstructions_across_epochs(originals, reconstructions_by_epoch, prefix, exp_type):
    """Plot original images and their reconstructions across epochs in a single image"""
    plt.figure(figsize=(20, 12))
    n = 10  # Number of images
    rows = 6  # Original + 5 epochs

    epochs = [1, 5, 10, 20, 30]

    # Plot original images in the first row
    for i in range(n):
        plt.subplot(rows, n, i + 1)
        plt.imshow(originals[i][0].cpu().detach().numpy(), cmap='gray')
        if i == 0:
            plt.ylabel("Original", fontsize=12, rotation=0, labelpad=40)
        plt.axis('off')

    # Plot reconstructions for each epoch in subsequent rows
    for row_idx, epoch in enumerate(epochs):
        for i in range(n):
            plt.subplot(rows, n, (row_idx + 1) * n + i + 1)
            plt.imshow(reconstructions_by_epoch[epoch][i][0].cpu().detach().numpy(), cmap='gray')
            if i == 0:
                plt.ylabel(f"Epoch {epoch}", fontsize=12, rotation=0, labelpad=40)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'results/{exp_type}/{prefix}_reconstructions_all_epochs.png')
    plt.close()


def plot_samples_across_epochs(samples_by_epoch, exp_type):
    """Plot samples generated from random latent vectors across epochs"""
    plt.figure(figsize=(20, 6))
    n = 10  # Number of samples
    rows = 5  # 5 epochs

    epochs = [1, 5, 10, 20, 30]

    for row_idx, epoch in enumerate(epochs):
        for i in range(n):
            plt.subplot(rows, n, row_idx * n + i + 1)
            plt.imshow(samples_by_epoch[epoch][i][0].cpu().detach().numpy(), cmap='gray')
            if i == 0:
                plt.ylabel(f"Epoch {epoch}", fontsize=12, rotation=0, labelpad=40)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'results/{exp_type}/samples_all_epochs.png')
    plt.close()


# Get experiment type from user at runtime
exp_type = input("Enter the experiment type (amortized/latent_optimization): ").strip()


print(f"Using experiment type: {exp_type}")

# random seed for reproducibility
torch.manual_seed(5)

print(f"Starting VAE visualization for experiment type: {exp_type}...")

# MNIST Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# take a stratified subset of the training data, keeping only 20000 samples
train_targets = train_dataset.targets
train_idx, _ = train_test_split(range(len(train_targets)), train_size=20000, stratify=train_targets)
train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

latent_dim = 200
model = ConvVAE(latent_dim=latent_dim)

# Fixed noise for visualization
fixed_noise = torch.randn(10, latent_dim)

# Select 10 random test images (1 from each class) for visualization
val_images = []
for digit in range(10):
    indices = (test_dataset.targets == digit).nonzero(as_tuple=True)[0]
    random_idx = torch.randint(0, indices.size(0), (1,)).item()
    img, _ = test_dataset[indices[random_idx]]
    val_images.append(img)
val_images = torch.stack(val_images)

# Select 10 random training images (1 from each class) for visualization
train_images = []
for digit in range(10):
    indices = (train_dataset.dataset.targets == digit).nonzero(as_tuple=True)[0]
    random_idx = torch.randint(0, len(indices), (1,)).item()
    img, _ = train_dataset.dataset[indices[random_idx]]
    train_images.append(img)
train_images = torch.stack(train_images)

# Store reconstructions and samples for each epoch
val_reconstructions_by_epoch = {}
train_reconstructions_by_epoch = {}
samples_by_epoch = {}

# Generate reconstructions and samples for all epochs
for epoch in [1, 5, 10, 20, 30]:
    # loading from file
    model_path = f'checkpoints/{exp_type}/vae_epoch_{epoch}.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        # Generate reconstructions
        val_recon, _, _ = model(val_images)
        val_reconstructions_by_epoch[epoch] = val_recon

        train_recon, _, _ = model(train_images)
        train_reconstructions_by_epoch[epoch] = train_recon

        # Generate samples
        samples = model.decode(fixed_noise)
        samples_by_epoch[epoch] = samples

# Create the consolidated visualizations
plot_reconstructions_across_epochs(val_images, val_reconstructions_by_epoch, "val", exp_type)
plot_reconstructions_across_epochs(train_images, train_reconstructions_by_epoch, "train", exp_type)
plot_samples_across_epochs(samples_by_epoch, exp_type)

print(f"All visualizations complete for experiment type: {exp_type}!")