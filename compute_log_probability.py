#this page was made to work only for amortized
#and then given to claude with a prompt to make work with a string given at run time
#for image making from the given model
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model import ConvVAE

def log_gaussian_pdf(x, mean, logvar):
    variance = torch.exp(logvar)
    log_2pi = torch.log(torch.tensor(2 * torch.pi))
    return -0.5 * ((x - mean) ** 2 / variance + logvar + log_2pi)


def compute_log_p_x(model, x, num_samples=1000, sigma_p=0.4):
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(x)  # mu and logvar are (batch_size, latent_dim)
        sigma = torch.exp(0.5 * logvar)

        log_p_x_samples = []
        for _ in range(num_samples):
            epsilon = torch.randn_like(sigma)
            z = mu + sigma * epsilon

            recon_x = model.decode(z)  # recon_x is (batch_size, 1, 28, 28)

            # Flatten images and reconstructions
            x_flat = x.flatten(start_dim=1)  # x_flat is (batch_size, 784)
            recon_x_flat = recon_x.flatten(start_dim=1)  # recon_x_flat is (batch_size, 784)

            # Log probabilities
            log_p_z = log_gaussian_pdf(z, torch.zeros_like(z), torch.zeros_like(z)).sum(dim=1)  # log N(0, I)

            # Ensure variance has the correct shape (batch_size, 784)
            log_p_x_given_z = log_gaussian_pdf(x_flat, recon_x_flat,
                                               torch.log(torch.ones_like(recon_x_flat) * sigma_p ** 2)).sum(
                dim=1)  # log N(G(z), sigma_p^2 I)
            log_q_z_given_x = log_gaussian_pdf(z, mu, logvar).sum(dim=1)  # log N(mu, sigma^2)

            log_p_x_samples.append(log_p_z + log_p_x_given_z - log_q_z_given_x)

        log_p_x_tensor = torch.stack(log_p_x_samples)
        log_p_x = torch.logsumexp(log_p_x_tensor, dim=0) - torch.log(
            torch.tensor(num_samples, dtype=float, device=x.device))
        return log_p_x


def get_digit_images(dataloader, num_images_per_digit):
    digit_images = {i: [] for i in range(10)}
    with torch.no_grad():
        for images, labels in dataloader:
            for i in range(images.size(0)):
                label = labels[i].item()
                if len(digit_images[label]) < num_images_per_digit:
                    digit_images[label].append(images[i])
            if all(len(digit_images[d]) == num_images_per_digit for d in range(10)):
                return {k: torch.stack(v) for k, v in digit_images.items()}

# Get experiment type from user at runtime
exp_type = input("Enter the experiment type (amortized/latent_optimization)): ").strip()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_targets = train_dataset.targets
train_idx, _ = train_test_split(range(len(train_targets)), train_size=20000, stratify=train_targets)
train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

train_images_by_digit = get_digit_images(train_loader, 5)
test_images_by_digit = get_digit_images(test_loader, 5)

latent_dim = 200
model = ConvVAE(latent_dim=latent_dim)
model.load_state_dict(torch.load(f'checkpoints/{exp_type}/vae_epoch_30.pth'))
model.eval()

plt.figure(figsize=(10, 2))
for digit in range(10):
    img = train_images_by_digit[digit][0].unsqueeze(0)
    log_p_x = compute_log_p_x(model, img)
    plt.subplot(1, 10, digit + 1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"{log_p_x.item():.1f}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(f'results/{exp_type}/digit_log_probabilities.png')

avg_log_probs_per_digit = {}
for digit in range(10):
    all_images = torch.cat([train_images_by_digit[digit], test_images_by_digit[digit]], dim=0)
    log_probs = compute_log_p_x(model, all_images).cpu().numpy()
    avg_log_probs_per_digit[digit] = np.mean(log_probs)

all_train_images = torch.cat(list(train_images_by_digit.values()))
all_test_images = torch.cat(list(test_images_by_digit.values()))

avg_log_p_train = compute_log_p_x(model, all_train_images).mean().item()
avg_log_p_test = compute_log_p_x(model, all_test_images).mean().item()

with open(f'results/{exp_type}/log_probability_results.txt', 'w') as f:
    f.write("Average log-probabilities per digit\n")
    for digit in avg_log_probs_per_digit:
        f.write(f"{digit} : {avg_log_probs_per_digit[digit]}\n")

    f.write(f"\nAverage log-probability (train): {avg_log_p_train}\n")
    f.write(f"Average log-probability (test): {avg_log_p_test}\n")