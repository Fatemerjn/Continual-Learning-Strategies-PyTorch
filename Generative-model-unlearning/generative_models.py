import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        
        self.decoder_fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(self.decoder_fc(z)), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

class GenerativeReplay(nn.Module):
    def __init__(self, device, num_tasks, latent_dim=128, generator_epochs=10):
        super(GenerativeReplay, self).__init__()
        self.device = device
        self.num_tasks = num_tasks
        self.generator_epochs = generator_epochs
        # Store a separate generator for each task
        self.generators = nn.ModuleList([
            VAE(latent_dim).to(device) for _ in range(num_tasks)
        ])
        
    def train_generator(self, task_id, train_loader):
        print(f"Training generator for Task {task_id + 1}...")
        generator = self.generators[task_id]
        generator.train()
        optimizer = optim.Adam(generator.parameters(), lr=1e-3)
        
        for epoch in range(self.generator_epochs):
            pbar = tqdm(train_loader, desc=f"Generator Epoch {epoch+1}/{self.generator_epochs}")
            for images, _ in pbar:
                images = images.to(self.device)
                optimizer.zero_grad()
                recon_images, mu, logvar = generator(images)
                loss = vae_loss_function(recon_images, images, mu, logvar)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item() / len(images))

    def get_rehearsal_batch(self, task_id, batch_size):
        if task_id == 0:
            return None
        
        samples_per_task = batch_size // task_id
        re_images, re_labels, re_task_ids = [], [], []

        for t_id in range(task_id):
            generator = self.generators[t_id]
            generator.eval()
            with torch.no_grad():
                # Generate random samples
                z = torch.randn(samples_per_task, self.generators[0].fc_mu.out_features).to(self.device)
                generated_images = generator.decoder(generator.decoder_fc(z))
                re_images.append(generated_images)
                # For simplicity, assign random labels from the task's class range
                re_labels.append(torch.randint(0, 10, (samples_per_task,)))
                re_task_ids.append(torch.full((samples_per_task,), t_id, dtype=torch.long))

        return torch.cat(re_images), torch.cat(re_labels), torch.cat(re_task_ids)

    def on_task_end(self, task_id, train_loader):
        self.train_generator(task_id, train_loader)