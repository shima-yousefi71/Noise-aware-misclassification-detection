import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = int(torch.prod(torch.tensor(input_shape)))
        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super(Decoder, self).__init__()
        self.output_shape = output_shape
        self.output_dim = int(torch.prod(torch.tensor(output_shape)))
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, self.output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        return out.view(-1, *self.output_shape)  # Reshape to original shape


class GaussianTransformer(nn.Module):
    def __init__(self, latent_dim):
        super(GaussianTransformer, self).__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return self.fc2(h)


class adVAE(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(adVAE, self).__init__()
        self.encoder = Encoder(input_shape, latent_dim)
        self.decoder = Decoder(latent_dim, input_shape)
        self.transformer = GaussianTransformer(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        z_t = self.transformer(z)  # adversarial latent
        x_t = self.decoder(z_t)    # adversarial reconstruction
        return x_hat, mu, logvar, z, z_t, x_t

