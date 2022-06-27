import torch
import torch.nn as nn
import torch.nn.functional as F


import hyper_parameters as hp


def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mean + eps * std

    return z


def enc2d_layers():
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
    )

    return model


class Contents_Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = enc2d_layers()

    def forward(self, x):
        z = self.model(x)
        z = z.view(-1, 64)

        return z


class Attribute_Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = enc2d_layers()
        self.fc_mean = nn.Linear(64, 64)
        self.fc_logvar = nn.Linear(64, 64)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 64)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar


class VectorQuantizer(nn.Module):
    def __init__(self, emb_num, emb_dim):
        super(VectorQuantizer, self).__init__()

        self._emb_num = emb_num
        self._emb_dim = emb_dim

        self._emb = nn.Embedding(self._emb_num, self._emb_dim)
        self._emb.weight.data.uniform_(-1/self._emb_num, 1/self._emb_num)

    def forward(self, z):
        # flatten input
        z_shape = z.shape
        flat_z = z.view(-1, self._emb_dim)

        distance = (torch.sum(flat_z ** 2, dim=1, keepdim=True)
                    + torch.sum(self._emb.weight**2, dim=1)
                    - 2 * torch.matmul(flat_z, self._emb.weight.t()))

        # Encoding
        encodings_indices = torch.argmin(distance, dim=1).unsqueeze(1)
        enoodings = torch.zeros(encodings_indices.size(0), self._emb_num, device=hp.device)
        enoodings.scatter_(1, encodings_indices, 1)

        # Quantize and unflatten
        quantized_z = torch.matmul(enoodings, self._emb.weight).view(z_shape)

        # Loss
        commitment_loss = 0.5 * F.mse_loss(quantized_z.detach(), z)
        emb_loss = 0.5 * F.mse_loss(quantized_z, z.detach())

        quantized_z = z + (quantized_z - z).detach()

        return quantized_z, commitment_loss, emb_loss


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU()),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU()),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU()),
            nn.Sequential(
                nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU()),
            nn.ConvTranspose2d(96, 1, kernel_size=4, stride=2, padding=1,bias=False)]
        )

    def forward(self, cts_z, atr_z):
        cts_z = cts_z.view(-1, 64, 1, 1)
        atr_z = atr_z.view(-1, 64, 1, 1)
        batch_size = cts_z.size(0)
        z_size = cts_z.size(1)
        dec_output = cts_z

        for block in self.blocks:
            img_size = dec_output.size(2)
            x = [dec_output, atr_z.expand(batch_size, z_size, img_size, img_size)]
            x = torch.cat(x, 1)
            dec_output = block(x)

        return dec_output


class LatentDiscriminator(nn.Module):
    def __init__(self, n_speaker):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, n_speaker),
        )

    def forward(self, x):
        x = self.model(x)

        return x


class VAE(nn.Module):

    def __init__(self, emb_num, emb_dim):
        super().__init__()

        self.cts_encoder = Contents_Encoder()
        self.atr_encoder = Attribute_Encoder()
        self.vector_quantizer = VectorQuantizer(emb_num, emb_dim)
        self.decoder = Decoder()


class SplitterVC(nn.Module):
    def __init__(self,  n_speaker, emb_num, emb_dim):
        super().__init__()

        self.vae = VAE(emb_num, emb_dim)
        self.cts_ld = LatentDiscriminator(n_speaker)
        self.atr_ld = LatentDiscriminator(n_speaker)

    def cts_encode(self, x):
        z = self.vae.cts_encoder(x)
        quantized_z, commitment_loss, emb_loss = self.vae.vector_quantizer(z)

        return quantized_z, commitment_loss, emb_loss

    def atr_encode(self, x):
        mean, logvar = self.vae.atr_encoder(x)
        z = reparameterize(mean, logvar)

        return z, mean, logvar

    def decode(self, cts_z, atr_z):
        x = self.vae.decoder(cts_z, atr_z)

        return x

    def cts_discriminate(self, cts_z):
        x = self.cts_ld(cts_z)

        return x

    def atr_discriminate(self, atr_z):
        x = self.atr_ld(atr_z)

        return x


class Classifier(nn.Module):

    def __init__(self, n_speaker):
        super().__init__()

        self.model1 = enc2d_layers()
        self.model2 = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, n_speaker)
        )

    def forward(self, x):
        x = self.model1(x)
        x = x.view(-1, 64)
        x = self.model2(x)

        return x