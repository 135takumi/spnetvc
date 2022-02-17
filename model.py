import torch
import torch.nn as nn


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


class Encoder2d(nn.Module):

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


class Encoder1d(nn.Module):

    def __init__(self, in_channels=32):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=1, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, kernel_size=1, stride=1),
            nn.AvgPool1d(32, stride=1)
        )
        self.fc_mean = nn.Linear(64, 64)
        self.fc_logvar = nn.Linear(64, 64)

    def forward(self, x):
        x = (torch.squeeze(x)).permute(0,2,1)
        x = self.model(x)
        x = x.view(-1, 64)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar


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
            # to do: わんちゃんバグる
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

    def __init__(self):
        super().__init__()

        self.cts_encoder = Encoder2d()
        self.atr_encoder = Encoder2d()
        self.decoder = Decoder()


class SplitterVC(nn.Module):
    def __init__(self,  n_speaker):
        super().__init__()

        self.vae = VAE()
        self.ld = LatentDiscriminator(n_speaker)

    def cts_encode(self, x):
        mean, logvar = self.vae.cts_encoder(x)
        z = reparameterize(mean, logvar)

        return z, mean, logvar

    def atr_encode(self, x):
        mean, logvar = self.vae.atr_encoder(x)
        z = reparameterize(mean, logvar)

        return z, mean, logvar

    def decode(self, cts_z, atr_z):
        x = self.vae.decoder(cts_z, atr_z)

        return x

    def cts_discriminate(self, cts_z):
        x = self.ld(cts_z)

        return x

    def atr_discriminate(self, atr_z):
        x = self.ld(atr_z)

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