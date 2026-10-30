"""Arquiteturas: DenseAE, ConvAE, TransformerAE (sem skip connections: o bottleneck de dimensão d é a
única via de informação entre encoder e decoder) e CNN supervisionada (teto de referência).
Especializados (adenda): ConvVAE (`vae_conv`) e o denoising AE (`dae_conv`, a própria ConvAE; só o
input de treino muda, ver train_ae.py).

Todos os autoencoders recebem e devolvem tensores [B,1,28,28] em [0,1] e expõem encode()/decode().
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DenseAE(nn.Module):
    """MLP 784-512-256-d / d-256-512-784."""

    def __init__(self, d: int, hidden=(512, 256)):
        super().__init__()
        h1, h2 = hidden
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, h1), nn.ReLU(inplace=True),
            nn.Linear(h1, h2), nn.ReLU(inplace=True),
            nn.Linear(h2, d),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d, h2), nn.ReLU(inplace=True),
            nn.Linear(h2, h1), nn.ReLU(inplace=True),
            nn.Linear(h1, 784),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return torch.sigmoid(self.decoder(z)).view(-1, 1, 28, 28)

    def forward(self, x):
        return self.decode(self.encode(x))


def _conv(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _deconv(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 3, 2, 1, output_padding=1), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


class ConvAE(nn.Module):
    """Conv 28->14->7 (32, 64, 64 canais) + Linear para d; decoder espelhado com ConvTranspose."""

    def __init__(self, d: int, ch=(32, 64, 64)):
        super().__init__()
        c1, c2, c3 = ch
        self.c3 = c3
        self.encoder = nn.Sequential(
            _conv(1, c1, 2),       # 14x14
            _conv(c1, c2, 2),      # 7x7
            _conv(c2, c3, 1),      # 7x7
            nn.Flatten(),
            nn.Linear(c3 * 49, d),
        )
        self.fc_dec = nn.Sequential(nn.Linear(d, c3 * 49), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            _conv(c3, c2, 1),      # 7x7
            _deconv(c2, c1),       # 14x14
            _deconv(c1, 16),       # 28x28
            nn.Conv2d(16, 1, 3, 1, 1),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self.c3, 7, 7)
        return torch.sigmoid(self.decoder(h))

    def forward(self, x):
        return self.decode(self.encode(x))


class ConvVAE(ConvAE):
    """VAE com o tronco e o decoder do ConvAE. A Linear(3136->d) do encoder passa a ser a cabeça de mu e
    junta-se uma segunda Linear(3136->d) para logvar. encode() devolve mu (determinístico), por isso o resto
    do projeto usa mu como latente sem código especial."""

    def __init__(self, d: int, ch=(32, 64, 64)):
        super().__init__(d, ch)
        trunk = self.encoder
        self.encoder = trunk[:-1]          # convs + Flatten, iguais ao ConvAE
        self.fc_mu = trunk[-1]             # a mesma camada (e a mesma inicialização) do ConvAE
        self.fc_logvar = nn.Linear(ch[2] * 49, d)

    def encode_stats(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def encode(self, x):
        return self.fc_mu(self.encoder(x))

    def forward_train(self, x):
        """Reconstrução a partir de z amostrado (reparametrização), mais mu e logvar para o KL."""
        mu, logvar = self.encode_stats(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar


class TransformerAE(nn.Module):
    """Patches 4x4 -> 49 tokens + embeddings posicionais -> encoder de self-attention (pre-LN) ->
    attention pooling (query aprendida) -> vetor z de dimensão d.
    Decoder simétrico: z é projetado e replicado pelos 49 tokens, somado a embeddings posicionais
    próprios do decoder, passa por blocos de self-attention e cada token é projetado no seu patch 4x4.
    O decoder só vê z (mais constantes aprendidas), portanto não há skip connections."""

    def __init__(self, d: int, patch: int = 4, embed_dim: int = 128, depth: int = 4, heads: int = 4,
                 mlp_ratio: int = 2):
        super().__init__()
        self.p = patch
        self.n = (28 // patch) ** 2
        pdim = patch * patch
        E = embed_dim
        self.patch_embed = nn.Linear(pdim, E)
        self.pos_enc = nn.Parameter(torch.zeros(1, self.n, E))
        self.pos_dec = nn.Parameter(torch.zeros(1, self.n, E))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)
        nn.init.trunc_normal_(self.pos_dec, std=0.02)

        def blocks():
            layer = nn.TransformerEncoderLayer(E, heads, dim_feedforward=mlp_ratio * E, dropout=0.0,
                                               activation="gelu", batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)

        self.enc_blocks = blocks()
        self.enc_norm = nn.LayerNorm(E)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, E))
        nn.init.trunc_normal_(self.pool_query, std=0.02)
        self.pool = nn.MultiheadAttention(E, heads, batch_first=True)
        self.to_latent = nn.Linear(E, d)

        self.from_latent = nn.Linear(d, E)
        self.dec_blocks = blocks()
        self.dec_norm = nn.LayerNorm(E)
        self.to_patch = nn.Linear(E, pdim)

    def _patchify(self, x):  # [B,1,28,28] -> [B,49,16]
        B, p = x.shape[0], self.p
        return x.unfold(2, p, p).unfold(3, p, p).reshape(B, -1, p * p)

    def _unpatchify(self, t):  # [B,49,16] -> [B,1,28,28]
        B, p, g = t.shape[0], self.p, 28 // self.p
        return t.view(B, g, g, p, p).permute(0, 1, 3, 2, 4).reshape(B, 1, 28, 28)

    def encode(self, x):
        h = self.patch_embed(self._patchify(x)) + self.pos_enc
        h = self.enc_norm(self.enc_blocks(h))
        q = self.pool_query.expand(h.shape[0], -1, -1)
        pooled, _ = self.pool(q, h, h, need_weights=False)
        return self.to_latent(pooled.squeeze(1))

    def decode(self, z):
        h = self.from_latent(z).unsqueeze(1) + self.pos_dec   # [B,49,E]
        h = self.dec_norm(self.dec_blocks(h))
        return torch.sigmoid(self._unpatchify(self.to_patch(h)))

    def forward(self, x):
        return self.decode(self.encode(x))


class SupervisedCNN(nn.Module):
    """CNN supervisionada ponta a ponta (teto de referência para a classificação)."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            _conv(1, 32, 1), _conv(32, 32, 1), nn.MaxPool2d(2), nn.Dropout(0.1),   # 14x14
            _conv(32, 64, 1), _conv(64, 64, 1), nn.MaxPool2d(2), nn.Dropout(0.2),  # 7x7
            _conv(64, 128, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_classes))

    def embed(self, x):
        return self.features(x)

    def forward(self, x):
        return self.head(self.features(x))


ARCHS = {"dense_ae": DenseAE, "conv_ae": ConvAE, "transformer_ae": TransformerAE,
         "vae_conv": ConvVAE, "dae_conv": ConvAE}


def build_ae(arch: str, d: int, cfg: dict | None = None) -> nn.Module:
    if arch == "transformer_ae":
        kw = (cfg or {}).get("ae", {}).get("transformer", {})
        return TransformerAE(d, **kw)
    return ARCHS[arch](d)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
