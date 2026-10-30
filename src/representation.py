"""Funções de representação pedidas no enunciado: projeção/reconstrução PCA e encode/decode de AEs.

Exemplo:
    from src.representation import pca_project, pca_reconstruct, ae_encode, ae_decode
    Z = pca_project(X, k=64)                 # X: [N,784] ou [N,28,28] em [0,1]
    Xr = pca_reconstruct(Z, k=64)            # [N,784]
    z = ae_encode(X, "conv_ae", 32)          # [N,32]
    Xr = ae_decode(z, "conv_ae", 32)         # [N,784]
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import torch

from .models import build_ae
from .utils import CFG, device, out


def _flat(X) -> np.ndarray:
    """Aceita [N,784], [N,28,28], [N,1,28,28] ou uma única imagem [28,28] -> [N,784] float32."""
    return np.asarray(X, dtype=np.float32).reshape(-1, 784)


# ------------------------------------------------------------------------------------- PCA
@lru_cache(maxsize=None)
def load_pca(k: int, whiten: bool = False):
    """Modelo PCA guardado (outputs/pca/pca_k{k}[_whiten].joblib); para k fora da grelha usa o PCA
    completo truncado."""
    p = out("pca", f"pca_k{k}" + ("_whiten" if whiten else "") + ".joblib")
    if p.exists():
        return joblib.load(p)
    if whiten:
        raise FileNotFoundError(p)
    return _TruncatedPCA(joblib.load(out("pca", "pca_full.joblib")), k)


class _TruncatedPCA:
    def __init__(self, full, k):
        self.mean_, self.components_, self.k = full.mean_, full.components_[:k], k

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_


def pca_project(X, k: int, whiten: bool = False) -> np.ndarray:
    """Projeta imagens no subespaço dos k primeiros componentes principais (ajustado no train)."""
    return load_pca(k, whiten).transform(_flat(X)).astype(np.float32)


def pca_reconstruct(Z, k: int, whiten: bool = False) -> np.ndarray:
    """Reconstrói imagens [N,784] a partir das coordenadas PCA."""
    return np.asarray(load_pca(k, whiten).inverse_transform(np.asarray(Z, dtype=np.float32)), dtype=np.float32)


def pca_recon_from_images(X, k: int) -> np.ndarray:
    return pca_reconstruct(pca_project(X, k), k)


# -------------------------------------------------------------------------------------- AE
@lru_cache(maxsize=None)
def load_ae(arch: str, d: int):
    model = build_ae(arch, d, CFG).to(device())
    ck = torch.load(out("ae", f"{arch}_d{d}", "best.pt"), map_location=device(), weights_only=False)
    model.load_state_dict(ck["model"])
    return model.eval()


@torch.no_grad()
def ae_encode(X, arch: str, d: int, bs: int = 2048) -> np.ndarray:
    """Codifica imagens no latente de dimensão d do autoencoder (arch em dense_ae/conv_ae/transformer_ae)."""
    model = load_ae(arch, d)
    X = torch.as_tensor(_flat(X)).view(-1, 1, 28, 28)
    return torch.cat([model.encode(X[s:s + bs].to(device())).cpu() for s in range(0, len(X), bs)]).numpy()


@torch.no_grad()
def ae_decode(Z, arch: str, d: int, bs: int = 2048) -> np.ndarray:
    """Descodifica latentes [N,d] em imagens [N,784] em [0,1]."""
    model = load_ae(arch, d)
    Z = torch.as_tensor(np.asarray(Z, dtype=np.float32))
    return torch.cat([model.decode(Z[s:s + bs].to(device())).cpu() for s in range(0, len(Z), bs)]).reshape(-1, 784).numpy()


def ae_reconstruct(X, arch: str, d: int) -> np.ndarray:
    return ae_decode(ae_encode(X, arch, d), arch, d)
