"""Adenda: robustez a queries ruidosas.

Queries = as 10 000 imagens do test com ruído gaussiano σ_teste ∈ {0, 0.1, 0.3, 0.5} (clip a [0,1]);
galeria = train limpo (54k). O ruído ε ~ N(0, I) é gerado uma vez (seed 42) e é o mesmo para todos os
espaços e todos os σ (query = clip(x + σ·ε)), por isso σ é a única coisa que muda.

Por espaço (raw, PCA-64, conv_ae_d64, dae_conv_d64, vae_conv_d64):
  - precision@10 (euclidiana) e accuracy kNN (k=5) das queries ruidosas;
  - MSE da reconstrução da query ruidosa contra a imagem limpa (no raw: o MSE do próprio input ruidoso,
    a referência "sem denoising").

Uso: python -m src.robustness   (salta o cálculo se robustness.csv já existir; refaz a figura)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .data import load_split
from .evaluate import knn_predict, knn_search
from .plots import SERIES, plt, savefig
from .representation import ae_reconstruct, pca_recon_from_images
from .retrieval import SPACES
from .utils import CFG, device, get_logger, out, register

RB = CFG["robustness"]
log = get_logger("robustness")
LABEL = {"raw": "pixels raw", "pca_k64": "PCA k=64", "conv_ae_d64": "ConvAE d=64 (base)",
         "dae_conv_d64": "DAE d=64", "vae_conv_d64": "VAE d=64"}


def reconstruct(space: str, X: np.ndarray) -> np.ndarray:
    if space == "raw":
        return X
    if space.startswith("pca_k"):
        return np.clip(pca_recon_from_images(X, int(space[5:])), 0, 1)
    arch, d = space.rsplit("_d", 1)
    return ae_reconstruct(X, arch, int(d))


def compute() -> pd.DataFrame:
    Xte, yte = load_split("test")
    ytr = load_split("train", mmap=True)[1]
    gen = torch.Generator(device=device())
    gen.manual_seed(CFG["seed"])
    eps = torch.randn(Xte.shape, device=device(), generator=gen).cpu().numpy()
    rows = []
    for space in RB["spaces"]:
        sp = SPACES[space]
        G = np.ascontiguousarray(sp.features("train"), dtype=np.float32)
        for sigma in RB["sigmas"]:
            Xn = np.clip(Xte + sigma * eps, 0, 1).astype(np.float32)
            Q = sp.embed(Xn)
            _, idx = knn_search(Q, G, RB["k"])
            p10 = float((ytr[idx] == yte[:, None]).mean())
            acc = float((knn_predict(G, ytr, Q, k=CFG["classification"]["knn_k"]) == yte).mean())
            R = reconstruct(space, Xn)
            mse = float(((R - Xte) ** 2).mean())
            rows.append({"espaco": space, "sigma_teste": sigma, f"precision@{RB['k']}": p10, "knn_acc": acc,
                         "mse_reconstrucao_vs_limpa": mse, "n_queries": len(Xte)})
            log.info(f"{space} σ={sigma}: P@{RB['k']} {p10:.4f} | kNN {acc:.4f} | MSE vs limpa {mse:.5f}")
        del G
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    base = df[df.sigma_teste == 0].set_index("espaco")
    df["queda_precision"] = df.apply(lambda r: base.loc[r.espaco, f"precision@{RB['k']}"] - r[f"precision@{RB['k']}"], axis=1)
    df["queda_knn_acc"] = df.apply(lambda r: base.loc[r.espaco, "knn_acc"] - r.knn_acc, axis=1)
    return df


def figure(df: pd.DataFrame) -> None:
    pk = f"precision@{RB['k']}"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, sp in enumerate(RB["spaces"]):
        s = df[df.espaco == sp].sort_values("sigma_teste")
        for ax, col in zip(axes, (pk, "knn_acc", "mse_reconstrucao_vs_limpa")):
            v = s[s[col] > 0]   # raw com σ=0 tem MSE exatamente 0, que não existe em escala log
            ax.plot(v.sigma_teste, v[col], "o-", color=SERIES[i], label=LABEL.get(sp, sp), ms=6)
    for ax, t in zip(axes, (f"precision@{RB['k']} (queries ruidosas vs galeria limpa)", "accuracy kNN (k=5)",
                            "MSE da reconstrução vs imagem limpa")):
        ax.set_xticks(RB["sigmas"])
        ax.set_xlabel("σ do ruído nas queries")
        ax.set_title(t, fontsize=10)
    axes[2].set_yscale("log")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Robustez a queries ruidosas (test 10k; galeria = train limpo)", y=1.03)
    fig.tight_layout()
    savefig(fig, "robustness.png", "precision@10, accuracy kNN e MSE de reconstrução vs σ do ruído nas queries")


def main():
    p = out("metrics", "robustness.csv")
    if p.exists():
        df = pd.read_csv(p)
    else:
        df = compute()
        df.to_csv(p, index=False)
    register(p, "robustez a queries ruidosas: precision@10, accuracy kNN e MSE de reconstrução vs imagem limpa, "
             "por espaço e σ do ruído", {"sigmas": RB["sigmas"], "spaces": RB["spaces"]})
    figure(df)
    log.info("robustez concluída")


if __name__ == "__main__":
    main()
