"""Fase 4 (análise): reconstruções original vs PCA vs AEs, grelha UMAP dos embeddings de todos os
modelos (linhas = arquitetura, colunas = d; PCA e LDA como referência) e métricas de agrupamento
por classe com ranking final.

- d=2 usa o latente diretamente (sem UMAP); restantes: UMAP 2D (n_neighbors=15, min_dist=0.1).
- Silhouette, Davies-Bouldin e NMI/ARI do k-means (k=10) calculados no espaço latente original, nos
  10k pontos do subconjunto de visualização; accuracy kNN (k=5) com o train completo como referência
  e o val como queries.

Adenda: os especializados (VAE/DAE) entram como linhas extra na grelha, nas métricas de agrupamento e no
ranking (coluna `specialized=True`), e como curvas tracejadas na figura MSE/SSIM vs d.

Uso: python -m src.embeddings_viz   (modelos ainda não treinados são ignorados; voltar a correr no fim)
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import torch

from .data import load_split, viz_indices
from .evaluate import clustering_metrics, cls_metrics, knn_predict, ssim_pairs
from .plots import SERIES, SHORT, class_legend, plt, savefig, scatter_classes, show_images
from .representation import ae_reconstruct, pca_recon_from_images
from .utils import CFG, device, exists, get_logger, out, register, save_npy

A = CFG["ae"]
DIMS = A["dims"]
ROWS = ["pca"] + list(A["archs"])
SPECIAL = list(A["specialized"])
LABEL = {"pca": "PCA", "dense_ae": "DenseAE", "conv_ae": "ConvAE", "transformer_ae": "TransformerAE", "lda": "LDA",
         "vae_conv": "VAE (conv)", "dae_conv": "DAE (conv)"}
COLOR = {"pca": SERIES[0], "dense_ae": SERIES[1], "conv_ae": SERIES[2], "transformer_ae": SERIES[3], "lda": SERIES[4],
         "vae_conv": SERIES[6], "dae_conv": SERIES[7]}
log = get_logger("embeddings_viz")


def latent(space: str, d: int | None, split: str):
    """Matriz de representação de um espaço (PCA-k, LDA ou AE) para um split, ou None se não existir."""
    if space == "pca":
        p = out("pca", "proj", f"pca_k{d}_{split}.npy")
    elif space == "lda":
        p = out("lda", f"lda_{split}.npy")
    else:
        p = out("latents", f"{space}_d{d}_{split}.npy")
    if not p.exists():
        return None
    Z = np.load(p)
    return Z[:, :d] if space == "lda" and d else Z


def ae_ready(arch, d):
    return out("ae", f"{arch}_d{d}", "metrics.json").exists()


def all_ready():
    return all(ae_ready(a, d) for a in A["archs"] for d in DIMS)


# --------------------------------------------------------------------------- reconstruções
def reconstructions():
    if not all_ready():
        log.info("reconstruções: à espera de todos os AEs")
        return
    p = out("metrics", "recon_summary.csv")
    Xv, yv = load_split("val")
    Xt, _ = load_split("test")
    # linhas do PCA calculadas uma vez; as dos AEs são sempre refeitas a partir de ae_summary.csv
    old = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if len(old) and (old.modelo == "pca").any():
        rows = old[old.modelo == "pca"][["modelo", "d", "split", "mse", "ssim"]].to_dict("records")
    else:
        rows = []
        for split, X in (("val", Xv), ("test", Xt)):
            xt = torch.as_tensor(X, device=device()).view(-1, 1, 28, 28)
            for d in DIMS:
                R = pca_recon_from_images(X, d)
                rt = torch.as_tensor(np.clip(R, 0, 1), device=device()).view(-1, 1, 28, 28)
                rows.append({"modelo": "pca", "d": d, "split": split, "mse": float(((R - X) ** 2).mean()),
                             "ssim": float(ssim_pairs(xt, rt).mean())})
    ae = pd.read_csv(out("metrics", "ae_summary.csv"))
    for split in ("val", "test"):
        for _, r in ae.iterrows():
            rows.append({"modelo": r.arch, "d": r.d, "split": split, "mse": r[f"{split}_mse"], "ssim": r[f"{split}_ssim"]})
    df = pd.DataFrame(rows)
    df["specialized"] = df.modelo.isin(SPECIAL)
    df.to_csv(p, index=False)
    register(p, "MSE e SSIM de reconstrução (val/test) de PCA, dos 18 AEs base e dos especializados (VAE/DAE) por d")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    for ax, met, lab in ((axes[0], "mse", "MSE (val)"), (axes[1], "ssim", "SSIM (val)")):
        for m in ROWS + SPECIAL:
            s = df[(df.modelo == m) & (df.split == "val")].sort_values("d")
            if len(s):
                ax.plot(s.d, s[met], "o--" if m in SPECIAL else "o-", color=COLOR[m], label=LABEL[m])
        ax.set_xscale("log", base=2)
        ax.set_xticks(DIMS, DIMS)
        ax.set_xlabel("dimensão do latente d")
        ax.set_ylabel(lab)
        if met == "mse":
            ax.set_yscale("log")
    axes[1].legend()
    fig.suptitle("Reconstrução: PCA vs autoencoders (tracejado: especializados)", y=1.02)
    savefig(fig, "recon_mse_ssim.png", "MSE e SSIM de reconstrução (val) vs d para PCA e AEs")

    ex = np.array([np.where(yv == c)[0][1] for c in range(10)])
    for d in DIMS:
        name = f"recon_panel_d{d}.png"
        if out("figures", name).exists():
            continue
        imgs = [Xv[ex], np.clip(pca_recon_from_images(Xv[ex], d), 0, 1)] + \
               [ae_reconstruct(Xv[ex], a, d) for a in A["archs"]]
        labs = ["original"] + [LABEL[m] for m in ROWS]
        fig, axes = plt.subplots(len(imgs), 10, figsize=(10, 1.1 * len(imgs)))
        for i, im in enumerate(imgs):
            show_images(axes[i], im, SHORT if i == 0 else None)
            axes[i, 0].text(-4, 14, labs[i], ha="right", va="center", fontsize=8)
        fig.suptitle(f"Reconstruções com d = {d} (val)", y=1.0)
        savefig(fig, name, f"painel original vs PCA vs AEs, d={d}")

    # painel compacto: 3 classes difíceis x todos os d
    cls = [6, 2, 9]
    fig, axes = plt.subplots(len(cls) * 5, len(DIMS) + 1, figsize=(7.5, 1.0 * 5 * len(cls)))
    for bi, c in enumerate(cls):
        x = Xv[ex[c]][None]
        for ri, m in enumerate(["original"] + ROWS):
            ax_row = axes[bi * 5 + ri]
            for ci in range(len(DIMS) + 1):
                ax_row[ci].axis("off")
            if m == "original":
                show_images([ax_row[0]], x)
                ax_row[0].set_title(SHORT[c], fontsize=8)
                continue
            ax_row[0].text(0.5, 0.5, LABEL[m], ha="center", va="center", fontsize=7, transform=ax_row[0].transAxes)
            for ci, d in enumerate(DIMS):
                r = np.clip(pca_recon_from_images(x, d), 0, 1) if m == "pca" else ae_reconstruct(x, m, d)
                show_images([ax_row[ci + 1]], r, [f"d={d}"] if ri == 1 else None)
    fig.suptitle("Reconstrução em função de d (Shirt, Pullover, Ankle boot)", y=1.0)
    savefig(fig, "recon_panel_by_d.png", "reconstruções de 3 classes para todos os d e modelos")


# ------------------------------------------------------------------- grelha de embeddings
def embed2d(space, d, Z):
    p = out("viz", f"emb_{space}_d{d}.npy")
    if p.exists():
        return np.load(p)
    if Z.shape[1] == 2:
        E = Z.astype(np.float32)
    else:
        import umap
        E = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=CFG["seed"]) \
            .fit_transform(Z).astype(np.float32)
    save_npy(E, p)
    register(p, f"embedding 2D (UMAP, ou latente direto se d=2) de {space} d={d}, viz10k", {"space": space, "d": d})
    return E


def embeddings_and_clustering():
    ix = viz_indices()
    ytr = load_split("train", mmap=True)[1]
    yva = load_split("val", mmap=True)[1]
    y = ytr[ix]
    mp = out("metrics", "embedding_clustering.csv")
    res = pd.read_csv(mp) if mp.exists() else pd.DataFrame()
    have = set(zip(res.get("espaco", []), res.get("d", [])))
    rows = res.to_dict("records")
    items = [(s, d) for s in ROWS + SPECIAL for d in DIMS] + [("lda", 2), ("lda", 9)]
    for space, d in items:
        if space in list(A["archs"]) + SPECIAL and not ae_ready(space, d):
            continue
        Ztr = latent(space, d, "train")
        if Ztr is None:
            continue
        E = embed2d(space, d, Ztr[ix])
        if (space, d) in have:
            continue
        cm = clustering_metrics(Ztr[ix], y)
        knn = cls_metrics(yva, knn_predict(Ztr, ytr, latent(space, d, "val"), k=5))
        sil2d = clustering_metrics(E, y)["silhouette"]
        rows.append({"espaco": space, "d": d, **cm, "knn_val_acc": knn["accuracy"], "silhouette_2d": sil2d})
        log.info(f"{space} d={d}: {rows[-1]}")
        pd.DataFrame(rows).to_csv(mp, index=False)
    df = pd.DataFrame(rows)
    if df.empty:
        return
    # ranking: média dos postos em silhouette(+), Davies-Bouldin(-), kNN(+), NMI(+), ARI(+)
    r = pd.DataFrame({"sil": df.silhouette.rank(ascending=False), "db": df.davies_bouldin.rank(ascending=True),
                      "knn": df.knn_val_acc.rank(ascending=False), "nmi": df.kmeans_nmi.rank(ascending=False),
                      "ari": df.kmeans_ari.rank(ascending=False)})
    df["posto_medio"] = r.mean(1)
    df = df.sort_values("posto_medio").reset_index(drop=True)
    df["ranking"] = np.arange(1, len(df) + 1)
    df["specialized"] = df.espaco.isin(SPECIAL)
    df.to_csv(mp, index=False)
    register(mp, "métricas de agrupamento por classe (silhouette, Davies-Bouldin, kNN, NMI/ARI k-means) e "
             "ranking por posto médio, para PCA, LDA, os 18 AEs base e os especializados (specialized=True)")

    grid_rows = ROWS + SPECIAL + ["lda"]
    fig, axes = plt.subplots(len(grid_rows), len(DIMS), figsize=(2.3 * len(DIMS), 2.4 * len(grid_rows)))
    for ri, space in enumerate(grid_rows):
        dims = DIMS if space != "lda" else [2, 9]
        for ci in range(len(DIMS)):
            ax = axes[ri, ci]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if ci >= len(dims):
                ax.axis("off")
                continue
            d = dims[ci]
            p = out("viz", f"emb_{space}_d{d}.npy")
            if not p.exists():
                ax.text(0.5, 0.5, "por treinar", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                continue
            scatter_classes(ax, np.load(p), y, s=1, alpha=0.5, labels=True, fontsize=5)
            m = df[(df.espaco == space) & (df.d == d)]
            sub = f"\nsil={m.silhouette.iloc[0]:.2f} kNN={m.knn_val_acc.iloc[0]:.3f}" if len(m) else ""
            ax.set_title(f"{LABEL[space]} d={d}{' (direto)' if d == 2 else ''}{sub}", fontsize=7)
    class_legend(fig, ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Separação dos embeddings (UMAP 2D; d=2 sem UMAP), 10k pontos do train", y=1.0)
    fig.tight_layout()
    savefig(fig, "embeddings_grid.png", "grelha UMAP dos embeddings de todos os modelos (linhas = arquitetura, colunas = d)")

    fig, ax = plt.subplots(figsize=(7, 0.24 * len(df) + 1))
    lab = [f"{LABEL[s]} d={d}" for s, d in zip(df.espaco, df.d)]
    col = [COLOR[s] for s in df.espaco]
    ax.barh(range(len(df))[::-1], df.posto_medio, color=col, height=0.7,
            hatch=["////" if s else None for s in df.specialized], edgecolor="#fcfcfb", linewidth=0)
    ax.set_yticks(range(len(df))[::-1], lab, fontsize=6.5)
    ax.set_xlabel("posto médio nas 5 métricas (menor = agrupa melhor)")
    ax.set_title("Ranking da capacidade de agrupar por classe (tracejado: especializados)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=COLOR[k], label=LABEL[k], hatch="////" if k in SPECIAL else None,
                             edgecolor="#fcfcfb") for k in ROWS + SPECIAL + ["lda"]],
              loc="upper right", fontsize=7, frameon=True, facecolor="#fcfcfb", edgecolor="none")
    savefig(fig, "embedding_ranking.png", "ranking dos espaços por capacidade de agrupar por classe")


def main():
    reconstructions()
    embeddings_and_clustering()


if __name__ == "__main__":
    main()
