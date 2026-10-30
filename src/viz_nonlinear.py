"""Fase 3: t-SNE e UMAP 2D no subconjunto estratificado de 10k (train), sobre pixels raw, PCA-50 e o
latente do ConvAE. Varrimento de perplexity {5,30,50} e n_neighbors {5,15,50}. Qualidade medida com
trustworthiness, continuity e preservação de vizinhança kNN (k=10) relativamente ao espaço de entrada.

Adenda: UMAP (os mesmos n_neighbors, sem t-SNE) dos latentes VAE/DAE com d=16 e d=64 e do ConvAE base
com os mesmos d (`viz.specialized_inputs` no config), com as mesmas métricas de qualidade.

Uso: python -m src.viz_nonlinear      (cada embedding salta se já existir; o latente AE só entra
                                        quando o modelo estiver treinado)
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from .data import load_split, viz_indices
from .evaluate import embedding_quality, knn_predict
from .plots import SERIES, class_legend, plt, savefig, scatter_classes
from .utils import CFG, exists, get_logger, out, register, save_npy

V = CFG["viz"]
log = get_logger("viz_nonlinear")


def inputs():
    """Espaços de entrada disponíveis para o subconjunto viz10k: nome -> matriz [10000, D]."""
    ix = viz_indices()
    Xtr, ytr = load_split("train", mmap=True)   # mmap: só os 10k pontos vão para a RAM
    pca = joblib.load(out("pca", "pca_full.joblib"))
    X = np.ascontiguousarray(Xtr[ix])
    res = {"raw": X, "pca50": ((X - pca.mean_) @ pca.components_[:V["pca_dims"]].T).astype(np.float32)}
    for name in [V["ae_for_viz"]] + list(V.get("specialized_inputs", [])):
        lat = out("latents", f"{name}_train.npy")
        if lat.exists():
            res[name] = np.load(lat, mmap_mode="r")[ix].astype(np.float32)
    return res, ytr[ix]


def methods_for(sp):
    """Entradas base: t-SNE e UMAP; entradas da adenda: só UMAP."""
    if sp in V.get("specialized_inputs", []) and sp != V["ae_for_viz"]:
        return (("umap", V["n_neighbors"]),)
    return (("tsne", V["perplexities"]), ("umap", V["n_neighbors"]))


def run_tsne(X, perplexity):
    return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", max_iter=1000,
                random_state=CFG["seed"], n_jobs=4).fit_transform(X).astype(np.float32)


def run_umap(X, n_neighbors):
    import umap
    return umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=CFG["seed"]) \
        .fit_transform(X).astype(np.float32)


def main():
    spaces, y = inputs()
    qual_p = out("metrics", "viz_quality.csv")
    qual = pd.read_csv(qual_p) if qual_p.exists() else pd.DataFrame()
    done = set(zip(qual.get("entrada", []), qual.get("metodo", []), qual.get("param", [])))
    rows = qual.to_dict("records")
    for sp, X in spaces.items():
        for method, params in methods_for(sp):
            for prm in params:
                tag = f"{method}_{sp}_{'p' if method == 'tsne' else 'n'}{prm}"
                p = out("viz", f"{tag}.npy")
                if not p.exists():
                    t0 = time.time()
                    Z = run_tsne(X, prm) if method == "tsne" else run_umap(X, prm)
                    save_npy(Z, p)
                    register(p, f"coordenadas 2D {method} (param={prm}) de {sp}, subconjunto viz10k",
                             {"metodo": method, "param": prm, "entrada": sp})
                    log.info(f"{tag}: {time.time() - t0:.0f}s")
                if (sp, method, prm) in done:
                    continue
                Z = np.load(p)
                q = embedding_quality(X, Z, V["quality_k"])
                acc2d = float((knn_predict(Z, y, Z, k=5, exclude_self=True) == y).mean())
                rows.append({"entrada": sp, "metodo": method, "param": prm, **q, "knn_acc_2d_loo": acc2d})
                log.info(f"{tag}: {q} | kNN-2D {acc2d:.3f}")
                pd.DataFrame(rows).to_csv(qual_p, index=False)
    register(qual_p, "trustworthiness, continuity, preservação kNN (k=10) e accuracy kNN 2D (leave-one-out) "
             "de cada embedding t-SNE/UMAP")
    qual = pd.DataFrame(rows)

    base = [sp for sp in spaces if len(methods_for(sp)) == 2]
    specialized_figures(spaces, y, qual)
    # uma figura por espaço de entrada: linha 1 = t-SNE (perplexities), linha 2 = UMAP (n_neighbors)
    for sp in base:
        fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
        for r, (method, params, lab) in enumerate((("tsne", V["perplexities"], "perplexity"),
                                                    ("umap", V["n_neighbors"], "n_neighbors"))):
            for c, prm in enumerate(params):
                Z = np.load(out("viz", f"{method}_{sp}_{'p' if method == 'tsne' else 'n'}{prm}.npy"))
                q = qual[(qual.entrada == sp) & (qual.metodo == method) & (qual.param == prm)].iloc[0]
                scatter_classes(axes[r, c], Z, y)
                axes[r, c].set_title(f"{'t-SNE' if method == 'tsne' else 'UMAP'} {lab}={prm}\n"
                                     f"T={q.trustworthiness:.3f}  C={q.continuity:.3f}  kNN-pres={q.knn_preservation:.2f}",
                                     fontsize=9)
        class_legend(fig, ncol=5, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"t-SNE e UMAP sobre {sp} (10k pontos do train)", y=1.0)
        fig.tight_layout()
        savefig(fig, f"viz_{sp}.png", f"grelha t-SNE/UMAP sobre {sp}")

    # resumo das métricas
    configs = [("tsne", p) for p in V["perplexities"]] + [("umap", n) for n in V["n_neighbors"]]
    xt = [f"t-SNE\np={p}" if m == "tsne" else f"UMAP\nn={p}" for m, p in configs]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, met in zip(axes, ("trustworthiness", "continuity", "knn_preservation", "knn_acc_2d_loo")):
        for i, sp in enumerate(base):
            v = [qual[(qual.entrada == sp) & (qual.metodo == m) & (qual.param == p)][met].iloc[0] for m, p in configs]
            ax.plot(range(3), v[:3], "o-", color=SERIES[i], label=sp)
            ax.plot(range(3, 6), v[3:], "o-", color=SERIES[i])
        ax.axvline(2.5, color="#bdbcb6", lw=0.8)
        ax.set_xticks(range(6), xt, fontsize=8)
        ax.set_title({"knn_preservation": "preservação kNN@10", "knn_acc_2d_loo": "accuracy kNN no mapa 2D"}.get(met, met))
    axes[0].legend(fontsize=8)
    fig.suptitle("Qualidade dos embeddings 2D por espaço de entrada e configuração", y=1.03)
    fig.tight_layout()
    savefig(fig, "viz_quality.png", "resumo das métricas de qualidade dos embeddings 2D")
    log.info("Fase 3: figuras atualizadas")


def specialized_figures(spaces, y, qual):
    """Adenda: UMAP (n_neighbors=15) de ConvAE / VAE / DAE lado a lado para d=16 e d=64, e as métricas
    de qualidade vs n_neighbors."""
    sel = [sp for sp in V.get("specialized_inputs", []) if sp in spaces]
    if not sel:
        return
    archs = ["conv_ae", "vae_conv", "dae_conv"]
    lab = {"conv_ae": "ConvAE (base)", "vae_conv": "VAE", "dae_conv": "DAE"}
    dims = sorted({int(sp.rsplit("_d", 1)[1]) for sp in sel})
    fig, axes = plt.subplots(len(dims), len(archs), figsize=(4.3 * len(archs), 4.1 * len(dims)), squeeze=False)
    for r, d in enumerate(dims):
        for c, a in enumerate(archs):
            ax, sp = axes[r, c], f"{a}_d{d}"
            if sp not in sel:
                ax.axis("off")
                continue
            Z = np.load(out("viz", f"umap_{sp}_n15.npy"))
            q = qual[(qual.entrada == sp) & (qual.metodo == "umap") & (qual.param == 15)].iloc[0]
            scatter_classes(ax, Z, y)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{lab[a]} d={d}\nT={q.trustworthiness:.3f}  C={q.continuity:.3f}  "
                         f"kNN-pres={q.knn_preservation:.2f}", fontsize=9)
    class_legend(fig, ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("UMAP (n_neighbors=15) dos latentes: ConvAE base vs especializados (10k pontos do train)", y=1.0)
    fig.tight_layout()
    savefig(fig, "viz_specialized.png", "UMAP dos latentes ConvAE/VAE/DAE com d=16 e d=64")

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, met in zip(axes, ("trustworthiness", "continuity", "knn_preservation", "knn_acc_2d_loo")):
        for i, a in enumerate(archs):
            for d, ls in zip(dims, ("-", "--")):
                s_ = qual[(qual.entrada == f"{a}_d{d}") & (qual.metodo == "umap")].sort_values("param")
                if len(s_):
                    ax.plot(range(len(s_)), s_[met], "o" + ls, color=SERIES[i], label=f"{lab[a]} d={d}")
        ax.set_xticks(range(len(V["n_neighbors"])), [f"n={n}" for n in V["n_neighbors"]])
        ax.set_title({"knn_preservation": "preservação kNN@10", "knn_acc_2d_loo": "accuracy kNN no mapa 2D"}.get(met, met))
    axes[0].legend(fontsize=7)
    fig.suptitle("Qualidade do UMAP por modelo e d (linha contínua d=16, tracejada d=64)", y=1.03)
    fig.tight_layout()
    savefig(fig, "viz_quality_specialized.png", "métricas de qualidade do UMAP para ConvAE/VAE/DAE vs n_neighbors")


if __name__ == "__main__":
    main()
