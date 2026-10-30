"""Fase 1: EDA mínima para confirmar as hipóteses.

- imagem média por classe;
- distância euclidiana média intra- vs inter-classe em espaço de pixels (matriz 10x10, train);
- kNN (k=5) em pixels raw, train -> val, com matriz de confusão: as classes de tronco
  (T-shirt, Pullover, Coat, Shirt) são de facto as mais confundidas?

Uso: python -m src.eda
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .data import load_split
from .evaluate import cls_metrics, confusion, knn_predict
from .plots import SEQ, SHORT, heatmap, plt, savefig, show_images
from .utils import CFG, CLASSES, device, exists, get_logger, out, register, save_json, save_npy

TOPS = {0, 2, 4, 6}  # T-shirt/top, Pullover, Coat, Shirt


@torch.no_grad()
def class_distance_matrix(X, y, block=2048):
    """Média das distâncias euclidianas entre todos os pares (i de classe a, j de classe b), sem auto-pares."""
    dev = device()
    G = torch.as_tensor(X, device=dev)
    Og = torch.nn.functional.one_hot(torch.as_tensor(y, device=dev), 10).float()
    S = torch.zeros(10, 10, device=dev, dtype=torch.float64)
    for s in range(0, len(X), block):
        D = torch.cdist(G[s:s + block], G)
        S += (Og[s:s + block].T @ D @ Og).double()
    n = np.bincount(y, minlength=10).astype(np.float64)
    cnt = np.outer(n, n) - np.diag(n)
    return S.cpu().numpy() / cnt


def main():
    log = get_logger("eda")
    files = [out("metrics", f) for f in ("eda_class_distances.csv", "eda_knn_raw_confusion.csv",
                                         "eda_confused_pairs.csv", "eda_summary.json")]
    if exists(*files):
        log.info("EDA já feita; a saltar")
        return
    Xtr, ytr = load_split("train")
    Xva, yva = load_split("val")

    # 1) imagem média por classe
    means = np.stack([Xtr[ytr == c].mean(0) for c in range(10)])
    save_npy(means.astype(np.float32), out("metrics", "eda_class_means.npy"))
    fig, axes = plt.subplots(1, 10, figsize=(12, 1.8))
    show_images(axes, means, SHORT, fontsize=9)
    fig.suptitle("Imagem média por classe (train)", y=1.05)
    savefig(fig, "eda_mean_images.png", "imagem média por classe")

    # 2) distâncias intra vs inter
    Dm = class_distance_matrix(Xtr, ytr)
    pd.DataFrame(Dm, index=CLASSES, columns=CLASSES).to_csv(files[0])
    intra = np.diag(Dm)
    inter = np.array([np.delete(Dm[c], c).mean() for c in range(10)])
    nearest = [CLASSES[np.argsort(Dm[c])[1] if np.argsort(Dm[c])[0] == c else np.argsort(Dm[c])[0]]
               for c in range(10)]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = heatmap(ax, Dm, SHORT, SHORT, fmt="{:.1f}")
    fig.colorbar(im, ax=ax, shrink=0.8, label="distância euclidiana média")
    ax.set_title("Distância média entre classes em espaço de pixels (diagonal = intra-classe)")
    savefig(fig, "eda_class_distances.png", "matriz 10x10 de distâncias médias intra/inter-classe (pixels)")

    # 3) kNN em pixels raw
    pred = knn_predict(Xtr, ytr, Xva, k=CFG["eda"]["knn_k"])
    m = cls_metrics(yva, pred)
    C = confusion(yva, pred)
    pd.DataFrame(C, index=CLASSES, columns=CLASSES).to_csv(files[1])
    off = C.copy().astype(float)
    np.fill_diagonal(off, 0)
    sym = off + off.T
    pairs = [(sym[i, j], i, j) for i in range(10) for j in range(i + 1, 10)]
    pairs.sort(reverse=True)
    top = pd.DataFrame([{"classe_a": CLASSES[i], "classe_b": CLASSES[j], "confusoes": int(v),
                         "ambas_tronco": (i in TOPS and j in TOPS)} for v, i, j in pairs[:10]])
    top.to_csv(files[2], index=False)
    per_class_acc = C.diagonal() / C.sum(1)

    Cn = C / C.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    heatmap(ax, Cn, SHORT, SHORT, fmt="{:.2f}", cmap=SEQ, vmin=0, vmax=1)
    ax.set_xlabel("previsto")
    ax.set_ylabel("verdadeiro")
    ax.set_title(f"kNN (k=5) em pixels raw, val: accuracy {m['accuracy']:.3f}")
    savefig(fig, "eda_knn_raw_confusion.png", "matriz de confusão normalizada do kNN em pixels raw (val)")

    top5_tops = int(top.head(5)["ambas_tronco"].sum())
    worst4 = [CLASSES[c] for c in np.argsort(per_class_acc)[:4]]
    # fração de todos os erros que são trocas entre duas classes de tronco
    tops = sorted(TOPS)
    frac_err_tops = float(off[np.ix_(tops, tops)].sum() / off.sum())
    # critério: os 4 pares mais confundidos são todos entre classes de tronco
    confirmed = bool(top.head(4)["ambas_tronco"].all())
    summary = {"knn_raw_val": m, "acc_por_classe": dict(zip(CLASSES, per_class_acc.round(4).tolist())),
               "4_piores_classes": worst4, "pares_tronco_no_top5_confusoes": top5_tops,
               "fracao_erros_entre_classes_tronco": frac_err_tops,
               "distancia_intra": dict(zip(CLASSES, intra.round(3).tolist())),
               "distancia_inter_media": dict(zip(CLASSES, inter.round(3).tolist())),
               "classe_mais_proxima": dict(zip(CLASSES, nearest)),
               "hipotese_tronco_confirmada": bool(confirmed)}
    save_json(summary, files[3])
    register(files, "EDA: distâncias intra/inter, confusão do kNN raw, pares mais confundidos, resumo")
    register(out("metrics", "eda_class_means.npy"), "imagens médias por classe (train)")
    log.info(f"kNN raw val: {m}")
    log.info(f"piores classes: {worst4}; pares de tronco no top-5 de confusões: {top5_tops}/5")
    log.info(f"pares mais confundidos:\n{top.to_string(index=False)}")
    log.info(f"fração dos erros entre classes de tronco: {frac_err_tops:.3f}; hipótese confirmada: {confirmed}")


if __name__ == "__main__":
    main()
