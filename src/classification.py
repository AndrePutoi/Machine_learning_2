"""Fase 5: os mesmos classificadores em todos os espaços, para comparar representações.

Espaços: pixels raw, PCA(k) com e sem whitening, LDA, IncrementalPCA/KernelPCA (extra, já calculados),
AE(arch, d), os AEs especializados da adenda (VAE: latente = mu; DAE) e o embedding da CNN supervisionada. Classificadores: kNN (k=5, GPU), regressão logística
(scikit-learn, com standardização) e um MLP pequeno (GPU). Teto de referência: a CNN ponta a ponta.

Uso: python -m src.classification   (resultados em cache por (espaço, classificador))
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import load_split
from .evaluate import cls_metrics, confusion, knn_predict, logreg
from .plots import SEQ, SERIES, SHORT, heatmap, plt, savefig
from .utils import CFG, CLASSES, device, get_logger, load_json, out, register, set_seed

C = CFG["classification"]
A = CFG["ae"]
log = get_logger("classification")


def spaces():
    """nome -> (família, dimensão, caminhos train/val/test)."""
    S = {"raw": ("raw", 784, [out("data", f"X_{s}_flat.npy") for s in ("train", "val", "test")])}
    for k in CFG["pca"]["ks"]:
        S[f"pca_k{k}"] = ("pca", k, [out("pca", "proj", f"pca_k{k}_{s}.npy") for s in ("train", "val", "test")])
        S[f"pca_k{k}_whiten"] = ("pca_whiten", k, [out("pca", "proj", f"pca_k{k}_whiten_{s}.npy")
                                                  for s in ("train", "val", "test")])
    S["lda"] = ("lda", 9, [out("lda", f"lda_{s}.npy") for s in ("train", "val", "test")])
    S["ipca_k128"] = ("ipca", 128, [out("pca", "proj", f"ipca_k128_{s}.npy") for s in ("train", "val", "test")])
    S["kpca_rbf_k128"] = ("kpca_rbf", 128, [out("pca", "proj", f"kpca_rbf_k128_{s}.npy") for s in ("train", "val", "test")])
    for a in list(A["archs"]) + list(A["specialized"]):
        for d in A["dims"]:
            S[f"{a}_d{d}"] = (a, d, [out("latents", f"{a}_d{d}_{s}.npy") for s in ("train", "val", "test")])
    S["cnn_emb"] = ("cnn_supervisionada", 128, [out("cnn", f"cnn_emb_{s}.npy") for s in ("train", "val", "test")])
    # descritores clássicos da Fase 6 (extra: permitem comparar classificação e retrieval nos mesmos espaços)
    for n, dim in (("silhouette", 784), ("sobel", 784), ("hog", 1296), ("lbp", 160)):
        S[n] = ("descritor", dim, [out("features", f"{n}_{s}.npy") for s in ("train", "val", "test")])
    return S


class _MLP(nn.Module):
    def __init__(self, din, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, 10))

    def forward(self, x):
        return self.net(x)


def mlp_fit_predict(Xtr, ytr, Xs, max_epochs=40, patience=5, bs=512):
    """MLP 1 camada escondida (256), Adam 1e-3, early stopping na accuracy de validação (Xs[0] = val)."""
    set_seed()
    dev = device()
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    T = lambda X: torch.as_tensor((X - mu) / sd, dtype=torch.float32, device=dev)  # noqa: E731
    xtr, ytr_t = T(Xtr), torch.as_tensor(ytr, device=dev)
    xs = [T(X) for X in Xs]
    model = _MLP(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator(device=dev)
    gen.manual_seed(CFG["seed"])
    best, best_state, bad = -1, None, 0
    yval = torch.as_tensor(load_split("val", mmap=True)[1], device=dev)
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=dev, generator=gen)
        for s in range(0, len(xtr), bs):
            ix = perm[s:s + bs]
            loss = nn.functional.cross_entropy(model(xtr[ix]), ytr_t[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(xs[0]).argmax(1) == yval).float().mean().item()
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    with torch.no_grad():
        return [model(x).argmax(1).cpu().numpy() for x in xs]


def run():
    S = spaces()
    y = {s: load_split(s, mmap=True)[1] for s in ("train", "val", "test")}
    res_p = out("metrics", "classification.csv")
    res = pd.read_csv(res_p) if res_p.exists() else pd.DataFrame()
    done = set(zip(res.get("espaco", []), res.get("classificador", [])))
    rows = res.to_dict("records")
    for name, (fam, dim, paths) in S.items():
        if not all(p.exists() for p in paths):
            log.info(f"{name}: em falta, a saltar")
            continue
        todo = [c for c in ("knn", "logreg", "mlp") if (name, c) not in done]
        if not todo:
            continue
        Xtr, Xva, Xte = (np.load(p).astype(np.float32) for p in paths)
        for clf in todo:
            t0 = time.time()
            if clf == "knn":
                pv = knn_predict(Xtr, y["train"], Xva, k=C["knn_k"])
                pt = knn_predict(Xtr, y["train"], Xte, k=C["knn_k"])
            elif clf == "logreg":
                m = logreg(Xtr, y["train"])
                pv, pt = m.predict(Xva), m.predict(Xte)
            else:
                pv, pt = mlp_fit_predict(Xtr, y["train"], [Xva, Xte])
            mv, mt = cls_metrics(y["val"], pv), cls_metrics(y["test"], pt)
            rows.append({"espaco": name, "familia": fam, "dim": dim, "classificador": clf,
                         "val_acc": mv["accuracy"], "val_macro_f1": mv["macro_f1"],
                         "test_acc": mt["accuracy"], "test_macro_f1": mt["macro_f1"], "tempo_s": time.time() - t0})
            cm_p = out("metrics", "confusion", f"{name}_{clf}_test.csv")
            pd.DataFrame(confusion(y["test"], pt), index=CLASSES, columns=CLASSES).to_csv(cm_p)
            log.info(f"{name} [{clf}] val {mv['accuracy']:.4f} test {mt['accuracy']:.4f} ({time.time() - t0:.0f}s)")
            pd.DataFrame(rows).to_csv(res_p, index=False)
    df = pd.DataFrame(rows)
    cnn_m = out("cnn", "metrics.json")
    if cnn_m.exists() and "cnn_ponta_a_ponta" not in set(df.espaco):
        m = load_json(cnn_m)
        df.loc[len(df)] = {"espaco": "cnn_ponta_a_ponta", "familia": "teto_cnn", "dim": 128, "classificador": "cnn",
                           "val_acc": m["val_accuracy"], "val_macro_f1": m["val_macro_f1"],
                           "test_acc": m["test_accuracy"], "test_macro_f1": m["test_macro_f1"], "tempo_s": m["train_time_s"]}
    df.to_csv(res_p, index=False)
    register(res_p, "tabela principal: accuracy e macro-F1 (val e test) por espaço, dimensão e classificador")
    register(out("metrics", "confusion", "x").parent, "matrizes de confusão (test) por espaço e classificador", kind="dir")
    return df


def saturation(df):
    """Para cada família e classificador: menor d com accuracy (val) a <= 0.5 p.p. do máximo da família."""
    fams = ["pca", "pca_whiten"] + list(A["archs"]) + list(A["specialized"])
    rows = []
    for clf in ("knn", "logreg", "mlp"):
        for f in fams:
            s = df[(df.familia == f) & (df.classificador == clf)].sort_values("dim")
            if s.empty:
                continue
            best = s.val_acc.max()
            rows.append({"familia": f, "classificador": clf, "acc_max_val": best,
                         "d_max": int(s.dim[s.val_acc.idxmax()]),
                         "d_saturacao": int(s[s.val_acc >= best - 0.005].dim.min()),
                         "specialized": f in A["specialized"]})
    sat = pd.DataFrame(rows)
    p = out("metrics", "saturation.csv")
    sat.to_csv(p, index=False)
    register(p, "d de saturação da accuracy (val) por família de representação e classificador")
    return sat


def figures(df):
    fams = [("pca", "PCA"), ("pca_whiten", "PCA + whitening")] + [(a, a.replace("_ae", "").capitalize() + "AE")
                                                                  for a in A["archs"]]
    spec = {"vae_conv": ("VAE (conv)", SERIES[6]), "dae_conv": ("DAE (conv)", "#8c5a2b")}
    raw = df[df.espaco == "raw"]
    ceil = df[df.espaco == "cnn_ponta_a_ponta"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, clf, t in zip(axes, ("knn", "logreg", "mlp"), ("kNN (k=5)", "Regressão logística", "MLP (256)")):
        for i, (f, lab) in enumerate(fams):
            s = df[(df.familia == f) & (df.classificador == clf)].sort_values("dim")
            if len(s):
                ax.plot(s.dim, s.val_acc, "o-", color=SERIES[i], label=lab, ms=4)
        for f, (lab, c) in spec.items():
            s = df[(df.familia == f) & (df.classificador == clf)].sort_values("dim")
            if len(s):
                ax.plot(s.dim, s.val_acc, "s--", color=c, label=lab, ms=4, lw=1.6)
        r = raw[raw.classificador == clf]
        if len(r):
            ax.axhline(r.val_acc.iloc[0], color="#7a7a7a", ls="--", lw=1.2)
            ax.text(2.1, r.val_acc.iloc[0] + 0.004, "pixels raw", fontsize=7, color="#52514e")
        lda = df[(df.espaco == "lda") & (df.classificador == clf)]
        if len(lda):
            ax.plot(9, lda.val_acc.iloc[0], "D", color=SERIES[5], ms=7, label="LDA (9)")
        if len(ceil):
            ax.axhline(ceil.val_acc.iloc[0], color=SERIES[7], ls=":", lw=1.5)
            ax.text(2.1, ceil.val_acc.iloc[0] + 0.004, "CNN supervisionada (teto)", fontsize=7, color="#52514e")
        ax.set_xscale("log", base=2)
        ax.set_xticks(A["dims"], A["dims"])
        ax.set_xlabel("dimensão (k ou d)")
        ax.set_title(t)
    axes[0].set_ylabel("accuracy (val)")
    axes[0].set_ylim(0.45, 0.96)
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Curvas de saturação: accuracy vs dimensão da representação", y=1.02)
    savefig(fig, "classification_saturation.png", "curvas de saturação da accuracy (val) por família e classificador")

    # matrizes de confusão (test, LogReg) para espaços representativos
    ae = df[(df.familia.isin(A["archs"])) & (df.classificador == "logreg")]
    best_ae = ae.loc[ae.val_acc.idxmax(), "espaco"] if len(ae) else None
    sel = [("raw", "logreg"), ("pca_k64", "logreg"), ("lda", "logreg")] + ([(best_ae, "logreg")] if best_ae else [])
    fig, axes = plt.subplots(1, len(sel) + 1, figsize=(4.3 * (len(sel) + 1), 4.2))
    for ax, (sp, clf) in zip(axes, sel):
        M = pd.read_csv(out("metrics", "confusion", f"{sp}_{clf}_test.csv"), index_col=0).values
        heatmap(ax, M / M.sum(1, keepdims=True), SHORT, SHORT, fmt="{:.2f}", cmap=SEQ, vmin=0, vmax=1, fontsize=5)
        acc = df[(df.espaco == sp) & (df.classificador == clf)].test_acc.iloc[0]
        ax.set_title(f"{sp} + LogReg\ntest acc {acc:.3f}", fontsize=9)
    if out("metrics", "cnn_confusion_test.csv").exists():
        M = pd.read_csv(out("metrics", "cnn_confusion_test.csv"), index_col=0).values
        heatmap(axes[-1], M / M.sum(1, keepdims=True), SHORT, SHORT, fmt="{:.2f}", cmap=SEQ, vmin=0, vmax=1, fontsize=5)
        axes[-1].set_title(f"CNN supervisionada\ntest acc {ceil.test_acc.iloc[0]:.3f}", fontsize=9)
    fig.tight_layout()
    savefig(fig, "classification_confusions.png", "matrizes de confusão normalizadas (test) de espaços representativos")


def main():
    df = run()
    saturation(df)
    figures(df)


if __name__ == "__main__":
    main()
