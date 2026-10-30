"""Fase 2: PCA e afins (PCA completo, whitening, PCA por classe, IncrementalPCA, KernelPCA RBF, LDA, TwoNN).

Todos os modelos são ajustados apenas no split train; val serve para comparar; test só é projetado
(para fases seguintes) e usado nas tabelas finais. Cada passo salta se os artefactos já existirem.

Uso: python -m src.pca_lda
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .data import load_split, stratified_subset
from .evaluate import cls_metrics, knn_predict, logreg, twonn
from .plots import DIV, SERIES, SHORT, plt, savefig, show_images
from .utils import CFG, CLASSES, exists, get_logger, out, register, save_json, save_npy, set_seed

P = CFG["pca"]
log = get_logger("pca_lda")
SPLITS = ("train", "val", "test")


def pca_name(k: int, whiten: bool) -> str:
    return f"pca_k{k}" + ("_whiten" if whiten else "")


def fit_pca(X, k=None, whiten=False):
    # 'covariance_eigh' é exato e muito mais rápido quando n_amostras >> n_features (54000 x 784)
    return PCA(n_components=k, whiten=whiten, svd_solver="covariance_eigh", random_state=CFG["seed"]).fit(X)


def recon_mse(pca_full, X, k):
    W = pca_full.components_[:k]
    Xc = X - pca_full.mean_
    R = Xc @ W.T @ W + pca_full.mean_
    return float(((R - X) ** 2).mean()), R


# ------------------------------------------------------------------------------ PCA completo
def step_full(D):
    model_p = out("pca", "pca_full.joblib")
    ev_p = out("metrics", "pca_explained_variance.csv")
    th_p = out("metrics", "pca_components_thresholds.csv")
    if exists(model_p, ev_p, th_p):
        return joblib.load(model_p)
    Xtr = D["train"][0]
    t0 = time.time()
    pca = fit_pca(Xtr)
    log.info(f"PCA completo ajustado em {time.time() - t0:.1f}s")
    joblib.dump(pca, model_p)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    pd.DataFrame({"componente": np.arange(1, len(evr) + 1), "variancia_explicada": evr,
                  "variancia_acumulada": cum}).to_csv(ev_p, index=False)
    th = pd.DataFrame([{"limiar": t, "n_componentes": int(np.searchsorted(cum, t) + 1)} for t in P["var_thresholds"]])
    th.to_csv(th_p, index=False)
    log.info("componentes por limiar de variância: " + ", ".join(f"{r.limiar:.0%}={r.n_componentes}" for r in th.itertuples()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(np.arange(1, 101), evr[:100], color=SERIES[0])
    axes[0].set_xlabel("componente")
    axes[0].set_ylabel("fração de variância")
    axes[0].set_title("Variância explicada (primeiros 100)")
    axes[1].plot(np.arange(1, len(cum) + 1), cum, color=SERIES[0])
    for r in th.itertuples():
        axes[1].axhline(r.limiar, color="#bdbcb6", lw=0.8, ls="--")
        axes[1].annotate(f"{r.limiar:.0%}: k={r.n_componentes}", (r.n_componentes, r.limiar),
                         xytext=(8, -12), textcoords="offset points", fontsize=8, color="#52514e")
        axes[1].plot(r.n_componentes, r.limiar, "o", color=SERIES[0], ms=5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("nº de componentes (escala log)")
    axes[1].set_ylabel("variância acumulada")
    axes[1].set_title("Variância acumulada")
    savefig(fig, "pca_variance.png", "variância explicada e acumulada do PCA")
    register([model_p], "PCA completo (784 componentes) ajustado no train")
    register([ev_p, th_p], "variância explicada/acumulada e nº de componentes para 80/90/95/99%")
    return pca


def step_reconstruction(D, pca):
    p = out("metrics", "pca_reconstruction.csv")
    if exists(p):
        return
    rows = []
    for k in P["ks"]:
        r = {"k": k}
        for s in ("val", "test"):
            r[f"{s}_mse"], _ = recon_mse(pca, D[s][0], k)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    log.info(f"MSE de reconstrução PCA:\n{df.to_string(index=False)}")

    # painel por classe: 1 exemplo de val por classe; linhas = original + cada k
    Xv, yv = D["val"]
    ex = np.array([np.where(yv == c)[0][0] for c in range(10)])
    rows_img = [Xv[ex]] + [recon_mse(pca, Xv[ex], k)[1] for k in P["ks"]]
    fig, axes = plt.subplots(len(rows_img), 10, figsize=(10, 1.05 * len(rows_img)))
    for i, imgs in enumerate(rows_img):
        show_images(axes[i], np.clip(imgs, 0, 1), SHORT if i == 0 else None)
        axes[i, 0].text(-6, 14, "original" if i == 0 else f"k={P['ks'][i - 1]}", ha="right", va="center", fontsize=8)
    fig.suptitle("Reconstrução PCA por classe (val)", y=1.0)
    savefig(fig, "pca_reconstructions.png", "painel de reconstruções PCA por classe para vários k")

    # eigen-fashion
    fig, axes = plt.subplots(2, 8, figsize=(10, 2.8))
    for i, ax in enumerate(axes.ravel()):
        comp = pca.components_[i].reshape(28, 28)
        v = np.abs(comp).max()
        ax.imshow(comp, cmap=DIV, vmin=-v, vmax=v)
        ax.set_title(f"PC{i + 1} ({pca.explained_variance_ratio_[i]:.1%})", fontsize=7)
        ax.axis("off")
    fig.suptitle("Eigen-fashion: primeiros 16 componentes principais (azul +, laranja -)", y=1.02)
    savefig(fig, "pca_eigenfashion.png", "primeiros componentes principais como imagens")

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(df.k, df.val_mse, "o-", color=SERIES[0])
    ax.set_xscale("log", base=2)
    ax.set_xticks(P["ks"], P["ks"])
    ax.set_xlabel("k (nº de componentes)")
    ax.set_ylabel("MSE de reconstrução (val)")
    ax.set_title("Erro de reconstrução do PCA")
    savefig(fig, "pca_reconstruction_mse.png", "MSE de reconstrução do PCA em função de k")
    register(p, "MSE de reconstrução PCA (val/test) para k em {2,...,128}")


def step_whitening(D):
    p = out("metrics", "pca_knn_whitening.csv")
    if exists(p):
        return pd.read_csv(p)
    rows = []
    Xtr, ytr = D["train"]
    for k in P["ks"]:
        for w in (False, True):
            name = pca_name(k, w)
            mp = out("pca", f"{name}.joblib")
            projs = [out("pca", "proj", f"{name}_{s}.npy") for s in SPLITS]
            if exists(mp, *projs):
                pca = joblib.load(mp)
            else:
                pca = fit_pca(Xtr, k, w)
                joblib.dump(pca, mp)
                for s, pp in zip(SPLITS, projs):
                    save_npy(pca.transform(D[s][0]).astype(np.float32), pp)
                register([mp], f"modelo PCA k={k}, whitening={w}", {"k": k, "whiten": w})
                register(projs, f"projeções train/val/test do {name}", {"k": k, "whiten": w})
            Ztr, Zva = np.load(projs[0]), np.load(projs[1])
            m = cls_metrics(D["val"][1], knn_predict(Ztr, ytr, Zva, k=P["knn_k"]))
            rows.append({"k": k, "whiten": w, "knn_val_acc": m["accuracy"], "knn_val_macro_f1": m["macro_f1"]})
            log.info(f"{name}: kNN val acc {m['accuracy']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for w, c, lab in ((False, SERIES[0], "sem whitening"), (True, SERIES[1], "com whitening")):
        s = df[df.whiten == w]
        ax.plot(s.k, s.knn_val_acc, "o-", color=c, label=lab)
        ax.annotate(lab, (s.k.iloc[-1], s.knn_val_acc.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color="#52514e", va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(P["ks"], P["ks"])
    ax.set_xlabel("k (nº de componentes)")
    ax.set_ylabel("accuracy kNN (k=5), val")
    ax.set_title("PCA com e sem whitening")
    ax.legend(loc="lower center")
    savefig(fig, "pca_knn_whitening.png", "accuracy kNN (val) do PCA com e sem whitening")
    register(p, "accuracy kNN (k=5, val) do PCA com e sem whitening por k")
    return df


def step_knn_curve(D, pca):
    """Curva fina de saturação da accuracy kNN em função de k (PCA sem whitening)."""
    p = out("metrics", "pca_knn_curve.csv")
    if exists(p):
        return pd.read_csv(p)
    Xtr, ytr = D["train"]
    Ztr_full = (Xtr - pca.mean_) @ pca.components_.T
    Zva_full = (D["val"][0] - pca.mean_) @ pca.components_.T
    rows = []
    for k in [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 187, 256, 400, 784]:
        acc = cls_metrics(D["val"][1], knn_predict(Ztr_full[:, :k], ytr, Zva_full[:, :k], k=P["knn_k"]))["accuracy"]
        rows.append({"k": k, "knn_val_acc": acc})
    df = pd.DataFrame(rows)
    best = df.knn_val_acc.max()
    df["saturado"] = df.knn_val_acc >= best - 0.005
    df.to_csv(p, index=False)
    register(p, "curva fina de accuracy kNN (val) vs nº de componentes PCA")
    return df


# --------------------------------------------------------------------------- PCA por classe
def step_per_class(D, pca):
    p = out("metrics", "pca_per_class.csv")
    mp = out("pca", "pca_per_class.joblib")
    if exists(p, mp):
        return
    Xtr, ytr = D["train"]
    Xv, yv = D["val"]
    models, rows = {}, []
    for c in range(10):
        pc = fit_pca(Xtr[ytr == c])
        models[c] = pc
        cum = np.cumsum(pc.explained_variance_ratio_)
        r = {"classe": CLASSES[c]}
        for t in P["var_thresholds"]:
            r[f"n_{int(t * 100)}"] = int(np.searchsorted(cum, t) + 1)
        for k in P["ks"]:
            r[f"mse_classe_k{k}"], _ = recon_mse(pc, Xv[yv == c], k)
            r[f"mse_global_k{k}"], _ = recon_mse(pca, Xv[yv == c], k)
        rows.append(r)
    joblib.dump(models, mp)
    df = pd.DataFrame(rows)
    gcum = np.cumsum(pca.explained_variance_ratio_)
    g = {"classe": "GLOBAL"}
    for t in P["var_thresholds"]:
        g[f"n_{int(t * 100)}"] = int(np.searchsorted(gcum, t) + 1)
    for k in P["ks"]:
        g[f"mse_global_k{k}"], _ = recon_mse(pca, Xv, k)
        g[f"mse_classe_k{k}"] = float(np.average([df[f"mse_classe_k{k}"][c] for c in range(10)],
                                                  weights=np.bincount(yv, minlength=10)))
    df = pd.concat([df, pd.DataFrame([g])], ignore_index=True)
    df.to_csv(p, index=False)
    log.info(f"PCA por classe (n_95):\n{df[['classe', 'n_95']].to_string(index=False)}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    sub = df.iloc[:10]
    axes[0].barh(SHORT[::-1], sub.n_95[::-1], color=SERIES[0], height=0.6)
    axes[0].axvline(g["n_95"], color=SERIES[1], lw=1.5, ls="--")
    axes[0].text(g["n_95"], 9.6, f" global: {g['n_95']}", color="#52514e", fontsize=8)
    axes[0].set_xlabel("componentes para 95% da variância")
    axes[0].set_title("PCA por classe vs global")
    ks = P["ks"]
    axes[1].plot(ks, [g[f"mse_global_k{k}"] for k in ks], "o-", color=SERIES[0], label="PCA global")
    axes[1].plot(ks, [g[f"mse_classe_k{k}"] for k in ks], "o-", color=SERIES[1], label="PCA por classe (oráculo)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(ks, ks)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("MSE de reconstrução (val)")
    axes[1].set_title("Reconstrução: global vs por classe")
    axes[1].legend()
    savefig(fig, "pca_per_class.png", "PCA por classe vs global: componentes para 95% e MSE")
    register([p, mp], "PCA por classe: nº componentes por limiar, MSE vs global; modelos por classe")


# --------------------------------------------------------------------------- IncrementalPCA
def step_ipca(D, pca):
    p = out("metrics", "ipca_vs_pca.csv")
    mp = out("pca", "ipca_k128.joblib")
    projs = [out("pca", "proj", f"ipca_k128_{s}.npy") for s in SPLITS]
    if exists(p, mp, *projs):
        return
    Xtr, ytr = D["train"]
    n = P["ipca_components"]
    t0 = time.time()
    ipca = IncrementalPCA(n_components=n, batch_size=P["ipca_batch"])
    for s in range(0, len(Xtr), P["ipca_batch"]):
        blk = Xtr[s:s + P["ipca_batch"]]
        if len(blk) >= n:
            ipca.partial_fit(blk)
    t_ipca = time.time() - t0
    t0 = time.time()
    fit_pca(Xtr, n)
    t_pca = time.time() - t0
    joblib.dump(ipca, mp)
    for s, pp in zip(SPLITS, projs):
        save_npy(ipca.transform(D[s][0]).astype(np.float32), pp)
    Ztr, Zva = np.load(projs[0]), np.load(projs[1])
    rows = []
    for k in P["ks"]:
        W = ipca.components_[:k]
        R = (D["val"][0] - ipca.mean_) @ W.T @ W + ipca.mean_
        angles = subspace_angles(pca.components_[:k].T, W.T)
        acc = cls_metrics(D["val"][1], knn_predict(Ztr[:, :k], ytr, Zva[:, :k], k=P["knn_k"]))["accuracy"]
        rows.append({"k": k, "ipca_var_acum": float(ipca.explained_variance_ratio_[:k].sum()),
                     "pca_var_acum": float(pca.explained_variance_ratio_[:k].sum()),
                     "ipca_val_mse": float(((R - D["val"][0]) ** 2).mean()),
                     "pca_val_mse": recon_mse(pca, D["val"][0], k)[0],
                     "angulo_max_graus": float(np.degrees(angles.max())),
                     "ipca_knn_val_acc": acc})
    df = pd.DataFrame(rows)
    df["tempo_ipca_s"], df["tempo_pca_s"] = t_ipca, t_pca
    df.to_csv(p, index=False)
    log.info(f"IncrementalPCA vs PCA:\n{df.to_string(index=False)}")
    register([mp], "IncrementalPCA (128 componentes, batches de 2048) ajustado no train")
    register(projs, "projeções train/val/test do IncrementalPCA-128")
    register(p, "comparação IncrementalPCA vs PCA (variância, MSE, ângulos entre subespaços, kNN, tempo)")


# ------------------------------------------------------------------------------ KernelPCA
def step_kpca(D):
    p = out("metrics", "kpca_knn.csv")
    mp = out("pca", "kpca_rbf_k128.joblib")
    projs = [out("pca", "proj", f"kpca_rbf_k128_{s}.npy") for s in SPLITS]
    if exists(p, mp, *projs):
        return
    set_seed()
    Xtr, ytr = D["train"]
    sub = stratified_subset(ytr, P["kpca_subset"])
    save_npy(sub, out("pca", "kpca_subset_idx.npy"))
    Xs = np.ascontiguousarray(Xtr[sub], dtype=np.float32)
    # gamma pela heurística da mediana: 1 / mediana(||xi - xj||^2) numa amostra de pares
    rng = np.random.default_rng(CFG["seed"])
    a, b = rng.integers(0, len(Xs), 20000), rng.integers(0, len(Xs), 20000)
    d2 = ((Xs[a] - Xs[b]) ** 2).sum(1)
    gamma = float(1.0 / np.median(d2[d2 > 0]))
    t0 = time.time()
    kpca = KernelPCA(n_components=P["kpca_components"], kernel="rbf", gamma=gamma, eigen_solver="randomized",
                     random_state=CFG["seed"], copy_X=False).fit(Xs)
    t_fit = time.time() - t0
    joblib.dump(kpca, mp)
    for s, pp in zip(SPLITS, projs):
        X = D[s][0]
        Z = np.concatenate([kpca.transform(np.asarray(X[i:i + P["kpca_chunk"]], dtype=np.float32))
                            for i in range(0, len(X), P["kpca_chunk"])])
        save_npy(Z.astype(np.float32), pp)
    Ztr, Zva = np.load(projs[0]), np.load(projs[1])
    rows = []
    for k in P["ks"]:
        acc = cls_metrics(D["val"][1], knn_predict(Ztr[:, :k], ytr, Zva[:, :k], k=P["knn_k"]))["accuracy"]
        rows.append({"k": k, "kpca_knn_val_acc": acc, "gamma": gamma, "subset": len(sub), "tempo_fit_s": t_fit})
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    log.info(f"KernelPCA RBF (gamma={gamma:.4g}, fit {t_fit:.0f}s):\n{df.to_string(index=False)}")
    register([mp, out("pca", "kpca_subset_idx.npy")], "KernelPCA RBF (128 comp.) ajustado num subconjunto "
             "estratificado de 10k do train; índices do subconjunto", {"gamma": gamma})
    register(projs, "projeções train/val/test do KernelPCA RBF-128", {"gamma": gamma})
    register(p, "accuracy kNN (val) do KernelPCA RBF por nº de componentes")


# ------------------------------------------------------------------------------------ LDA
def step_lda(D, pca):
    p = out("metrics", "lda_vs_pca.csv")
    mp = out("lda", "lda.joblib")
    projs = [out("lda", f"lda_{s}.npy") for s in SPLITS]
    if exists(p, mp, *projs):
        return
    Xtr, ytr = D["train"]
    lda = LinearDiscriminantAnalysis(n_components=9, solver="svd").fit(Xtr, ytr)
    joblib.dump(lda, mp)
    for s, pp in zip(SPLITS, projs):
        save_npy(lda.transform(D[s][0]).astype(np.float32), pp)
    L = {s: np.load(pp) for s, pp in zip(SPLITS, projs)}
    Pz = {s: ((D[s][0] - pca.mean_) @ pca.components_[:9].T).astype(np.float32) for s in ("train", "val")}
    rows = []
    for m in range(1, 10):
        for name, Z in (("LDA", L), ("PCA", Pz)):
            knn = cls_metrics(D["val"][1], knn_predict(Z["train"][:, :m], ytr, Z["val"][:, :m], k=P["knn_k"]))
            lr = logreg(Z["train"][:, :m], ytr)
            lrm = cls_metrics(D["val"][1], lr.predict(Z["val"][:, :m]))
            rows.append({"metodo": name, "dim": m, "knn_val_acc": knn["accuracy"], "logreg_val_acc": lrm["accuracy"]})
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    log.info(f"LDA vs PCA:\n{df.pivot(index='dim', columns='metodo', values=['knn_val_acc', 'logreg_val_acc']).to_string()}")
    pd.DataFrame({"componente": np.arange(1, 10), "variancia_explicada_lda": lda.explained_variance_ratio_}).to_csv(
        out("metrics", "lda_explained_variance.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, col, t in ((axes[0], "knn_val_acc", "kNN (k=5)"), (axes[1], "logreg_val_acc", "Regressão logística")):
        for name, c in (("LDA", SERIES[1]), ("PCA", SERIES[0])):
            s = df[df.metodo == name]
            ax.plot(s.dim, s[col], "o-", color=c, label=name)
        ax.set_xlabel("nº de dimensões")
        ax.set_title(t)
        ax.legend()
    axes[0].set_ylabel("accuracy (val)")
    savefig(fig, "lda_vs_pca.png", "LDA vs PCA (1 a 9 dimensões) em kNN e regressão logística")
    register([mp], "LDA (9 componentes) ajustada no train")
    register(projs, "projeções LDA train/val/test (9 dimensões)")
    register([p, out("metrics", "lda_explained_variance.csv")], "LDA vs PCA em kNN/LogReg; variância discriminante")


# ------------------------------------------------------------------------------------ TwoNN
def step_twonn(D, curve):
    p = out("metrics", "twonn.csv")
    if exists(p):
        return
    Xtr, ytr = D["train"]
    rows = [{"conjunto": "GLOBAL (train)", **twonn(Xtr)}]
    for c in range(10):
        rows.append({"conjunto": CLASSES[c], **twonn(Xtr[ytr == c])})
    df = pd.DataFrame(rows)
    k_sat = int(curve[curve.saturado].k.min())
    df["k_saturacao_knn_pca"] = k_sat
    df.to_csv(p, index=False)
    log.info(f"TwoNN:\n{df.to_string(index=False)}")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sub = df.iloc[1:]
    ax.barh(SHORT[::-1], sub.id_twonn.values[::-1], color=SERIES[0], height=0.6)
    ax.axvline(df.id_twonn[0], color=SERIES[1], ls="--", lw=1.5)
    ax.text(df.id_twonn[0], 9.6, f" global: {df.id_twonn[0]:.1f}", fontsize=8, color="#52514e")
    ax.axvline(k_sat, color="#52514e", ls=":", lw=1.5)
    ax.text(k_sat, -0.9, f" saturação kNN-PCA: k={k_sat}", fontsize=8, color="#52514e")
    ax.set_xlabel("dimensão intrínseca estimada (TwoNN)")
    ax.set_title("Dimensão intrínseca global e por classe")
    savefig(fig, "twonn.png", "dimensão intrínseca TwoNN global e por classe vs k de saturação")
    register(p, "dimensão intrínseca TwoNN (ajuste e MLE) global e por classe; k de saturação do kNN-PCA")


def main():
    D = {s: load_split(s) for s in SPLITS}
    pca = step_full(D)
    step_reconstruction(D, pca)
    step_whitening(D)
    curve = step_knn_curve(D, pca)
    step_per_class(D, pca)
    step_lda(D, pca)
    step_twonn(D, curve)
    step_ipca(D, pca)
    step_kpca(D)
    save_json({"concluido_em": __import__("datetime").datetime.now().isoformat()}, out("pca", "_done.json"))
    log.info("Fase 2 concluída")


if __name__ == "__main__":
    main()
