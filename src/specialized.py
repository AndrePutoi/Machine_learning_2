"""Adenda: análises próprias dos autoencoders especializados (Conv-VAE e Conv-DAE).

  - tabela de reconstrução ao lado do ConvAE base (MSE/SSIM por d; VAE: KL e unidades ativas; DAE: MSE
    de denoising) -> outputs/metrics/specialized_summary.csv;
  - painéis de reconstrução original / PCA / ConvAE / VAE / DAE e painel de denoising;
  - amostras do prior N(0, I) do VAE (d ∈ {2, 16, 64}), comparadas com amostras de uma gaussiana ajustada
    aos latentes do ConvAE, e a variedade 2D descodificada do VAE d=2;
  - interpolação latente (exemplo Sneaker -> Ankle boot) e medidas de suavidade em 200 pares do val:
      * cv_passos: coeficiente de variação dos tamanhos dos passos (L2 em pixels) entre frames
        consecutivos (0 = transição uniforme; valores altos = saltos);
      * dist_nn_interior / dist_nn_extremos: distância média (L2 em pixels) de cada frame à imagem do
        train mais próxima; o rácio interior/extremos mede se os frames intermédios são tão plausíveis
        como as próprias reconstruções (desconta o "blur" típico de cada modelo);
  - veredicto das hipóteses H1-H3 com os números -> outputs/metrics/hypotheses.csv (corre no fim).

Uso: python -m src.specialized               (figuras, tabelas e interpolação)
     python -m src.specialized --hypotheses  (depois da classificação, retrieval e robustez)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from .data import load_split
from .evaluate import knn_search
from .plots import SERIES, SHORT, plt, savefig, show_images
from .representation import ae_decode, ae_encode, ae_reconstruct, pca_project, pca_reconstruct
from .utils import CFG, device, get_logger, load_json, out, register

A = CFG["ae"]
SP = CFG["specialized"]
log = get_logger("specialized")
LABEL = {"pixels": "pixels (cross-fade)", "pca": "PCA", "conv_ae": "ConvAE (base)", "vae_conv": "VAE",
         "dae_conv": "DAE"}


def _ready(arch, d):
    return out("ae", f"{arch}_d{d}", "metrics.json").exists()


def all_ready():
    return all(_ready(a, d) for a in ["conv_ae"] + list(A["specialized"]) for d in A["dims"])


def _mosaic(imgs, n=10):
    imgs = np.asarray(imgs).reshape(-1, 28, 28)
    return imgs[: n * n].reshape(n, n, 28, 28).transpose(0, 2, 1, 3).reshape(n * 28, n * 28)


# ------------------------------------------------------------------------- tabela resumo
def summary_table() -> pd.DataFrame:
    rows = []
    for d in A["dims"]:
        for arch in ["conv_ae"] + list(A["specialized"]):
            m = load_json(out("ae", f"{arch}_d{d}", "metrics.json"))
            rows.append({"modelo": arch, "d": d, "params": m["params"], "epochs_run": m["epochs_run"],
                         "best_epoch": m["best_epoch"], "val_mse": m["val_mse"], "val_ssim": m["val_ssim"],
                         "test_mse": m["test_mse"], "test_ssim": m["test_ssim"],
                         "test_kl": m.get("test_kl"), "active_units_val": m.get("val_active_units"),
                         "active_units_test": m.get("test_active_units"),
                         "test_denoise_mse_sigma03": m.get("test_denoise_mse"),
                         "time_per_epoch_s": m["time_per_epoch_s"], "peak_gpu_mem_mb": m["peak_gpu_mem_mb"]})
    df = pd.DataFrame(rows)
    # MSE de denoising (σ=0.3, mesmo ruído) também para o ConvAE base e o VAE, para comparar com o DAE
    Xte = torch.as_tensor(load_split("test", flat=False)[0], device=device())
    g = torch.Generator(device=device())
    g.manual_seed(CFG["seed"] + 1)
    Xn = (Xte + A["dae"]["sigma"] * torch.randn(Xte.shape, device=device(), generator=g)).clamp_(0, 1)
    Xn = Xn.cpu().numpy()
    Xc = Xte.cpu().numpy().reshape(-1, 784)
    for i, r in df.iterrows():
        if r.modelo != "dae_conv":
            df.loc[i, "test_denoise_mse_sigma03"] = float(((ae_reconstruct(Xn, r.modelo, int(r.d)) - Xc) ** 2).mean())
    p = out("metrics", "specialized_summary.csv")
    df.to_csv(p, index=False)
    register(p, "ConvAE base vs VAE vs DAE por d: parâmetros, épocas, MSE/SSIM (val/test), KL e unidades ativas "
             "(VAE), MSE de denoising σ=0.3 no test (os três), tempo/época e memória")
    return df


# ------------------------------------------------------------------------------ painéis
def recon_panels():
    Xv, yv = load_split("val")
    ex = np.array([np.where(yv == c)[0][1] for c in range(10)])   # os mesmos exemplos dos painéis base
    x = Xv[ex]
    for d in SP["sample_dims"]:
        rows = [("original", x), ("PCA", np.clip(pca_reconstruct(pca_project(x, d), d), 0, 1))] + \
               [(LABEL[a], ae_reconstruct(x, a, d)) for a in ["conv_ae"] + list(A["specialized"])]
        fig, axes = plt.subplots(len(rows), 10, figsize=(10, 1.1 * len(rows)))
        for i, (lab, im) in enumerate(rows):
            show_images(axes[i], im, SHORT if i == 0 else None)
            axes[i, 0].text(-4, 14, lab, ha="right", va="center", fontsize=8)
        fig.suptitle(f"Reconstruções com d = {d} (val): base vs especializados", y=1.0)
        savefig(fig, f"recon_panel_specialized_d{d}.png", f"painel original / PCA / ConvAE / VAE / DAE, d={d}")

    # denoising: input ruidoso -> saída de cada modelo (d=64)
    d = 64
    g = torch.Generator(device=device())
    g.manual_seed(CFG["seed"])
    eps = torch.randn(x.shape, device=device(), generator=g).cpu().numpy()
    fig, axes = plt.subplots(9, 10, figsize=(10, 9.9))
    r = 0
    for sigma in (0.3, 0.5):
        xn = np.clip(x + sigma * eps, 0, 1)
        rows = [(f"ruidosa σ={sigma}", xn)] + [(LABEL[a], ae_reconstruct(xn, a, d))
                                             for a in ["conv_ae"] + list(A["specialized"])]
        if sigma == 0.3:
            rows = [("original", x)] + rows
        for lab, im in rows:
            show_images(axes[r], im, SHORT if r == 0 else None)
            axes[r, 0].text(-4, 14, lab, ha="right", va="center", fontsize=8)
            r += 1
    fig.suptitle("Denoising com d = 64: input ruidoso e saída de cada autoencoder", y=0.93)
    savefig(fig, "denoise_panel.png", "painel de denoising (σ=0.3 e 0.5) para ConvAE, VAE e DAE com d=64")


# ------------------------------------------------------------------------------ amostras
def samples():
    dims = SP["sample_dims"]
    fig, axes = plt.subplots(2, len(dims), figsize=(4.2 * len(dims), 8.8))
    for j, d in enumerate(dims):
        rng = np.random.default_rng(CFG["seed"])
        z = rng.standard_normal((100, d)).astype(np.float32)
        axes[0, j].imshow(_mosaic(ae_decode(z, "vae_conv", d)), cmap="gray_r", vmin=0, vmax=1)
        axes[0, j].set_title(f"VAE d={d}: z ~ N(0, I)", fontsize=10)
        # referência: gaussiana com a média e a covariância dos latentes de treino do ConvAE base
        Z = np.load(out("latents", f"conv_ae_d{d}_train.npy"))
        zc = rng.multivariate_normal(Z.mean(0), np.cov(Z, rowvar=False), size=100).astype(np.float32)
        axes[1, j].imshow(_mosaic(ae_decode(zc, "conv_ae", d)), cmap="gray_r", vmin=0, vmax=1)
        axes[1, j].set_title(f"ConvAE d={d}: z ~ gaussiana ajustada\naos latentes de treino", fontsize=10)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Amostras geradas: prior do VAE vs gaussiana ajustada ao latente do ConvAE", y=1.0)
    fig.tight_layout()
    savefig(fig, "vae_samples.png", "amostras 10x10 do prior N(0,I) do VAE (d=2,16,64) e referência do ConvAE")

    # variedade descodificada do VAE d=2 (grelha de quantis da normal)
    n = 15
    q = norm.ppf(np.linspace(0.03, 0.97, n)).astype(np.float32)
    zz = np.stack(np.meshgrid(q, q[::-1]), -1).reshape(-1, 2)
    imgs = ae_decode(zz, "vae_conv", 2).reshape(n, n, 28, 28).transpose(0, 2, 1, 3).reshape(n * 28, n * 28)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(imgs, cmap="gray_r", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title("VAE d=2: imagens descodificadas numa grelha de quantis 3%–97% de N(0, I)", fontsize=10)
    savefig(fig, "vae_manifold_d2.png", "variedade 2D descodificada do VAE d=2")


# --------------------------------------------------------------------------- interpolação
def _paths(method: str, d: int, xa: np.ndarray, xb: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Frames [P, T, 784] da interpolação linear entre xa[p] e xb[p] no espaço do método."""
    T = t[None, :, None]
    if method == "pixels":
        return (1 - T) * xa[:, None] + T * xb[:, None]
    if method == "pca":
        za, zb = pca_project(xa, d), pca_project(xb, d)
        Z = ((1 - T) * za[:, None] + T * zb[:, None]).reshape(-1, d)
        return np.clip(pca_reconstruct(Z, d), 0, 1).reshape(len(xa), len(t), 784)
    za, zb = ae_encode(xa, method, d), ae_encode(xb, method, d)
    Z = ((1 - T) * za[:, None] + T * zb[:, None]).reshape(-1, d).astype(np.float32)
    return ae_decode(Z, method, d).reshape(len(xa), len(t), 784)


def interpolation():
    Xv, yv = load_split("val")
    Xtr = load_split("train")[0]
    t = np.linspace(0, 1, SP["interp_steps"])
    methods = ["pixels", "pca", "conv_ae", "dae_conv", "vae_conv"]

    # exemplos visuais: Sneaker -> Ankle boot e T-shirt -> Dress
    rng = np.random.default_rng(CFG["seed"])
    pairs = [(7, 9), (0, 3)]
    ex = [(int(rng.choice(np.where(yv == a)[0])), int(rng.choice(np.where(yv == b)[0]))) for a, b in pairs]
    for d in SP["interp_dims"]:
        fig, axes = plt.subplots(len(methods) * len(ex), len(t), figsize=(len(t) * 0.95, len(methods) * len(ex) * 1.0))
        for e, (ia, ib) in enumerate(ex):
            for m_i, m in enumerate(methods):
                fr = _paths(m, d, Xv[[ia]], Xv[[ib]], t)[0]
                row = e * len(methods) + m_i
                show_images(axes[row], fr, [f"t={v:.2f}" for v in t] if row == 0 else None, fontsize=6)
                axes[row, 0].text(-4, 14, LABEL[m], ha="right", va="center", fontsize=7)
            axes[e * len(methods), 0].text(-4, -10, f"{SHORT[yv[ia]]} → {SHORT[yv[ib]]}", ha="right", fontsize=8,
                                           fontweight="bold")
        fig.suptitle(f"Interpolação linear no espaço de cada método (d = {d})", y=0.94)
        savefig(fig, f"interpolation_d{d}.png", f"interpolação latente pixels/PCA/ConvAE/DAE/VAE, d={d}")

    # medidas em pares aleatórios do val
    p = out("metrics", "interpolation.csv")
    if p.exists():
        return pd.read_csv(p)
    rng = np.random.default_rng(CFG["seed"] + 7)
    n = SP["interp_pairs"]
    ia = rng.choice(len(Xv), n, replace=False)
    ib = rng.choice(np.setdiff1d(np.arange(len(Xv)), ia), n, replace=False)
    rows = []
    for d in SP["interp_dims"]:
        for m in methods:
            if m == "pixels" and d != SP["interp_dims"][0]:
                rows.append({**rows[0], "d": d})   # não depende de d
                continue
            F = _paths(m, d, Xv[ia], Xv[ib], t)                         # [n, T, 784]
            steps = np.linalg.norm(np.diff(F, axis=1), axis=2)          # [n, T-1]
            cv = steps.std(1) / (steps.mean(1) + 1e-12)
            dnn, _ = knn_search(F.reshape(-1, 784), Xtr, 1)
            dnn = dnn[:, 0].reshape(n, len(t))
            inner, ends = dnn[:, 1:-1].mean(), dnn[:, [0, -1]].mean()
            rows.append({"metodo": m, "d": d, "n_pares": n, "cv_passos": float(cv.mean()),
                         "passo_max_frac": float((steps.max(1) / steps.sum(1)).mean()),
                         "dist_nn_interior": float(inner), "dist_nn_extremos": float(ends),
                         "racio_interior_extremos": float(inner / ends)})
            log.info(f"interpolação {m} d={d}: {rows[-1]}")
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    register(p, "suavidade da interpolação latente em 200 pares do val: CV dos passos, fração do maior passo, "
             "distância ao vizinho real mais próximo (frames interiores vs extremos)",
             {"pares": n, "frames": len(t), "dims": SP["interp_dims"]})
    return df


# ----------------------------------------------------------------------------- hipóteses
def hypotheses() -> pd.DataFrame:
    ae = pd.read_csv(out("metrics", "specialized_summary.csv")).set_index(["modelo", "d"])
    itp = pd.read_csv(out("metrics", "interpolation.csv")).set_index(["metodo", "d"])
    cl = pd.read_csv(out("metrics", "classification.csv"))
    ret = pd.read_csv(out("metrics", "retrieval.csv"))
    ret = ret[ret.metrica == "euclidean"].set_index("espaco")
    rb = pd.read_csv(out("metrics", "robustness.csv")).set_index(["espaco", "sigma_teste"])
    rvc = pd.read_csv(out("metrics", "retrieval_vs_classification.csv"))
    agr = pd.read_csv(out("metrics", "retrieval_rank_agreement.csv"))
    dims = A["dims"]
    rows = []

    def add(h, crit, val, ok):
        rows.append({"hipotese": h, "criterio": crit, "valor": val, "cumprido": ok})

    # H1
    worse = [d for d in dims if ae.loc[("vae_conv", d), "val_mse"] > ae.loc[("conv_ae", d), "val_mse"]]
    h1a = len(worse) >= 5
    add("H1", "(a) MSE val do VAE > ConvAE em ≥5 de 6 d",
        "; ".join(f"d={d}: {ae.loc[('vae_conv', d), 'val_mse']:.4f} vs {ae.loc[('conv_ae', d), 'val_mse']:.4f}"
                  for d in dims) + f" → pior em {len(worse)}/6", h1a)
    h1b = []    # uma entrada por (d, medida): as duas medidas têm de concordar nos dois d
    for d in SP["interp_dims"]:
        v, c = itp.loc[("vae_conv", d)], itp.loc[("conv_ae", d)]
        ok_cv = bool(v.cv_passos < c.cv_passos)
        ok_ra = bool(v.racio_interior_extremos < c.racio_interior_extremos)
        h1b += [ok_cv, ok_ra]
        add("H1", f"(b-i) interpolação d={d}: CV dos passos menor no VAE",
            f"{v.cv_passos:.3f} vs {c.cv_passos:.3f}", ok_cv)
        add("H1", f"(b-ii) interpolação d={d}: rácio interior/extremos menor no VAE",
            f"{v.racio_interior_extremos:.3f} vs {c.racio_interior_extremos:.3f}", ok_ra)
    ok_c = [d for d in dims if ret.loc[f"vae_conv_d{d}", "map"] >= ret.loc[f"conv_ae_d{d}", "map"] - 0.01
            and ret.loc[f"vae_conv_d{d}", "precision@10"] >= ret.loc[f"conv_ae_d{d}", "precision@10"] - 0.01]
    add("H1", "(c, secundário) retrieval: mAP e P@10 do VAE ≥ ConvAE − 1 p.p.",
        "; ".join(f"d={d}: mAP {ret.loc[f'vae_conv_d{d}', 'map']:.3f} vs {ret.loc[f'conv_ae_d{d}', 'map']:.3f}, "
                  f"P@10 {ret.loc[f'vae_conv_d{d}', 'precision@10']:.3f} vs {ret.loc[f'conv_ae_d{d}', 'precision@10']:.3f}"
                  for d in dims) + f" → cumprido em {len(ok_c)}/6", len(ok_c) >= 5)
    # critério fixado em DECISIONS.md: medidas de (b) divergentes -> inconclusiva
    if h1a and all(h1b):
        v1 = "confirmada"
    elif not h1a or not any(h1b):
        v1 = "refutada"
    else:
        v1 = "inconclusiva"

    # H2
    h2a = {}
    for clf in ("knn", "logreg"):
        s = cl[cl.classificador == clf].set_index("espaco")
        okd = [d for d in dims if s.loc[f"dae_conv_d{d}", "test_acc"] >= s.loc[f"conv_ae_d{d}", "test_acc"] - 0.005]
        h2a[clf] = len(okd) >= 5
        add("H2", f"(a) accuracy test {clf}: DAE ≥ ConvAE − 0.5 p.p. em ≥5 de 6 d",
            "; ".join(f"d={d}: {s.loc[f'dae_conv_d{d}', 'test_acc']:.3f} vs {s.loc[f'conv_ae_d{d}', 'test_acc']:.3f}"
                      for d in dims) + f" → {len(okd)}/6", h2a[clf])
    pk = f"precision@{CFG['robustness']['k']}"
    h2b = True
    for sg in (0.3, 0.5):
        dd, cc = rb.loc[("dae_conv_d64", sg)], rb.loc[("conv_ae_d64", sg)]
        ok = bool(dd[pk] - cc[pk] >= 0.03 and dd.queda_precision < cc.queda_precision)
        h2b &= ok
        add("H2", f"(b) robustez σ={sg}: P@10 do DAE ≥ ConvAE + 3 p.p. e queda menor",
            f"P@10 {dd[pk]:.3f} vs {cc[pk]:.3f} (dif. {100 * (dd[pk] - cc[pk]):+.1f} p.p.); queda "
            f"{100 * dd.queda_precision:.1f} vs {100 * cc.queda_precision:.1f} p.p.", ok)
    h2a_all = all(h2a.values())
    v2 = "confirmada" if h2a_all and h2b else ("parcialmente confirmada" if h2a_all or h2b else "refutada")

    # H3
    best_cls = rvc.loc[rvc.logreg.idxmax(), "espaco"]
    best_vis = rvc.loc[rvc["ndcg@10_ssim"].idxmax(), "espaco"]
    a = agr[(agr.subconjunto == "sem_d2") & (agr.criterio_a == "logreg") & (agr.criterio_b == "spearman_ssim")].iloc[0]
    add("H3", "melhor espaço em accuracy LogReg (val) ≠ melhor em NDCG@10-SSIM",
        f"{best_cls} (acc {rvc.logreg.max():.3f}) vs {best_vis} (NDCG {rvc['ndcg@10_ssim'].max():.3f})", best_cls != best_vis)
    add("H3", "|ρ| Spearman entre espaços (accuracy LogReg vs Spearman(−dist, SSIM), sem d=2) < 0.3",
        f"ρ = {a.spearman_entre_espacos:.3f} (p = {a.p_valor:.2f}, {int(a.n_espacos)} espaços)",
        abs(a.spearman_entre_espacos) < 0.3)
    v3 = "confirmada" if best_cls != best_vis and abs(a.spearman_entre_espacos) < 0.3 else "refutada"

    df = pd.DataFrame(rows)
    df["veredicto"] = df.hipotese.map({"H1": v1, "H2": v2, "H3": v3})
    p = out("metrics", "hypotheses.csv")
    df.to_csv(p, index=False)
    register(p, "hipóteses H1-H3 da adenda: critérios fixados à partida, números e veredicto")
    log.info(f"\n{df.to_string(index=False)}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypotheses", action="store_true")
    a = ap.parse_args()
    if a.hypotheses:
        hypotheses()
        return
    if not all_ready():
        log.info("à espera dos 12 modelos especializados")
        return
    summary_table()
    recon_panels()
    samples()
    interpolation()
    log.info("análises dos especializados concluídas")


if __name__ == "__main__":
    main()
