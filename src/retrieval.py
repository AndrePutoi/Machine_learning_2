"""Fase 6: retrieval por semelhança em vários espaços de representação.

    from src.retrieval import retrieve
    res = retrieve(query_image, space="conv_ae_d32", k=10, metric="cosine")
    res["indices"], res["distances"]       # índices no split train (galeria) e distâncias

`space` é qualquer representação registada (ver `available_spaces()`):
  shape    : "raw", "silhouette", "pca_k{k}", "pca_k{k}_whiten", "ipca_k128", "kpca_rbf_k128"
  texture  : "sobel", "hog", "lbp"
  learned  : "dense_ae_d{d}", "conv_ae_d{d}", "transformer_ae_d{d}"
  learned (especializados, adenda): "vae_conv_d{d}" (latente = mu do VAE), "dae_conv_d{d}"
  supervis.: "lda", "cnn" (usam labels no ajuste da representação; ficam como referência)
As labels nunca entram no ranking: só são usadas em `evaluate_all()` para medir precisão.

Uso (avaliação completa + painéis): python -m src.retrieval
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from .data import load_split, stratified_subset
from .evaluate import SSIMGallery, pairwise_dist, silhouette_iou
from .plots import SERIES, SHORT, plt, savefig
from .utils import CFG, device, exists, get_logger, out, register, save_npy

R = CFG["retrieval"]
A = CFG["ae"]
log = get_logger("retrieval")


# ------------------------------------------------------------------ descritores clássicos
def _imgs(X) -> np.ndarray:
    return np.asarray(X, dtype=np.float32).reshape(-1, 28, 28)


def feat_silhouette(X):
    return (_imgs(X) > R["silhouette_threshold"]).astype(np.float32).reshape(-1, 784)


def feat_sobel(X):
    I = _imgs(X)
    gx, gy = ndimage.sobel(I, axis=2, mode="constant"), ndimage.sobel(I, axis=1, mode="constant")
    return np.hypot(gx, gy).reshape(-1, 784).astype(np.float32)


def feat_hog(X):
    from skimage.feature import hog
    return np.stack([hog(im, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2),
                         block_norm="L2-Hys", feature_vector=True) for im in _imgs(X)]).astype(np.float32)


def feat_lbp(X, grid=4):
    """Histogramas de LBP uniforme (P=8, R=1; 10 códigos) numa grelha 4x4 de células de 7x7 px."""
    from skimage.feature import local_binary_pattern
    cs = 28 // grid
    out_ = np.zeros((len(X), grid * grid * 10), dtype=np.float32)
    for n, im in enumerate(_imgs(X)):
        codes = local_binary_pattern((im * 255).astype(np.uint8), P=8, R=1, method="uniform").astype(np.int64)
        h = [np.bincount(codes[i * cs:(i + 1) * cs, j * cs:(j + 1) * cs].ravel(), minlength=10)
             for i in range(grid) for j in range(grid)]
        out_[n] = np.concatenate(h) / (cs * cs)
    return out_


FEATURES: dict[str, Callable] = {"silhouette": feat_silhouette, "sobel": feat_sobel, "hog": feat_hog, "lbp": feat_lbp}


def compute_features():
    for name, fn in FEATURES.items():
        paths = [out("features", f"{name}_{s}.npy") for s in ("train", "val", "test")]
        if exists(*paths):
            continue
        t0 = time.time()
        for s, p in zip(("train", "val", "test"), paths):
            X, _ = load_split(s, mmap=True)
            F = np.concatenate([fn(np.asarray(X[i:i + 5000])) for i in range(0, len(X), 5000)])
            save_npy(F, p)
        register(paths, f"descritor {name} para train/val/test", {"dim": int(F.shape[1])})
        log.info(f"descritor {name}: dim {F.shape[1]}, {time.time() - t0:.0f}s")


# ------------------------------------------------------------------- registo de espaços
@dataclass
class Space:
    name: str
    perspective: str
    paths: list
    embed: Callable[[np.ndarray], np.ndarray]

    def features(self, split="train") -> np.ndarray:
        return np.load(self.paths[("train", "val", "test").index(split)], mmap_mode="r")


def _proj(model_path):
    return lambda X: np.asarray(joblib.load(model_path).transform(np.asarray(X, np.float32).reshape(-1, 784)),
                                dtype=np.float32)


def _registry() -> dict[str, Space]:
    from .representation import ae_encode
    S = {}
    sp = lambda f: [f(s) for s in ("train", "val", "test")]  # noqa: E731
    S["raw"] = Space("raw", "shape", sp(lambda s: out("data", f"X_{s}_flat.npy")),
                     lambda X: np.asarray(X, np.float32).reshape(-1, 784))
    for k in CFG["pca"]["ks"]:
        for w in ("", "_whiten"):
            n = f"pca_k{k}{w}"
            S[n] = Space(n, "shape", sp(lambda s, n=n: out("pca", "proj", f"{n}_{s}.npy")), _proj(out("pca", f"{n}.joblib")))
    S["ipca_k128"] = Space("ipca_k128", "shape", sp(lambda s: out("pca", "proj", f"ipca_k128_{s}.npy")),
                           _proj(out("pca", "ipca_k128.joblib")))
    S["kpca_rbf_k128"] = Space("kpca_rbf_k128", "shape", sp(lambda s: out("pca", "proj", f"kpca_rbf_k128_{s}.npy")),
                               _proj(out("pca", "kpca_rbf_k128.joblib")))
    for n, fn in FEATURES.items():
        S[n] = Space(n, "shape" if n == "silhouette" else "texture", sp(lambda s, n=n: out("features", f"{n}_{s}.npy")), fn)
    for a in list(A["archs"]) + list(A["specialized"]):
        for d in A["dims"]:
            n = f"{a}_d{d}"
            S[n] = Space(n, "learned", sp(lambda s, n=n: out("latents", f"{n}_{s}.npy")),
                         lambda X, a=a, d=d: ae_encode(X, a, d))
    S["lda"] = Space("lda", "supervised", sp(lambda s: out("lda", f"lda_{s}.npy")), _proj(out("lda", "lda.joblib")))
    S["cnn"] = Space("cnn", "supervised", sp(lambda s: out("cnn", f"cnn_emb_{s}.npy")), _cnn_embed)
    return S


def _cnn_embed(X):
    m = _load_cnn()
    x = torch.as_tensor(np.asarray(X, np.float32).reshape(-1, 1, 28, 28), device=device())
    with torch.no_grad():
        return m.embed(x).cpu().numpy()


@lru_cache(maxsize=1)
def _load_cnn():
    from .models import SupervisedCNN
    m = SupervisedCNN().to(device())
    m.load_state_dict(torch.load(out("cnn", "best.pt"), map_location=device(), weights_only=False)["model"])
    return m.eval()


SPACES = _registry()


def available_spaces() -> list[str]:
    return [n for n, s in SPACES.items() if all(p.exists() for p in s.paths)]


@lru_cache(maxsize=4)
def _gallery(space: str) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(SPACES[space].features("train")), dtype=torch.float32, device=device())


@torch.no_grad()
def retrieve(query_image, space: str, k: int = 10, metric: str = "euclidean") -> dict:
    """Top-k da galeria (split train, 54k imagens) para uma ou mais imagens-query no espaço escolhido.
    query_image: [28,28], [784], [N,28,28] ou [N,784] em [0,1]. Devolve índices (no split train) e distâncias."""
    if space not in SPACES:
        raise KeyError(f"espaço desconhecido: {space}. Disponíveis: {available_spaces()}")
    q = SPACES[space].embed(np.asarray(query_image, np.float32).reshape(-1, 784))
    D = pairwise_dist(torch.as_tensor(q, dtype=torch.float32, device=device()), _gallery(space), metric)
    d, i = torch.topk(D, k, dim=1, largest=False)
    single = np.asarray(query_image).size == 784
    res = {"indices": i.cpu().numpy(), "distances": d.cpu().numpy(), "space": space, "metric": metric}
    if single:
        res["indices"], res["distances"] = res["indices"][0], res["distances"][0]
    return res


# --------------------------------------------------------------------------- avaliação
def query_indices():
    p = out("retrieval", "query_idx_test.npy")
    pp = out("retrieval", "proxy_query_idx_test.npy")
    if not exists(p, pp):
        yte = load_split("test", mmap=True)[1]
        q = stratified_subset(yte, R["n_queries"])
        sub = stratified_subset(yte[q], R["proxy_queries"])
        save_npy(q, p)
        save_npy(q[sub], pp)
        register([p, pp], "índices (no test) das queries de retrieval e do subconjunto usado na proxy visual")
    return np.load(p), np.load(pp)


@torch.no_grad()
def label_metrics(Q, G, yq, yg, metric, ks, block=None):
    """precision@k, class-consistency@k (voto maioritário do top-k = classe da query), mAP@k e mAP total."""
    block = block or R["block"]
    dev = device()
    G = torch.as_tensor(np.ascontiguousarray(G), dtype=torch.float32, device=dev)
    yg_t = torch.as_tensor(yg, device=dev)
    kmax = max(ks)
    acc = {f"{m}@{k}": 0.0 for m in ("precision", "consistency", "map") for k in ks}
    acc["map"] = 0.0
    for s in range(0, len(Q), block):
        q = torch.as_tensor(np.ascontiguousarray(Q[s:s + block]), dtype=torch.float32, device=dev)
        yq_t = torch.as_tensor(yq[s:s + block], device=dev)
        D = pairwise_dist(q, G, metric)
        order = torch.argsort(D, dim=1)
        rel = (yg_t[order] == yq_t[:, None]).float()               # [b, n] relevância por posição
        csum = rel.cumsum(1)
        pos = torch.arange(1, rel.shape[1] + 1, device=dev)
        prec_at = csum / pos
        nrel = rel.sum(1)
        acc["map"] += ((prec_at * rel).sum(1) / nrel).sum().item()
        top = yg_t[order[:, :kmax]]
        for k in ks:
            acc[f"precision@{k}"] += rel[:, :k].mean(1).sum().item()
            votes = torch.nn.functional.one_hot(top[:, :k], 10).sum(1)
            acc[f"consistency@{k}"] += (votes.argmax(1) == yq_t).float().sum().item()
            acc[f"map@{k}"] += ((prec_at[:, :k] * rel[:, :k]).sum(1) / k).sum().item()
    return {k_: v / len(Q) for k_, v in acc.items()}


def proxy_cache(Xg_img, Xg_flat, Xq_img, Xq_flat):
    """SSIM e IoU de silhueta de cada query-proxy contra toda a galeria (guardados em disco)."""
    ps, pi = out("retrieval", "proxy_ssim.npy"), out("retrieval", "proxy_iou.npy")
    if not exists(ps, pi):
        sg = SSIMGallery(Xg_img)
        S = np.stack([sg(q).cpu().numpy() for q in Xq_img]).astype(np.float32)
        I = silhouette_iou(Xq_flat, Xg_flat).cpu().numpy().astype(np.float32)
        save_npy(S, ps)
        save_npy(I, pi)
        register([ps, pi], "matrizes [queries-proxy x galeria] de SSIM e IoU de silhueta (proxy visual sem labels)")
    return np.load(ps), np.load(pi)


@torch.no_grad()
def proxy_metrics(Q, G, S, I, metric, pool, ks=(1, 5, 10, 20)):
    """Concordância do ranking do espaço com a proxy visual: SSIM/IoU médios do top-k, NDCG@10 com ganho
    = SSIM (ideal calculado sobre a galeria inteira) e Spearman entre -distância e SSIM/IoU num conjunto
    fixo de 1000 imagens aleatórias da galeria por query."""
    dev = device()
    D = pairwise_dist(torch.as_tensor(np.ascontiguousarray(Q), dtype=torch.float32, device=dev),
                      torch.as_tensor(np.ascontiguousarray(G), dtype=torch.float32, device=dev), metric)
    S_t, I_t = torch.as_tensor(S, device=dev), torch.as_tensor(I, device=dev)
    top = torch.topk(D, max(ks), dim=1, largest=False).indices
    res = {}
    for k in ks:
        res[f"ssim@{k}"] = torch.gather(S_t, 1, top[:, :k]).mean().item()
        res[f"iou@{k}"] = torch.gather(I_t, 1, top[:, :k]).mean().item()
    disc = 1.0 / torch.log2(torch.arange(2, 12, device=dev).float())
    for nm, M in (("ssim", S_t), ("iou", I_t)):
        gain = torch.gather(M, 1, top[:, :10]).clamp(min=0)
        ideal = torch.topk(M, 10, dim=1).values.clamp(min=0)
        res[f"ndcg@10_{nm}"] = ((gain * disc).sum(1) / (ideal * disc).sum(1)).mean().item()

    def spearman(a, b):
        ra, rb = a.argsort(1).argsort(1).float(), b.argsort(1).argsort(1).float()
        ra, rb = ra - ra.mean(1, keepdim=True), rb - rb.mean(1, keepdim=True)
        return ((ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1))).mean().item()

    P = torch.as_tensor(pool, device=dev)
    negd = -torch.gather(D, 1, P)
    res["spearman_ssim"] = spearman(negd, torch.gather(S_t, 1, P))
    res["spearman_iou"] = spearman(negd, torch.gather(I_t, 1, P))
    return res


def evaluate_all():
    compute_features()
    q_idx, pq_idx = query_indices()
    ytr, yte = load_split("train", mmap=True)[1], load_split("test", mmap=True)[1]
    Xtr_img, _ = load_split("train", flat=False, mmap=True)
    Xte_img, _ = load_split("test", flat=False, mmap=True)
    Xtr_flat, _ = load_split("train", mmap=True)
    Xte_flat, _ = load_split("test", mmap=True)
    S, I = proxy_cache(np.asarray(Xtr_img), np.asarray(Xtr_flat), np.asarray(Xte_img[pq_idx]), np.asarray(Xte_flat[pq_idx]))
    rng = np.random.default_rng(CFG["seed"])
    pool = np.stack([rng.choice(len(ytr), 1000, replace=False) for _ in range(len(pq_idx))])

    res_p = out("metrics", "retrieval.csv")
    res = pd.read_csv(res_p) if res_p.exists() else pd.DataFrame()
    done = set(zip(res.get("espaco", []), res.get("metrica", [])))
    rows = res.to_dict("records")
    for name in available_spaces():
        sp = SPACES[name]
        G = None
        for metric in ("euclidean", "cosine"):
            if (name, metric) in done:
                continue
            if G is None:
                G = np.ascontiguousarray(sp.features("train"), dtype=np.float32)
                Qall = np.asarray(sp.features("test"), dtype=np.float32)
            t0 = time.time()
            lm = label_metrics(Qall[q_idx], G, yte[q_idx], ytr, metric, R["ks"])
            pm = proxy_metrics(Qall[pq_idx], G, S, I, metric, pool)
            rows.append({"espaco": name, "perspetiva": sp.perspective, "dim": G.shape[1], "metrica": metric, **lm, **pm})
            log.info(f"{name} [{metric}] P@10 {lm['precision@10']:.3f} mAP {lm['map']:.3f} "
                     f"NDCG-SSIM {pm['ndcg@10_ssim']:.3f} rho-SSIM {pm['spearman_ssim']:.3f} ({time.time() - t0:.1f}s)")
            pd.DataFrame(rows).to_csv(res_p, index=False)
        del G
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df["specialized"] = df.espaco.map(is_specialized)
    df.to_csv(res_p, index=False)
    register(res_p, "métricas de retrieval por (espaço, métrica): precision@k, class-consistency@k, mAP@k, mAP, "
             "SSIM/IoU@k, NDCG@10 e Spearman com a proxy visual (coluna specialized = VAE/DAE da adenda)")
    return df


def is_specialized(space: str) -> bool:
    return space.rsplit("_d", 1)[0] in A["specialized"]


def compare_with_classification(ret: pd.DataFrame):
    """Junta accuracy de classificação (kNN/LogReg, val) às métricas de retrieval e mede a concordância
    entre rankings de espaços: melhor a classificar != melhor em semelhança visual?"""
    from scipy.stats import spearmanr, kendalltau
    cp = out("metrics", "classification.csv")
    if not cp.exists():
        return None
    cl = pd.read_csv(cp)
    name_map = {"cnn": "cnn_emb"}
    cl_p = cl.pivot_table(index="espaco", columns="classificador", values="val_acc")
    r = ret[ret.metrica == "euclidean"].copy()
    r["espaco_cls"] = r.espaco.map(lambda n: name_map.get(n, n))
    r = r.merge(cl_p, left_on="espaco_cls", right_index=True, how="inner")
    keep = ["espaco", "perspetiva", "dim", "knn", "logreg", "precision@10", "map", "ndcg@10_ssim", "ndcg@10_iou",
            "spearman_ssim", "spearman_iou", "ssim@10", "iou@10"]
    r = r[keep].sort_values("logreg", ascending=False)
    r["specialized"] = r.espaco.map(is_specialized)
    p = out("metrics", "retrieval_vs_classification.csv")
    r.to_csv(p, index=False)
    # "sem_d2": os espaços com d=2 são maus em tudo e inflacionam artificialmente a correlação.
    # A silhueta binarizada é excluída das correlações com a proxy IoU (seria circular: mesma binarização).
    corr = []
    for sub, rs in (("todos", r), ("sem_d2", r[r.dim > 2])):
        for a in ("knn", "logreg", "precision@10", "map"):
            for b in ("ndcg@10_ssim", "spearman_ssim", "ndcg@10_iou", "spearman_iou"):
                rr = rs[rs.espaco != "silhouette"] if b.endswith("iou") else rs
                rho, pv = spearmanr(rr[a], rr[b])
                tau, _ = kendalltau(rr[a], rr[b])
                corr.append({"subconjunto": sub, "n_espacos": len(rr), "criterio_a": a, "criterio_b": b,
                             "spearman_entre_espacos": rho, "p_valor": pv, "kendall": tau})
    pc = out("metrics", "retrieval_rank_agreement.csv")
    pd.DataFrame(corr).to_csv(pc, index=False)
    register([p, pc], "retrieval vs classificação por espaço e correlação entre os rankings de espaços")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {"shape": SERIES[0], "texture": SERIES[1], "learned": SERIES[2], "supervised": SERIES[6]}
    for ax, yk, lab in ((axes[0], "ndcg@10_ssim", "NDCG@10 com ganho = SSIM"), (axes[1], "spearman_ssim", "Spearman(−dist, SSIM)")):
        for _, row in r.iterrows():
            ax.scatter(row.logreg, row[yk], s=34 if row.specialized else 28, marker="s" if row.specialized else "o",
                       color=colors[row.perspetiva], edgecolor="#0b0b0b" if row.specialized else "white",
                       linewidth=0.8, zorder=3)
        show = {"cnn": (4, 3), "lda": (4, -10), "raw": (-26, 5), "hog": (4, 3), "lbp": (4, 3), "sobel": (4, 3),
                "silhouette": (4, 3), "conv_ae_d128": (4, 6), "transformer_ae_d16": (-40, 8),
                "pca_k128_whiten": (4, -3), "pca_k64": (4, -10)}
        # zona densa de cada painel: etiquetas afastadas, com linha de ligação
        show.update({"transformer_ae_d16": (-110, 22), "pca_k64": (-95, -26), "raw": (-60, 16)} if yk == "ndcg@10_ssim"
                    else {"pca_k64": (16, -34), "conv_ae_d128": (28, -12), "raw": (-30, 8)})
        leader = {"transformer_ae_d16", "pca_k64", "raw"} | ({"conv_ae_d128"} if yk != "ndcg@10_ssim" else set())
        for _, row in r[r.espaco.isin(show)].iterrows():
            ax.annotate(row.espaco, (row.logreg, row[yk]), fontsize=7, color="#52514e",
                        xytext=show[row.espaco], textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="#7a7a7a", lw=0.6) if row.espaco in leader else None)
        ax.set_xlabel("accuracy LogReg (val)")
        ax.set_ylabel(lab)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in colors.items()] + \
              [plt.Line2D([], [], marker="s", ls="", color=colors["learned"], markeredgecolor="#0b0b0b",
                          label="learned, especializado (VAE/DAE)")]
    axes[1].legend(handles=handles, loc="lower left", frameon=True, facecolor="#fcfcfb", edgecolor="none")
    fig.suptitle("Melhor para classificar ≠ melhor para semelhança visual?", y=1.02)
    savefig(fig, "retrieval_vs_classification.png", "accuracy de classificação vs concordância com a proxy visual")
    return r


def panels(spaces_show=None, n_queries=6, k=8):
    spaces_show = spaces_show or ["raw", "silhouette", "pca_k64", "sobel", "hog", "lbp", "dense_ae_d16", "conv_ae_d32",
                                  "vae_conv_d32", "dae_conv_d32", "transformer_ae_d64", "lda"]
    spaces_show = [s for s in spaces_show if s in available_spaces()]
    q_idx, _ = query_indices()
    Xte, yte = load_split("test", mmap=True)
    Xtr, ytr = load_split("train", mmap=True)
    rng = np.random.default_rng(CFG["seed"])
    cls_q = [6, 0, 2, 4, 5, 8][:n_queries]  # Shirt, T-shirt, Pullover, Coat, Sandal, Bag
    picks = [int(rng.choice(q_idx[yte[q_idx] == c])) for c in cls_q]
    for qi in picks:
        fig, axes = plt.subplots(len(spaces_show), k + 1, figsize=(1.0 * (k + 1), 1.08 * len(spaces_show)))
        for r_, sp in enumerate(spaces_show):
            res = retrieve(np.asarray(Xte[qi]), sp, k=k)
            ax = axes[r_, 0]
            ax.imshow(np.asarray(Xte[qi]).reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(sp, rotation=0, ha="right", va="center", fontsize=7)
            if r_ == 0:
                ax.set_title(f"query\n{SHORT[yte[qi]]}", fontsize=7)
            for j, gi in enumerate(res["indices"]):
                ax = axes[r_, j + 1]
                ax.imshow(np.asarray(Xtr[gi]).reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ok = ytr[gi] == yte[qi]
                for s_ in ax.spines.values():
                    s_.set_visible(True)
                    s_.set_color("#008300" if ok else "#e34948")
                    s_.set_linewidth(1.6)
                if not ok:
                    ax.set_title(SHORT[ytr[gi]], fontsize=6, color="#e34948", pad=1)
        fig.suptitle(f"Retrieval top-{k} (euclidiana) para a query test #{qi}; moldura vermelha = classe diferente",
                     y=1.0, fontsize=9)
        savefig(fig, f"retrieval_panel_q{qi}.png", f"painel de retrieval (query test {qi}) em vários espaços")


def main():
    ret = evaluate_all()
    compare_with_classification(ret)
    panels()
    log.info("Fase 6 concluída")


if __name__ == "__main__":
    main()
