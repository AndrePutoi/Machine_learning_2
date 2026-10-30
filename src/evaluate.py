"""Métricas: kNN em GPU (torch.cdist em blocos), classificação, qualidade de embeddings 2D,
agrupamento por classe, dimensão intrínseca (TwoNN) e SSIM/IoU vetorizados em GPU."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, confusion_matrix, davies_bouldin_score, f1_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .utils import CFG, device


# ----------------------------------------------------------------------------- kNN em GPU
def _t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


@torch.no_grad()
def pairwise_dist(q: torch.Tensor, g: torch.Tensor, metric: str = "euclidean") -> torch.Tensor:
    if metric == "cosine":
        return 1.0 - F.normalize(q, dim=1) @ F.normalize(g, dim=1).T
    if metric == "euclidean":
        return torch.cdist(q, g)
    raise ValueError(metric)


@torch.no_grad()
def knn_search(Q, G, k: int, metric: str = "euclidean", block: int = 1024, exclude_self: bool = False):
    """k vizinhos mais próximos de cada linha de Q em G. exclude_self=True quando Q é G."""
    dev = device()
    G_t = _t(G, dev)
    D_out, I_out = [], []
    for s in range(0, len(Q), block):
        q = _t(Q[s:s + block], dev)
        d = pairwise_dist(q, G_t, metric)
        if exclude_self:
            r = torch.arange(len(q), device=dev)
            d[r, r + s] = float("inf")
        dk, ik = torch.topk(d, k, dim=1, largest=False)
        D_out.append(dk.cpu())
        I_out.append(ik.cpu())
    return torch.cat(D_out).numpy(), torch.cat(I_out).numpy()


def knn_predict(Xtr, ytr, Xq, k: int = 5, metric: str = "euclidean", exclude_self: bool = False) -> np.ndarray:
    """Voto maioritário uniforme (empates -> classe de menor índice, como no scikit-learn)."""
    _, idx = knn_search(Xq, Xtr, k, metric, exclude_self=exclude_self)
    votes = np.zeros((len(Xq), int(ytr.max()) + 1), dtype=np.int64)
    np.add.at(votes, (np.repeat(np.arange(len(Xq)), k), ytr[idx].ravel()), 1)
    return votes.argmax(1)


def cls_metrics(y_true, y_pred) -> dict:
    return {"accuracy": float((y_true == y_pred).mean()),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}


def confusion(y_true, y_pred, n: int = 10) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n)))


def logreg(Xtr, ytr, C: float | None = None, max_iter: int | None = None):
    c = CFG["classification"]
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=C or c["logreg_C"], max_iter=max_iter or c["logreg_max_iter"],
                                             random_state=CFG["seed"]))
    return model.fit(Xtr, ytr)


# ------------------------------------------------------------ qualidade de embeddings (2D)
@torch.no_grad()
def _knn_sets(X, k):
    return knn_search(X, X, k, exclude_self=True)[1]


@torch.no_grad()
def _rank_penalty(X_rank, nn_other, nn_self, k, block=256) -> float:
    """Soma de (r(i,j) - k) para j nos vizinhos de 'other' que não estão nos vizinhos de 'self',
    com r(i,j) o posto de j na ordenação de distâncias de X_rank a partir de i."""
    dev = device()
    X = _t(X_rank, dev)
    n = len(X)
    nn_other_t = torch.as_tensor(nn_other, device=dev)
    nn_self_t = torch.as_tensor(nn_self, device=dev)
    total = 0.0
    for s in range(0, n, block):
        rows = torch.arange(s, min(s + block, n), device=dev)
        d = torch.cdist(X[rows], X)                       # [b, n]
        d[torch.arange(len(rows), device=dev), rows] = -1.0  # o próprio ponto conta como "mais próximo"
        cand = nn_other_t[rows]                           # [b, k]
        d_c = torch.gather(d, 1, cand)                    # [b, k]
        ranks = (d.unsqueeze(1) < d_c.unsqueeze(2)).sum(2)  # [b, k]  posto (1 = vizinho mais próximo)
        missing = ~(cand.unsqueeze(2) == nn_self_t[rows].unsqueeze(1)).any(2)
        total += ((ranks - k).clamp(min=0) * missing).sum().item()
    return total


def embedding_quality(X_high, X_low, k: int | None = None) -> dict:
    """Trustworthiness, continuity (Venna & Kaski) e preservação de vizinhança kNN."""
    k = k or CFG["viz"]["quality_k"]
    n = len(X_high)
    nn_h, nn_l = _knn_sets(X_high, k), _knn_sets(X_low, k)
    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))
    trust = 1.0 - norm * _rank_penalty(X_high, nn_l, nn_h, k)
    cont = 1.0 - norm * _rank_penalty(X_low, nn_h, nn_l, k)
    overlap = np.mean([len(set(a) & set(b)) / k for a, b in zip(nn_h, nn_l)])
    return {"trustworthiness": float(trust), "continuity": float(cont), "knn_preservation": float(overlap)}


# ------------------------------------------------------------------- agrupamento por classe
def clustering_metrics(Z, y, seed: int | None = None) -> dict:
    seed = CFG["seed"] if seed is None else seed
    Z = np.asarray(Z, dtype=np.float32)
    km = KMeans(n_clusters=10, n_init=10, random_state=seed).fit(Z)
    return {"silhouette": float(silhouette_score(Z, y, random_state=seed)),
            "davies_bouldin": float(davies_bouldin_score(Z, y)),
            "kmeans_nmi": float(normalized_mutual_info_score(y, km.labels_)),
            "kmeans_ari": float(adjusted_rand_score(y, km.labels_))}


# --------------------------------------------------------------------- dimensão intrínseca
def twonn(X, discard: float | None = None) -> dict:
    """Estimador TwoNN (Facco et al., 2017): ajuste linear de -log(1-F(mu)) vs log(mu) pela origem,
    descartando a fração 'discard' de mu mais altos; devolve também a versão de máxima verosimilhança."""
    discard = CFG["twonn"]["discard_fraction"] if discard is None else discard
    d, _ = knn_search(X, X, 2, exclude_self=True)
    r1, r2 = d[:, 0].astype(np.float64), d[:, 1].astype(np.float64)
    ok = r1 > 1e-12                               # remove duplicados exatos
    mu = np.sort(r2[ok] / r1[ok])
    n = len(mu)
    Femp = np.arange(1, n + 1) / n
    keep = int(np.floor(n * (1 - discard)))
    x, yv = np.log(mu[:keep]), -np.log(1 - Femp[:keep])
    d_fit = float((x * yv).sum() / (x * x).sum())
    d_mle = float((n - 1) / np.log(mu).sum())
    return {"id_twonn": d_fit, "id_mle": d_mle, "n_used": int(n), "n_duplicates": int((~ok).sum())}


# ------------------------------------------------------------------- SSIM e IoU em GPU
_K1, _K2, _WIN = 0.01, 0.03, 7


def _box(x):  # média local 7x7 "valid" (equivale ao recorte de bordas do skimage)
    return F.avg_pool2d(x, _WIN, stride=1)


@torch.no_grad()
def ssim_pairs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SSIM entre pares (a[i], b[i]); imagens [N,1,28,28] em [0,1]. Reproduz os valores por omissão
    de skimage.metrics.structural_similarity (janela uniforme 7x7, covariância amostral, data_range=1)."""
    C1, C2 = _K1 ** 2, _K2 ** 2
    cov = _WIN * _WIN / (_WIN * _WIN - 1)
    ua, ub = _box(a), _box(b)
    va = cov * (_box(a * a) - ua * ua)
    vb = cov * (_box(b * b) - ub * ub)
    vab = cov * (_box(a * b) - ua * ub)
    s = ((2 * ua * ub + C1) * (2 * vab + C2)) / ((ua ** 2 + ub ** 2 + C1) * (va + vb + C2))
    return s.mean(dim=(1, 2, 3))


class SSIMGallery:
    """SSIM de uma imagem-query contra toda a galeria (estatísticas da galeria pré-calculadas)."""

    def __init__(self, G_img: np.ndarray):
        dev = device()
        self.G = _t(G_img, dev).reshape(-1, 1, 28, 28)
        self.ug = _box(self.G)
        self.vg = (_WIN * _WIN / (_WIN * _WIN - 1)) * (_box(self.G * self.G) - self.ug ** 2)

    @torch.no_grad()
    def __call__(self, q_img: np.ndarray, block: int = 16384) -> torch.Tensor:
        C1, C2 = _K1 ** 2, _K2 ** 2
        cov = _WIN * _WIN / (_WIN * _WIN - 1)
        q = _t(q_img, self.G.device).reshape(1, 1, 28, 28)
        uq = _box(q)
        vq = cov * (_box(q * q) - uq ** 2)
        out = []
        for s in range(0, len(self.G), block):
            g, ug, vg = self.G[s:s + block], self.ug[s:s + block], self.vg[s:s + block]
            vqg = cov * (_box(q * g) - uq * ug)
            m = ((2 * uq * ug + C1) * (2 * vqg + C2)) / ((uq ** 2 + ug ** 2 + C1) * (vq + vg + C2))
            out.append(m.mean(dim=(1, 2, 3)))
        return torch.cat(out)


@torch.no_grad()
def silhouette_iou(Q_flat, G_flat, thr: float | None = None) -> torch.Tensor:
    """IoU entre silhuetas binarizadas (pixel > thr) de cada query e de toda a galeria. [nq, ng]"""
    thr = CFG["retrieval"]["silhouette_threshold"] if thr is None else thr
    dev = device()
    q = (_t(Q_flat, dev) > thr).float()
    g = (_t(G_flat, dev) > thr).float()
    inter = q @ g.T
    union = q.sum(1, keepdim=True) + g.sum(1).unsqueeze(0) - inter
    return inter / union.clamp(min=1.0)
