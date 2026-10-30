"""Fase 8: mineração automática de casos de falha em vários espaços.

Dois tipos de caso por espaço:
  (a) vizinho mais próximo com classe diferente: para cada imagem do test, o seu 1-NN no train;
      ficam os pares com classes diferentes e menor distância (uma imagem de test por caso);
  (b) mesma classe, distância máxima: para cada classe, o par de imagens do test mais afastado.
A causa provável é atribuída por regras simples (sem olhar para o modelo):
  - classes de tronco trocadas entre si (T-shirt/Pullover/Coat/Shirt) -> "ambiguidade entre classes de tronco"
  - Sandal/Sneaker/Ankle boot trocados -> "calçado de forma semelhante"
  - outras trocas com silhueta muito parecida (IoU >= 0.75) -> "silhueta semelhante entre classes"
  - mesma classe com brilho médio muito diferente (> 2x) -> "diferença de brilho/contraste"
  - mesma classe com silhueta pouco sobreposta (IoU < 0.5) -> "forma/estilo diferente dentro da classe"
  - resto -> "padrão/textura diferente" ou "outra troca entre classes"

Adenda: `vae_conv_d32` e `dae_conv_d32` entram na mineração e são comparados com `conv_ae_d32`
(taxa de erro 1-NN, pares de classes trocados em todos os erros 1-NN do test e sobreposição de Jaccard
entre os conjuntos de imagens falhadas de cada espaço).

Uso: python -m src.failures
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .data import load_split
from .evaluate import knn_search, pairwise_dist
from .plots import SEQ, SERIES, SHORT, heatmap, plt, savefig
from .retrieval import SPACES
from .utils import CFG, device, get_logger, out, register

F = CFG["failures"]
TOPS, SHOES = {0, 2, 4, 6}, {5, 7, 9}
log = get_logger("failures")


def iou(a, b, thr=0.1):
    A_, B_ = a > thr, b > thr
    return float((A_ & B_).sum() / max((A_ | B_).sum(), 1))


def cause(kind, ya, yb, xa, xb):
    if kind == "vizinho_classe_diferente":
        if ya in TOPS and yb in TOPS:
            return "ambiguidade entre classes de tronco"
        if ya in SHOES and yb in SHOES:
            return "calçado de forma semelhante"
        return "silhueta semelhante entre classes" if iou(xa, xb) >= 0.75 else "outra troca entre classes"
    ma, mb = xa[xa > 0.1].mean(), xb[xb > 0.1].mean()
    if max(ma, mb) / max(min(ma, mb), 1e-6) > 2.0:
        return "diferença de brilho/contraste"
    return "forma/estilo diferente dentro da classe" if iou(xa, xb) < 0.5 else "padrão/textura diferente"


@torch.no_grad()
def mine(space: str, Xtr, ytr, Xte, yte):
    sp = SPACES[space]
    Gtr = np.ascontiguousarray(sp.features("train"), dtype=np.float32)
    Gte = np.ascontiguousarray(sp.features("test"), dtype=np.float32)
    rows = []
    # (a) 1-NN com classe diferente, menor distância
    d, i = knn_search(Gte, Gtr, 1)
    d, i = d[:, 0], i[:, 0]
    wrong = np.where(ytr[i] != yte)[0]
    used = set()
    for q in wrong[np.argsort(d[wrong])]:
        if i[q] in used:
            continue
        used.add(i[q])
        rows.append({"espaco": space, "tipo": "vizinho_classe_diferente", "split_a": "test", "idx_a": int(q),
                     "split_b": "train", "idx_b": int(i[q]), "classe_a": int(yte[q]), "classe_b": int(ytr[i[q]]),
                     "distancia": float(d[q]), "taxa_erro_1nn": float(len(wrong) / len(yte))})
        if len(rows) >= F["n_cases"]:
            break
    # (b) mesma classe, par mais afastado (um por classe)
    dev = device()
    for c in range(10):
        ix = np.where(yte == c)[0]
        Z = torch.as_tensor(Gte[ix], device=dev)
        D = pairwise_dist(Z, Z, "euclidean")
        a, b = np.unravel_index(int(torch.argmax(D)), D.shape)
        rows.append({"espaco": space, "tipo": "mesma_classe_distante", "split_a": "test", "idx_a": int(ix[a]),
                     "split_b": "test", "idx_b": int(ix[b]), "classe_a": c, "classe_b": c,
                     "distancia": float(D[a, b]), "taxa_erro_1nn": float(len(wrong) / len(yte))})
    for r in rows:
        xa = Xte[r["idx_a"]]
        xb = (Xtr if r["split_b"] == "train" else Xte)[r["idx_b"]]
        r["causa"] = cause(r["tipo"], r["classe_a"], r["classe_b"], xa, xb)
        r["iou_silhueta"] = iou(xa, xb)
        r["nome_a"], r["nome_b"] = SHORT[r["classe_a"]], SHORT[r["classe_b"]]
    return rows


def figure_by_cause(df, Xtr, Xte):
    for c, g in df.groupby("causa"):
        g = g.head(16)
        n = len(g)
        cols = 4
        rws = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rws, cols * 2, figsize=(cols * 2 * 1.15, rws * 1.45))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for k, (_, r) in enumerate(g.iterrows()):
            ra, ca = divmod(k, cols)
            xa = Xte[r.idx_a]
            xb = (Xtr if r.split_b == "train" else Xte)[r.idx_b]
            for j, (x, nm) in enumerate(((xa, r.nome_a), (xb, r.nome_b))):
                ax = axes[ra, ca * 2 + j]
                ax.imshow(x.reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
                ax.set_title(nm, fontsize=7, color="#e34948" if r.classe_a != r.classe_b else "#0b0b0b", pad=2)
            axes[ra, ca * 2].text(0, 31, f"{r.espaco} d={r.distancia:.2f}", fontsize=5.5, color="#52514e", va="top")
        fig.suptitle(f"Causa provável: {c} ({len(df[df.causa == c])} casos)", y=1.02, fontsize=10)
        fig.tight_layout()
        safe = c.replace("/", "_").replace(" ", "_")
        savefig(fig, f"failures_{safe}.png", f"casos de falha agrupados pela causa provável: {c}")


def compare_spaces(ytr, yte, ref="conv_ae_d32", others=("vae_conv_d32", "dae_conv_d32")):
    """Todos os erros 1-NN (test -> train) por espaço: pares de classes trocados e sobreposição (Jaccard)
    dos conjuntos de imagens falhadas."""
    spaces = [s for s in F["spaces"] if all(p.exists() for p in SPACES[s].paths)]
    wrong, pair_rows = {}, []
    for space in spaces:
        sp = SPACES[space]
        _, i = knn_search(np.ascontiguousarray(sp.features("test"), dtype=np.float32),
                          np.ascontiguousarray(sp.features("train"), dtype=np.float32), 1)
        pred = ytr[i[:, 0]]
        w = np.where(pred != yte)[0]
        wrong[space] = set(w.tolist())
        pairs = pd.Series([" ↔ ".join(sorted((SHORT[yte[q]], SHORT[pred[q]]))) for q in w]).value_counts()
        for pair, n in pairs.items():
            pair_rows.append({"espaco": space, "par": pair, "n_erros": int(n), "frac_dos_erros": n / len(w),
                              "taxa_erro_1nn": len(w) / len(yte), "specialized": space.split("_d")[0] in CFG["ae"]["specialized"]})
    pairs = pd.DataFrame(pair_rows)
    J = pd.DataFrame([[len(wrong[a] & wrong[b]) / len(wrong[a] | wrong[b]) for b in spaces] for a in spaces],
                     index=spaces, columns=spaces)
    pp, pj = out("failures", "failure_pairs_by_space.csv"), out("failures", "failure_overlap_jaccard.csv")
    pairs.to_csv(pp, index=False)
    J.to_csv(pj)
    register([pp, pj], "todos os erros 1-NN do test por espaço: contagem por par de classes trocado e sobreposição "
             "(Jaccard) dos conjuntos de imagens falhadas entre espaços")

    sel = [s for s in (ref,) + tuple(others) if s in spaces]
    top = pairs[pairs.espaco.isin(sel)].groupby("par").n_erros.sum().sort_values(ascending=False).index[:10]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    w = 0.8 / len(sel)
    for k, sp in enumerate(sel):
        v = pairs[pairs.espaco == sp].set_index("par").n_erros.reindex(top).fillna(0)
        axes[0].barh(np.arange(len(top))[::-1] + (len(sel) / 2 - k - 0.5) * w, v.values, height=w,
                     color=[SERIES[2], SERIES[6], "#8c5a2b"][k], label=f"{sp} (erro {100 * len(wrong[sp]) / len(yte):.1f}%)")
    axes[0].set_yticks(np.arange(len(top))[::-1], top, fontsize=8)
    axes[0].set_xlabel("nº de erros 1-NN no test")
    axes[0].set_title("Pares de classes trocados (10 mais frequentes)")
    axes[0].legend(fontsize=8, loc="lower right")
    heatmap(axes[1], J.values, list(J.columns), list(J.index), fmt="{:.2f}", cmap=SEQ, vmin=0, vmax=1, fontsize=7)
    axes[1].set_title("Sobreposição (Jaccard) das imagens falhadas")
    fig.tight_layout()
    savefig(fig, "failures_specialized_comparison.png",
            "pares de classes trocados (ConvAE vs VAE vs DAE, d=32) e Jaccard das imagens falhadas entre espaços")
    return pairs, J


def main():
    Xtr, ytr = load_split("train")
    Xte, yte = load_split("test")
    rows = []
    for space in F["spaces"]:
        if not all(p.exists() for p in SPACES[space].paths):
            log.info(f"{space}: em falta, a saltar")
            continue
        r = mine(space, Xtr, ytr, Xte, yte)
        rows += r
        log.info(f"{space}: {len(r)} casos; erro 1-NN test {r[0]['taxa_erro_1nn']:.3f}")
    df = pd.DataFrame(rows)
    df["specialized"] = df.espaco.str.split("_d").str[0].isin(CFG["ae"]["specialized"])
    p = out("failures", "failure_cases.csv")
    df.to_csv(p, index=False)
    summ = df.groupby(["tipo", "causa"]).size().rename("n").reset_index()
    ps = out("failures", "failure_causes_summary.csv")
    summ.to_csv(ps, index=False)
    # imagens que falham em vários espaços ao mesmo tempo (robustas à representação)
    multi = df[df.tipo == "vizinho_classe_diferente"].groupby("idx_a").espaco.nunique()
    pm = out("failures", "failures_shared_across_spaces.csv")
    multi[multi > 1].rename("n_espacos").reset_index().to_csv(pm, index=False)
    register([p, ps, pm], "casos de falha minados automaticamente (índices, classes, distâncias, causa provável), "
             "resumo por causa e casos partilhados entre espaços")
    figure_by_cause(df, Xtr, Xte)
    log.info(f"resumo de causas:\n{summ.to_string(index=False)}")
    compare_spaces(ytr, yte)


if __name__ == "__main__":
    main()
