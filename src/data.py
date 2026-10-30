"""Carregamento do Fashion-MNIST (IDX/ubyte .gz), normalização [0,1] e splits estratificados.

Uso: python -m src.data   (gera outputs/splits/ e outputs/data/; idempotente)
"""
from __future__ import annotations

import gzip
import struct

import numpy as np
from sklearn.model_selection import train_test_split

from .utils import CFG, DATA_DIR, exists, get_logger, out, register, save_npy, set_seed

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _read_idx(path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        zero, dtype, ndim = struct.unpack(">HBB", f.read(4))
        assert zero == 0 and dtype == 0x08, f"formato IDX inesperado em {path}"
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_raw():
    """Devolve (X_train60k uint8 [N,28,28], y_train60k, X_test uint8, y_test)."""
    Xtr = _read_idx(DATA_DIR / FILES["train_images"])
    ytr = _read_idx(DATA_DIR / FILES["train_labels"]).astype(np.int64)
    Xte = _read_idx(DATA_DIR / FILES["test_images"])
    yte = _read_idx(DATA_DIR / FILES["test_labels"]).astype(np.int64)
    return Xtr, ytr, Xte, yte


def _paths(split):
    return (out("data", f"X_{split}_flat.npy"), out("data", f"X_{split}_img.npy"), out("data", f"y_{split}.npy"))


def prepare(force: bool = False) -> None:
    log = get_logger("data")
    set_seed()
    split_files = [out("splits", f"{s}_idx.npy") for s in ("train", "val", "test")]
    data_files = [p for s in ("train", "val", "test") for p in _paths(s)]
    viz_file = out("splits", "viz10k_idx.npy")
    if not force and exists(*split_files, *data_files, viz_file):
        log.info("splits e dados normalizados já existem; nada a fazer")
        return

    Xtr, ytr, Xte, yte = load_raw()
    log.info(f"lidos: train {Xtr.shape}, test {Xte.shape}; classes {np.bincount(ytr).tolist()}")
    idx = np.arange(len(ytr))
    tr_idx, va_idx = train_test_split(idx, test_size=CFG["data"]["val_size"], stratify=ytr,
                                      random_state=CFG["seed"])
    tr_idx, va_idx = np.sort(tr_idx), np.sort(va_idx)
    te_idx = np.arange(len(yte))
    for name, ix in zip(("train", "val", "test"), (tr_idx, va_idx, te_idx)):
        save_npy(ix, out("splits", f"{name}_idx.npy"))

    # Subconjunto estratificado do train (índices relativos ao split train) para visualizações.
    viz_rel, _ = train_test_split(np.arange(len(tr_idx)), train_size=CFG["data"]["viz_subset"],
                                  stratify=ytr[tr_idx], random_state=CFG["seed"])
    save_npy(np.sort(viz_rel), viz_file)

    for name, X, y in (("train", Xtr[tr_idx], ytr[tr_idx]), ("val", Xtr[va_idx], ytr[va_idx]), ("test", Xte, yte)):
        Xf = (X.astype(np.float32) / 255.0)
        pf, pi, py = _paths(name)
        save_npy(Xf.reshape(len(Xf), 784), pf)
        save_npy(Xf.reshape(len(Xf), 1, 28, 28), pi)
        save_npy(y, py)
        log.info(f"{name}: {len(y)} amostras, por classe {np.bincount(y).tolist()}")

    register(split_files, "índices dos splits estratificados (train/val relativos aos 60k de treino; test = 10k)",
             {"seed": CFG["seed"], "val_size": CFG["data"]["val_size"]})
    register(viz_file, "subconjunto estratificado de 10k índices do split train (visualizações)",
             {"seed": CFG["seed"]})
    register(data_files, "imagens normalizadas [0,1] float32 (flatten 784 e 1x28x28) e labels por split")


def load_split(split: str, flat: bool = True, mmap: bool = False):
    """Devolve (X, y) normalizados. flat=True -> [N,784]; False -> [N,1,28,28]."""
    pf, pi, py = _paths(split)
    if not exists(pf, pi, py):
        prepare()
    X = np.load(pf if flat else pi, mmap_mode="r" if mmap else None)
    return X, np.load(py)


def load_all(flat: bool = True):
    return {s: load_split(s, flat) for s in ("train", "val", "test")}


def viz_indices() -> np.ndarray:
    p = out("splits", "viz10k_idx.npy")
    if not p.exists():
        prepare()
    return np.load(p)


def stratified_subset(y: np.ndarray, n: int, seed: int | None = None) -> np.ndarray:
    seed = CFG["seed"] if seed is None else seed
    ix, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return np.sort(ix)


if __name__ == "__main__":
    prepare()
