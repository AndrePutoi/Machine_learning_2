"""Utilitários comuns: configuração, seeds, logging, manifesto de artefactos."""
from __future__ import annotations

import datetime as _dt
import fcntl
import json
import logging
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


CFG = load_config()
OUT = ROOT / CFG["paths"]["outputs"]
DATA_DIR = ROOT / CFG["paths"]["data_dir"]
CLASSES = CFG["data"]["classes"]


def out(*parts: str) -> Path:
    """Caminho dentro de outputs/ (cria a pasta-mãe)."""
    p = OUT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int | None = None) -> int:
    seed = CFG["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


def now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_logger(name: str) -> logging.Logger:
    """Logger que escreve em consola e em outputs/logs/<name>.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(out("logs", f"{name}.log"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


@contextmanager
def _locked(path: Path):
    lock = path.with_suffix(path.suffix + ".lock")
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def register(paths, description: str, config: dict | None = None, kind: str | None = None) -> None:
    """Acrescenta/atualiza entradas em outputs/manifest.json (seguro com processos em paralelo)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    man = OUT / "manifest.json"
    with _locked(man):
        data = json.loads(man.read_text()) if man.exists() else {"artifacts": {}}
        for p in paths:
            rel = str(Path(p).resolve().relative_to(ROOT))
            data["artifacts"][rel] = {
                "descricao": description,
                "tipo": kind or Path(p).suffix.lstrip("."),
                "gerado_em": now(),
                "config": config or {},
            }
        data["atualizado_em"] = now()
        tmp = man.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        tmp.replace(man)


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))
    tmp.replace(path)


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def save_npy(arr, path: Path) -> None:
    """Escrita atómica de .npy (evita ficheiros truncados se o processo morrer)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)


def exists(*paths) -> bool:
    return all(Path(p).exists() for p in paths)


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
