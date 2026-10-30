"""Treino dos autoencoders (script único parametrizado).

  python -m src.train_ae --arch conv_ae --dim 32     # um modelo (retoma de last.pt; salta se já concluído)
  python -m src.train_ae --grid --jobs 2 --with-cnn  # grelha completa 3x6 (+ CNN) em subprocessos
  python -m src.train_ae --smoke                     # 1 época por arquitetura com d=2 e d=128
  python -m src.train_ae --probe                     # sondagem de batch size / memória / tempo
  python -m src.train_ae --collect                   # agrega métricas -> outputs/metrics/ae_summary.csv

Comparação justa: mesma loss (MSE), otimizador, LR, warmup, nº máximo de épocas, early stopping
(patience fixa no MSE de validação), ReduceLROnPlateau e seed; só a arquitetura muda.

Adenda (especializados, mesmo protocolo; só o objetivo de treino muda):
  python -m src.train_ae --arch vae_conv --dim 32    # Conv-VAE: MSE somado + beta*KL (warm-up do KL)
  python -m src.train_ae --arch dae_conv --dim 32    # Conv-DAE: input com ruído gaussiano, alvo limpo
  python -m src.train_ae --grid --specialized        # os 12 especializados (também --smoke --specialized)
(também funciona como "python src/train_ae.py ...")
"""
from __future__ import annotations

if __package__ in (None, ""):   # execução direta do ficheiro: torna os imports relativos válidos
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import load_split
from .evaluate import ssim_pairs
from .models import build_ae, n_params
from .utils import CFG, ROOT, exists, get_logger, load_json, out, register, save_json, save_npy, set_seed

ACFG = CFG["ae"]
SPEC = list(ACFG["specialized"])
ALL_ARCHS = list(ACFG["archs"]) + SPEC
VAE, DAE = ACFG["vae"], ACFG["dae"]


def name_of(arch: str, d: int) -> str:
    return f"{arch}_d{d}"


class GPUBatchLoader:
    """Iterador de mini-batches sobre tensores já residentes na GPU (substitui o DataLoader:
    o dataset cabe inteiro na memória da GPU, por isso não há cópias host->device nem workers)."""

    def __init__(self, X: torch.Tensor, batch_size: int, shuffle: bool = True, seed: int = 0):
        self.X, self.bs, self.shuffle = X, batch_size, shuffle
        self.gen = torch.Generator(device=X.device)
        self.gen.manual_seed(seed)

    def __len__(self):
        return (len(self.X) + self.bs - 1) // self.bs

    def __iter__(self):
        n = len(self.X)
        idx = torch.randperm(n, device=self.X.device, generator=self.gen) if self.shuffle else torch.arange(n, device=self.X.device)
        for s in range(0, n, self.bs):
            yield self.X[idx[s:s + self.bs]]

    def state(self):
        return self.gen.get_state()

    def load_state(self, st):
        self.gen.set_state(st)


def load_gpu_data(dev):
    data = {}
    for s in ("train", "val", "test"):
        X, y = load_split(s, flat=False)
        data[s] = (torch.from_numpy(np.ascontiguousarray(X)).to(dev), y)
    return data


def add_noise(x: torch.Tensor, sigma: float, gen: torch.Generator) -> torch.Tensor:
    """x + sigma*N(0,1), cortado a [0,1] (input do DAE e queries ruidosas da experiência de robustez)."""
    return (x + sigma * torch.randn(x.shape, device=x.device, generator=gen)).clamp_(0, 1)


def kl_div(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, exp(logvar)) || N(0, I)) por imagem (somado nas d dimensões)."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)


@torch.no_grad()
def evaluate_mse(model, X: torch.Tensor, bs: int = 2048, amp: bool = False, X_in: torch.Tensor | None = None) -> float:
    """MSE por pixel entre model(X_in) e X (X_in = X por defeito; no DAE, X_in é a versão ruidosa)."""
    model.eval()
    X_in = X if X_in is None else X_in
    tot = 0.0
    for s in range(0, len(X), bs):
        x = X[s:s + bs]
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            r = model(X_in[s:s + bs])
        tot += F.mse_loss(r.float(), x, reduction="sum").item()
    return tot / X.numel()


def train_loss(model, arch: str, x: torch.Tensor, beta_t: float, noise_gen) -> torch.Tensor:
    """Base: MSE por pixel. VAE: MSE somado por imagem + beta_t*KL (média no batch).
    DAE: MSE por pixel entre a reconstrução do input ruidoso e a imagem limpa."""
    if arch == "vae_conv":
        r, mu, logvar = model.forward_train(x)
        rec = F.mse_loss(r.float(), x, reduction="none").sum((1, 2, 3)).mean()
        return rec + beta_t * kl_div(mu.float(), logvar.float()).mean()
    if arch == "dae_conv":
        return F.mse_loss(model(add_noise(x, DAE["sigma"], noise_gen)).float(), x)
    return F.mse_loss(model(x).float(), x)


@torch.no_grad()
def vae_val(model, X: torch.Tensor, bs: int = 2048) -> dict:
    """Objetivo do VAE com beta=1 (reconstrução a partir de mu), MSE por pixel e KL médio por imagem."""
    model.eval()
    rec = kl = 0.0
    for s in range(0, len(X), bs):
        x = X[s:s + bs]
        mu, logvar = model.encode_stats(x)
        rec += F.mse_loss(model.decode(mu).float(), x, reduction="sum").item()
        kl += kl_div(mu, logvar).sum().item()
    n = len(X)
    return {"val_loss": (rec + kl) / n, "val_mse": rec / X.numel(), "val_kl": kl / n}


def validate(model, arch: str, Xval: torch.Tensor, Xval_noisy, amp: bool) -> dict:
    """val_loss = critério de early stopping (o objetivo de cada modelo); val_mse = MSE com input limpo."""
    if arch == "vae_conv":
        return vae_val(model, Xval)
    mse = evaluate_mse(model, Xval, amp=amp)
    if arch == "dae_conv":
        return {"val_loss": evaluate_mse(model, Xval, amp=amp, X_in=Xval_noisy), "val_mse": mse}
    return {"val_loss": mse, "val_mse": mse}


@torch.no_grad()
def encode_all(model, X: torch.Tensor, bs: int = 2048) -> np.ndarray:
    model.eval()
    return torch.cat([model.encode(X[s:s + bs]).float().cpu() for s in range(0, len(X), bs)]).numpy()


@torch.no_grad()
def recon_metrics(model, X: torch.Tensor, bs: int = 2048) -> dict:
    model.eval()
    mse, ssim = [], []
    for s in range(0, len(X), bs):
        x = X[s:s + bs]
        r = model(x).float()
        mse.append(((r - x) ** 2).mean(dim=(1, 2, 3)).cpu())
        ssim.append(ssim_pairs(x, r).cpu())
    return {"mse": float(torch.cat(mse).mean()), "ssim": float(torch.cat(ssim).mean())}


def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def fit(arch: str, d: int, data: dict, run_dir: Path | None, log, max_epochs: int | None = None,
        batch_size: int | None = None, lr: float | None = None, amp: bool | None = None, resume: bool = True):
    """Treina um AE. Se run_dir for dado, guarda best.pt/last.pt/history.csv e retoma de last.pt."""
    dev = data["train"][0].device
    max_epochs = max_epochs or ACFG["max_epochs"]
    bs = batch_size or ACFG["batch_size"]
    base_lr = lr or ACFG["lr"]
    amp = ACFG["amp"] if amp is None else amp
    set_seed()
    model = build_ae(arch, d, CFG).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=ACFG["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loader = GPUBatchLoader(data["train"][0], bs, shuffle=True, seed=CFG["seed"])
    noise_gen, Xval_noisy = None, None
    if arch == "dae_conv":
        noise_gen = torch.Generator(device=dev)
        noise_gen.manual_seed(CFG["seed"])
        gv = torch.Generator(device=dev)
        gv.manual_seed(CFG["seed"] + 1)
        Xval_noisy = add_noise(data["val"][0], DAE["sigma"], gv)   # ruído fixo no val
    kl_warm = VAE["kl_warmup_epochs"] * len(loader)
    st = {"epoch": 0, "step": 0, "best": float("inf"), "best_epoch": -1, "bad": 0, "plateau_bad": 0,
          "lr_scale": 1.0, "train_time": 0.0, "history": []}

    last_p = run_dir / "last.pt" if run_dir else None
    if run_dir and resume and last_p.exists():
        ck = torch.load(last_p, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        loader.load_state(ck["loader"].cpu())
        if noise_gen is not None and "noise_gen" in ck:
            noise_gen.set_state(ck["noise_gen"].cpu())
        if "cuda_rng" in ck:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), dev)
        st = ck["state"]
        log.info(f"retomado de last.pt na época {st['epoch']}")

    torch.cuda.reset_peak_memory_stats(dev)
    Xval = data["val"][0]
    warm = ACFG["warmup_steps"]
    while st["epoch"] < max_epochs:
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        for x in loader:
            st["step"] += 1
            cur_lr = base_lr * st["lr_scale"] * min(1.0, st["step"] / warm)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            beta_t = VAE["beta"] * min(1.0, st["step"] / kl_warm) if arch == "vae_conv" else 0.0
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = train_loss(model, arch, x, beta_t, noise_gen)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.detach()
            nb += 1
        torch.cuda.synchronize(dev)
        train_l = (tot / nb).item()
        v = validate(model, arch, Xval, Xval_noisy, amp)
        ep_time = time.time() - t0
        st["train_time"] += ep_time
        st["epoch"] += 1
        # no warm-up do KL o objetivo ainda está a mudar: patience e plateau só contam depois
        frozen = arch == "vae_conv" and st["epoch"] <= VAE["kl_warmup_epochs"]
        improved = v["val_loss"] < st["best"] - ACFG["min_delta"]
        if improved:
            st.update(best=v["val_loss"], best_epoch=st["epoch"], bad=0, plateau_bad=0)
            if run_dir:
                _atomic_save({"model": model.state_dict(), "arch": arch, "d": d, "epoch": st["epoch"],
                              "val_loss": v["val_loss"], "val_mse": v["val_mse"]}, run_dir / "best.pt")
        elif not frozen:
            st["bad"] += 1
            st["plateau_bad"] += 1
            if st["plateau_bad"] >= ACFG["plateau_patience"]:
                st["lr_scale"] *= ACFG["plateau_factor"]
                st["plateau_bad"] = 0
        if arch in SPEC:
            h = {"epoch": st["epoch"], "train_loss": train_l, "val_loss": v["val_loss"], "val_mse": v["val_mse"]}
            if arch == "vae_conv":
                h.update(val_kl=v["val_kl"], beta=beta_t)
            h.update(lr=cur_lr, epoch_time_s=ep_time, best_so_far=st["best"])
            msg = f"train {train_l:.5f} | val {v['val_loss']:.5f} | val_mse {v['val_mse']:.5f}" + \
                  (f" | KL {v['val_kl']:.2f} | beta {beta_t:.2f}" if arch == "vae_conv" else "")
        else:
            h = {"epoch": st["epoch"], "train_mse": train_l, "val_mse": v["val_mse"], "lr": cur_lr,
                 "epoch_time_s": ep_time, "best_so_far": st["best"]}
            msg = f"train {train_l:.5f} | val {v['val_mse']:.5f}"
        st["history"].append(h)
        log.info(f"{name_of(arch, d)} ep {st['epoch']:3d} | {msg} | lr {cur_lr:.2e} | {ep_time:.1f}s{' *' if improved else ''}")
        if run_dir:
            pd.DataFrame(st["history"]).to_csv(run_dir / "history.csv", index=False)
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                  "loader": loader.state(), "state": st, "cuda_rng": torch.cuda.get_rng_state(dev)}
            if noise_gen is not None:
                ck["noise_gen"] = noise_gen.get_state()
            _atomic_save(ck, last_p)
        if st["bad"] >= ACFG["patience"]:
            log.info(f"early stopping na época {st['epoch']} (melhor época {st['best_epoch']})")
            break
    st["peak_mem_mb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 20
    return model, st


def train_one(arch: str, d: int) -> None:
    name = name_of(arch, d)
    run_dir = out("ae", name, "x").parent
    metrics_p = run_dir / "metrics.json"
    lat_paths = [out("latents", f"{name}_{s}.npy") for s in ("train", "val", "test")]
    lv_paths = [out("latents", f"{name}_logvar_{s}.npy") for s in ("train", "val", "test")] if arch == "vae_conv" else []
    log = get_logger(f"ae_{name}")
    if exists(metrics_p, *lat_paths, *lv_paths, run_dir / "best.pt"):
        log.info(f"{name}: já concluído, a saltar")
        return
    dev = torch.device("cuda")
    data = load_gpu_data(dev)
    log.info(f"{name}: início (bs={ACFG['batch_size']}, lr={ACFG['lr']}, max_epochs={ACFG['max_epochs']}, "
             f"patience={ACFG['patience']}, amp={ACFG['amp']})")
    model, st = fit(arch, d, data, run_dir, log)

    ck = torch.load(run_dir / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    for s, p in zip(("train", "val", "test"), lat_paths):
        save_npy(encode_all(model, data[s][0]).astype(np.float32), p)
    rv, rt = recon_metrics(model, data["val"][0]), recon_metrics(model, data["test"][0])
    metrics = {"arch": arch, "d": d, "params": n_params(model), "epochs_run": st["epoch"],
               "best_epoch": st["best_epoch"], "val_mse": rv["mse"], "val_ssim": rv["ssim"],
               "test_mse": rt["mse"], "test_ssim": rt["ssim"], "train_time_s": st["train_time"],
               "time_per_epoch_s": st["train_time"] / max(st["epoch"], 1), "peak_gpu_mem_mb": st["peak_mem_mb"],
               "batch_size": ACFG["batch_size"], "lr": ACFG["lr"], "amp": ACFG["amp"]}
    if arch == "vae_conv":
        # mu já está em lat_paths (encode() devolve mu); logvar, KL e unidades ativas à parte
        for s, p in zip(("train", "val", "test"), lv_paths):
            save_npy(encode_logvar(model, data[s][0]), p)
        for s in ("val", "test"):
            ex = vae_stats(model, data[s][0])
            metrics.update({f"{s}_kl": ex["kl"], f"{s}_active_units": ex["active_units"]})
            if s == "val":
                metrics["mu_var_val"] = ex["mu_var"]
        metrics.update(beta=VAE["beta"], kl_warmup_epochs=VAE["kl_warmup_epochs"],
                       active_unit_threshold=VAE["active_unit_threshold"])
    if arch == "dae_conv":
        g = torch.Generator(device=dev)
        for s in ("val", "test"):
            g.manual_seed(CFG["seed"] + 1)
            X = data[s][0]
            metrics[f"{s}_denoise_mse"] = evaluate_mse(model, X, X_in=add_noise(X, DAE["sigma"], g))
        metrics["sigma_train"] = DAE["sigma"]
    if arch in SPEC:
        metrics["best_val_loss"] = st["best"]
    save_json(metrics, metrics_p)
    (run_dir / "last.pt").unlink(missing_ok=True)   # só servia para retomar
    cfg_used = {k: ACFG[k] for k in ("optimizer", "lr", "batch_size", "max_epochs", "patience", "warmup_steps")}
    register([run_dir / "best.pt"], f"pesos do melhor checkpoint (val MSE) do {name}", cfg_used, "pt")
    register([run_dir / "history.csv"], f"histórico de treino do {name}", cfg_used, "csv")
    register([metrics_p], f"métricas de reconstrução (MSE/SSIM val e test), parâmetros, tempo e memória do {name}")
    register(lat_paths, f"latentes train/val/test do {name} (float32)" + (" (= mu do VAE)" if lv_paths else ""), {"d": d})
    if lv_paths:
        register(lv_paths, f"logvar train/val/test do {name} (float32)", {"d": d})
    log.info(f"{name}: concluído | val MSE {rv['mse']:.5f} SSIM {rv['ssim']:.4f} | test MSE {rt['mse']:.5f} "
             f"SSIM {rt['ssim']:.4f} | {st['train_time']:.0f}s | pico {st['peak_mem_mb']:.0f} MB")


@torch.no_grad()
def encode_logvar(model, X: torch.Tensor, bs: int = 2048) -> np.ndarray:
    model.eval()
    return torch.cat([model.encode_stats(X[s:s + bs])[1].float().cpu() for s in range(0, len(X), bs)]).numpy()


@torch.no_grad()
def vae_stats(model, X: torch.Tensor, bs: int = 2048) -> dict:
    """KL médio por imagem e unidades ativas (Var_x[mu_j] > limiar, Burda et al. 2016)."""
    model.eval()
    mus, kl = [], 0.0
    for s in range(0, len(X), bs):
        mu, logvar = model.encode_stats(X[s:s + bs])
        mus.append(mu.float().cpu())
        kl += kl_div(mu, logvar).sum().item()
    var = torch.cat(mus).var(0)
    return {"kl": kl / len(X), "active_units": int((var > VAE["active_unit_threshold"]).sum()),
            "mu_var": [round(float(v), 6) for v in var]}


def collect() -> pd.DataFrame:
    rows = []
    for arch in ALL_ARCHS:
        for d in ACFG["dims"]:
            p = out("ae", name_of(arch, d), "metrics.json")
            if p.exists():
                m = load_json(p)
                m.pop("mu_var_val", None)
                rows.append({**m, "specialized": arch in SPEC})
    df = pd.DataFrame(rows)
    if len(df):
        p = out("metrics", "ae_summary.csv")
        df.to_csv(p, index=False)
        register(p, "resumo dos autoencoders: 18 base + especializados (VAE/DAE, coluna specialized); MSE/SSIM, "
                 "parâmetros, tempo, memória; no VAE também KL e unidades ativas, no DAE o MSE de denoising")
    return df


# --------------------------------------------------------------------------- grelha
def grid(jobs: int, with_cnn: bool, specialized: bool = False) -> None:
    log = get_logger("ae_grid_specialized" if specialized else "ae_grid")
    queue = []
    # intercala arquiteturas para que modelos pesados (transformer) corram ao lado de leves
    order = SPEC if specialized else [a for a in ("transformer_ae", "conv_ae", "dense_ae") if a in ACFG["archs"]]
    for d in ACFG["dims"]:
        for arch in order:
            queue.append(["--arch", arch, "--dim", str(d)])
    if with_cnn:
        queue.insert(0, ["--cnn"])
    running = []
    t0 = time.time()
    log.info(f"grelha: {len(queue)} trabalhos, {jobs} em paralelo")
    while queue or running:
        while queue and len(running) < jobs:
            args = queue.pop(0)
            tag = "cnn" if args == ["--cnn"] else f"ae_{args[1]}_d{args[3]}"
            mod = "src.train_cnn" if args == ["--cnn"] else "src.train_ae"
            cmd = [sys.executable, "-m", mod] + ([] if args == ["--cnn"] else args)
            err = open(out("logs", f"{tag}.err"), "w")
            running.append((tag, subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=err), err))
            log.info(f"lançado {tag}")
        time.sleep(5)
        for item in list(running):
            tag, p, err = item
            if p.poll() is not None:
                err.close()
                log.info(f"terminado {tag} (código {p.returncode}) | decorrido {(time.time() - t0) / 60:.1f} min")
                running.remove(item)
    collect()
    log.info(f"grelha concluída em {(time.time() - t0) / 60:.1f} min")


# --------------------------------------------------------------------------- smoke test
def smoke(specialized: bool = False) -> None:
    log = get_logger("ae_smoke")
    dev = torch.device("cuda")
    data = load_gpu_data(dev)
    rows = []
    for arch in (SPEC if specialized else ACFG["archs"]):
        for d in (2, 128):
            model = build_ae(arch, d, CFG).to(dev)
            x = data["val"][0][:8]
            z = model.encode(x)
            assert z.shape == (8, d), z.shape
            assert model.decode(z).shape == x.shape
            if arch == "vae_conv":
                r, mu, lv = model.forward_train(x)
                assert r.shape == x.shape and mu.shape == lv.shape == (8, d)
            model, st = fit(arch, d, data, None, log, max_epochs=1)
            h = st["history"][-1]
            rows.append({"arch": arch, "d": d, "params": n_params(model), "train_loss": h.get("train_loss", h.get("train_mse")),
                         "val_mse": h["val_mse"], "epoch_time_s": h["epoch_time_s"], "peak_mem_mb": st["peak_mem_mb"]})
            log.info(f"smoke {arch} d={d}: {rows[-1]}")
            del model
            torch.cuda.empty_cache()
    p = out("metrics", "smoke_test_specialized.csv" if specialized else "smoke_test.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    register(p, "smoke test: 1 época por arquitetura com d=2 e d=128" + (" (especializados)" if specialized else ""))


# --------------------------------------------------------------------------- sondagem GPU
def probe(margin_gb: float = 1.5) -> None:
    """Aumenta o batch size até OOM (ou até ao tamanho do train) e regista memória e tempo/época.
    A GPU é partilhada: limita-se a fração de memória deste processo para nunca comprometer os outros."""
    log = get_logger("gpu_probe")
    dev = torch.device("cuda")
    free, total = torch.cuda.mem_get_info(dev)
    cap = max(free - margin_gb * 2 ** 30, 1 * 2 ** 30)
    torch.cuda.set_per_process_memory_fraction(cap / total, 0)
    log.info(f"GPU {torch.cuda.get_device_name(dev)}: total {total / 2**30:.2f} GB, livre {free / 2**30:.2f} GB, "
             f"limite para este processo {cap / 2**30:.2f} GB")
    data = load_gpu_data(dev)
    X = data["train"][0]
    n_train = len(X)
    rows = []
    for arch in ACFG["archs"]:
        bs = 128
        while bs <= 32768:
            for amp in (False, True):
                if amp and bs not in (256, 1024):
                    continue
                set_seed()
                model = build_ae(arch, 128, CFG).to(dev)
                opt = torch.optim.AdamW(model.parameters(), lr=ACFG["lr"])
                scaler = torch.amp.GradScaler("cuda", enabled=amp)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
                row = {"arch": arch, "d": 128, "batch_size": bs, "amp_fp16": amp}
                try:
                    steps, warm = 25, 5
                    for i in range(steps + warm):
                        if i == warm:
                            torch.cuda.synchronize(dev)
                            t0 = time.time()
                        x = X[torch.randint(0, n_train, (bs,), device=dev)]
                        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                            loss = F.mse_loss(model(x).float(), x)
                        opt.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    torch.cuda.synchronize(dev)
                    per_step = (time.time() - t0) / steps
                    row.update(status="ok", peak_mem_mb=torch.cuda.max_memory_allocated(dev) / 2 ** 20,
                               step_time_ms=per_step * 1000, epoch_time_s=per_step * np.ceil(n_train / bs),
                               frac_of_cap=torch.cuda.max_memory_allocated(dev) / cap)
                except torch.OutOfMemoryError:
                    row.update(status="OOM", peak_mem_mb=np.nan, step_time_ms=np.nan, epoch_time_s=np.nan,
                               frac_of_cap=np.nan)
                log.info(str(row))
                rows.append(row)
                del model, opt
                torch.cuda.empty_cache()
            if rows[-1]["status"] == "OOM" or bs >= n_train:
                break
            bs *= 2
    df = pd.DataFrame(rows)
    df["gpu"] = torch.cuda.get_device_name(dev)
    df["cap_gb"] = cap / 2 ** 30
    p = out("metrics", "gpu_probe.csv")
    df.to_csv(p, index=False)
    register(p, "sondagem de batch size: pico de memória e tempo/época por arquitetura (d=128)")

    # convergência vs batch size: 5 épocas de cada arquitetura com d=32 (mesmo LR)
    crow = []
    for arch in ACFG["archs"]:
        for b in (128, 256, 512, 1024, 2048):
            _, st = fit(arch, 32, data, None, log, max_epochs=5, batch_size=b)
            crow.append({"arch": arch, "d": 32, "batch_size": b, "val_mse_5ep": st["history"][-1]["val_mse"],
                         "time_5ep_s": st["train_time"], "peak_mem_mb": st["peak_mem_mb"]})
            log.info(str(crow[-1]))
    p2 = out("metrics", "batch_convergence.csv")
    pd.DataFrame(crow).to_csv(p2, index=False)
    register(p2, "convergência (val MSE após 5 épocas) em função do batch size, d=32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ALL_ARCHS)
    ap.add_argument("--dim", type=int)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--jobs", type=int, default=ACFG["parallel_jobs"])
    ap.add_argument("--with-cnn", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--specialized", action="store_true", help="com --grid/--smoke: só os modelos especializados")
    a = ap.parse_args()
    if a.smoke:
        smoke(a.specialized)
    elif a.probe:
        probe()
    elif a.grid:
        grid(a.jobs, a.with_cnn and not a.specialized, a.specialized)
    elif a.collect:
        print(collect().to_string())
    elif a.arch and a.dim:
        train_one(a.arch, a.dim)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
