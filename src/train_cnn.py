"""CNN supervisionada ponta a ponta: teto de referência para a classificação (Fase 5).

Uso: python -m src.train_cnn   (retoma de last.pt; salta se já concluída)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .evaluate import cls_metrics, confusion
from .models import SupervisedCNN, n_params
from .train_ae import GPUBatchLoader, _atomic_save, load_gpu_data
from .utils import CFG, CLASSES, exists, get_logger, out, register, save_json, save_npy, set_seed

C = CFG["cnn"]


@torch.no_grad()
def predict(model, X, bs=2048):
    model.eval()
    logits, emb = [], []
    for s in range(0, len(X), bs):
        e = model.embed(X[s:s + bs])
        emb.append(e.cpu())
        logits.append(model.head(e).cpu())
    return torch.cat(logits).numpy(), torch.cat(emb).numpy()


def main():
    log = get_logger("cnn")
    run = out("cnn", "x").parent
    mp = run / "metrics.json"
    embs = [run / f"cnn_emb_{s}.npy" for s in ("train", "val", "test")]
    if exists(mp, run / "best.pt", *embs):
        log.info("CNN já treinada; a saltar")
        return
    dev = torch.device("cuda")
    data = load_gpu_data(dev)
    Xtr, ytr = data["train"][0], torch.as_tensor(data["train"][1], device=dev)
    Xva, yva = data["val"]
    set_seed()
    model = SupervisedCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=C["lr"], weight_decay=1e-4)
    gen = torch.Generator(device=dev)
    gen.manual_seed(CFG["seed"])
    st = {"epoch": 0, "best": -1.0, "best_epoch": -1, "bad": 0, "train_time": 0.0, "history": []}
    if (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"].cpu())
        st = ck["state"]
        log.info(f"retomado na época {st['epoch']}")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C["max_epochs"], last_epoch=st["epoch"] - 1) \
        if st["epoch"] else torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C["max_epochs"])
    torch.cuda.reset_peak_memory_stats(dev)
    n = len(Xtr)
    while st["epoch"] < C["max_epochs"]:
        model.train()
        t0 = time.time()
        perm = torch.randperm(n, device=dev, generator=gen)
        tot = 0.0
        for s in range(0, n, C["batch_size"]):
            ix = perm[s:s + C["batch_size"]]
            loss = F.cross_entropy(model(Xtr[ix]), ytr[ix])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach() * len(ix)
        sched.step()
        torch.cuda.synchronize(dev)
        ep_t = time.time() - t0
        st["train_time"] += ep_t
        st["epoch"] += 1
        logits, _ = predict(model, Xva)
        val_loss = float(F.cross_entropy(torch.as_tensor(logits), torch.as_tensor(yva)))
        acc = float((logits.argmax(1) == yva).mean())
        better = acc > st["best"]
        if better:
            st.update(best=acc, best_epoch=st["epoch"], bad=0)
            _atomic_save({"model": model.state_dict(), "epoch": st["epoch"], "val_acc": acc}, run / "best.pt")
        else:
            st["bad"] += 1
        st["history"].append({"epoch": st["epoch"], "train_loss": (tot / n).item(), "val_loss": val_loss,
                              "val_acc": acc, "lr": opt.param_groups[0]["lr"], "epoch_time_s": ep_t})
        pd.DataFrame(st["history"]).to_csv(run / "history.csv", index=False)
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(), "state": st},
                     run / "last.pt")
        log.info(f"CNN ep {st['epoch']:2d} | train loss {(tot / n).item():.4f} | val loss {val_loss:.4f} | "
                 f"val acc {acc:.4f} | {ep_t:.1f}s{' *' if better else ''}")
        if st["bad"] >= C["patience"]:
            log.info(f"early stopping (melhor época {st['best_epoch']})")
            break
    peak = torch.cuda.max_memory_allocated(dev) / 2 ** 20
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev, weights_only=False)["model"])
    res = {"params": n_params(model), "epochs_run": st["epoch"], "best_epoch": st["best_epoch"],
           "train_time_s": st["train_time"], "peak_gpu_mem_mb": peak}
    for s, p in zip(("train", "val", "test"), embs):
        logits, emb = predict(model, data[s][0])
        save_npy(emb.astype(np.float32), p)
        if s != "train":
            pred = logits.argmax(1)
            m = cls_metrics(data[s][1], pred)
            res[f"{s}_accuracy"], res[f"{s}_macro_f1"] = m["accuracy"], m["macro_f1"]
            pd.DataFrame(confusion(data[s][1], pred), index=CLASSES, columns=CLASSES).to_csv(
                out("metrics", f"cnn_confusion_{s}.csv"))
    save_json(res, mp)
    (run / "last.pt").unlink(missing_ok=True)
    register([run / "best.pt"], "pesos da CNN supervisionada (melhor accuracy de validação)", C, "pt")
    register([run / "history.csv", mp], "histórico e métricas (accuracy/macro-F1 val e test) da CNN supervisionada")
    register(embs, "embeddings de 128 dimensões (penúltima camada) da CNN supervisionada")
    register([out("metrics", f"cnn_confusion_{s}.csv") for s in ("val", "test")], "matrizes de confusão da CNN")
    log.info(f"CNN concluída: {res}")


if __name__ == "__main__":
    main()
