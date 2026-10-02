"""Train the colour-invariant embedder. Selects the best epoch on the VAL split only."""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import Embedder, TrainSet, norm, run_eval, seed_all, supcon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="work/manifest.csv")
    ap.add_argument("--out", default="runs/full")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone lr; head uses 10x")
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--no_color_aug", action="store_true", help="ablation: geometric views only")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    seed_all(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(a.manifest)
    df = df[df.status == "ok"]
    tr, va = df[df.split == "train"].reset_index(drop=True), df[df.split == "val"].reset_index(drop=True)
    print(f"train={len(tr)} val={len(va)} groups(train)={tr.group.nunique()} device={dev}")

    dl = DataLoader(TrainSet(tr, not a.no_color_aug), batch_size=a.bs, shuffle=True, drop_last=True,
                    num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    model = Embedder(a.backbone, a.dim).to(dev)
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": a.lr},
                             {"params": model.head.parameters(), "lr": a.lr * 10}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * 10], total_steps=a.epochs * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    best, hist = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for v1, v2, y in dl:
            x = norm(torch.cat([v1, v2]).to(dev, non_blocking=True))
            y = torch.cat([y, y]).to(dev)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                z = model(x)
            loss = supcon(F.normalize(z.float(), dim=1), y, a.temp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        val = run_eval(model, va, dev, workers=a.workers, seed=a.seed, quick=True)["identification_synthetic_recolor"]
        rec = dict(epoch=ep + 1, loss=tot / len(dl), val_top1=val["top1"], val_top5=val["top5"], sec=time.time() - t0)
        hist.append(rec)
        print(rec)
        if val["top1"] > best:
            best = val["top1"]
            torch.save({"model": model.state_dict(), "args": vars(a)}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(hist, indent=2))


if __name__ == "__main__":
    main()
