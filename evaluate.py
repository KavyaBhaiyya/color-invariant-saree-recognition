"""Evaluate on the TEST split + measure efficiency. Writes results.json and results.md.

  python evaluate.py --checkpoint runs/full/best.pt --out runs/full
  python evaluate.py --checkpoint imagenet --backbone resnet50 --out runs/imagenet_baseline   # no training
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from common import SIZE, Embedder, run_eval, seed_all


def efficiency(model, dev):
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(torch.randn(1, 3, SIZE, SIZE, device=dev))
    res = dict(params_million=params / 1e6, embedding_dim=model.out_dim, input=f"{SIZE}x{SIZE}",
               flops_giga_per_image=fc.get_total_flops() / 1e9, flops_note="torch FlopCounterMode, multiply-add = 2 FLOPs",
               latency_device=str(dev), latency_precision="fp32")
    for bs in (1, 32):
        x = torch.randn(bs, 3, SIZE, SIZE, device=dev)
        with torch.no_grad():
            for _ in range(10):
                model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            for _ in range(50):
                model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
        res[f"latency_ms_batch{bs}"] = (time.time() - t) / 50 * 1000
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="work/manifest.csv")
    ap.add_argument("--checkpoint", required=True, help="path to best.pt, or 'imagenet' for the untrained baseline")
    ap.add_argument("--backbone", default="resnet50", help="only used with --checkpoint imagenet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.checkpoint == "imagenet":
        model = Embedder(a.backbone, dim=0).to(dev)
    else:
        ck = torch.load(a.checkpoint, map_location=dev)
        model = Embedder(ck["args"]["backbone"], ck["args"]["dim"], pretrained=False).to(dev)
        model.load_state_dict(ck["model"])

    df = pd.read_csv(a.manifest)
    test = df[(df.status == "ok") & (df.split == "test")].reset_index(drop=True)
    res = run_eval(model, test, dev, workers=a.workers, seed=a.seed)
    res["efficiency"] = efficiency(model, dev)
    res["checkpoint"] = a.checkpoint
    (out / "results.json").write_text(json.dumps(res, indent=2))

    idn = res["identification_synthetic_recolor"]
    lines = [f"Test gallery size: {res['n_gallery']}", "",
             "| Identification (synthetic recolour query vs original gallery) | Top-1 | Top-5 | Top-10 |", "|---|---|---|---|",
             f"| all test images | {idn['top1']:.4f} | {idn['top5']:.4f} | {idn['top10']:.4f} |"]
    for k in ("natural_identification_any_colour", "natural_identification_different_colour"):
        r = res[k]
        lines.append(f"| {k} (n_queries={r['n_queries']}) | {r['top1']:.4f} | {r['top5']:.4f} | {r['top10']:.4f} |" if r
                     else f"| {k} | n/a (no such pairs) | | |")
    lines += ["", "| Verification | ROC-AUC | EER | TPR@1%FPR | n_pos | n_neg |", "|---|---|---|---|---|---|"]
    for k, v in res["verification"].items():
        lines.append(f"| {k} | {v['roc_auc']:.4f} | {v['eer']:.4f} | {v['tpr_at_1pct_fpr']:.4f} | {v['n_pos']} | {v['n_neg']} |"
                     if v else f"| {k} | n/a | | | | |")
    e = res["efficiency"]
    lines += ["", f"Params: {e['params_million']:.2f} M | embedding dim: {e['embedding_dim']} | FLOPs: {e['flops_giga_per_image']:.2f} G/image"
              f" | latency ({e['latency_device']}, fp32): {e['latency_ms_batch1']:.1f} ms (bs1), {e['latency_ms_batch32']:.1f} ms (bs32)"]
    (out / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
