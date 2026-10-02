
"""Evaluate REAL same-design / different-colour pairs with a FIXED threshold.

Pairs CSV columns: design_id,path_a,path_b   (one row per design; a and b = two colourways, separate photos)

  python eval_pairs.py --checkpoint runs/full/best.pt --pairs real_pairs.csv --out runs/full/real_pairs

Identification: gallery = every path_a + distractors (images from the manifest split, default test).
                query = every path_b. Correct = the path_a of the same design. Ties count against the model.
Verification:   positives = (a, b) of the same design.
                negatives = (1) cross-design: b_i vs a_j (j != i); (2) colour-matched: for each b_i, the gallery
                image of a different design with the closest colour histogram (a's and distractors).
Threshold is NOT tuned here. Default 0.4526 = EER threshold from `infer.py calibrate` (val, synthetic).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score

from common import color_hist, load_rgb, norm
from infer import DEV, PRE, load_model


@torch.no_grad()
def embed(model, paths, bs=64):
    E, H = [], []
    for i in range(0, len(paths), bs):
        x = torch.stack([TF.to_tensor(PRE(load_rgb(p))) for p in paths[i:i + bs]]).to(DEV)
        H.append(color_hist(x).cpu())
        with torch.autocast(device_type=DEV.type, enabled=DEV.type == "cuda"):
            e = model(norm(x))
        E.append(F.normalize(e.float(), dim=1).cpu())
    return torch.cat(E), torch.cat(H)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--manifest", default="work/manifest.csv")
    ap.add_argument("--split", default="test", help="manifest split used as distractors")
    ap.add_argument("--max_distractors", type=int, default=0, help="0 = all")
    ap.add_argument("--threshold", type=float, default=0.4526)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="real_pairs_results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- load + validate pairs
    pr = pd.read_csv(a.pairs)
    assert {"design_id", "path_a", "path_b"} <= set(pr.columns), "need columns design_id,path_a,path_b"
    assert pr.design_id.is_unique, "design_id must be unique (one row per design)"
    missing = [p for p in list(pr.path_a) + list(pr.path_b) if not Path(p).exists()]
    assert not missing, f"missing files: {missing[:5]}"
    pr["md5_a"], pr["md5_b"] = pr.path_a.map(md5), pr.path_b.map(md5)
    assert (pr.md5_a != pr.md5_b).all(), "some pairs have identical files for a and b"
    n = len(pr)
    warnings = []

    # ---- manifest: leakage check + distractors
    df = pd.read_csv(a.manifest)
    df = df[df.status == "ok"]
    pair_md5 = set(pr.md5_a) | set(pr.md5_b)
    leaked = df[df.split.isin(["train", "val"]) & df.md5.isin(pair_md5)]
    if len(leaked):
        warnings.append(f"{len(leaked)} pair image(s) are byte-identical to TRAIN/VAL images: remove those pairs")
    dis = df[(df.split == a.split) & ~df.md5.isin(pair_md5)]
    if a.max_distractors and len(dis) > a.max_distractors:
        dis = dis.sample(a.max_distractors, random_state=a.seed)
    dpaths = dis.path.tolist()

    # ---- embed
    model = load_model(a.checkpoint)
    A, hA = embed(model, pr.path_a.tolist())
    B, hB = embed(model, pr.path_b.tolist())
    D, hD = embed(model, dpaths) if dpaths else (A[:0], hA[:0])
    G, hG = torch.cat([A, D]), torch.cat([hA, hD])
    ar = torch.arange(n)

    # ---- identification: query b_i, truth = gallery column i
    sim = B @ G.T
    rank = (sim >= sim[ar, ar][:, None]).sum(1) - 1
    res = {"n_pairs": n, "gallery_size": len(G), "n_distractors": len(dpaths), "threshold": a.threshold,
           "identification": {}}
    for k in (1, 5, 10):
        hit = int((rank < k).sum())
        res["identification"][f"top{k}"] = {"value": hit / n, "hits": hit, "n_queries": n, "wilson95": wilson(hit, n)}

    # ---- colour difference of the pairs (are they really different colours?)
    dp = torch.cdist(hB.sqrt(), hA.sqrt())
    pair_d = dp[ar, ar]
    off = dp[~torch.eye(n, dtype=torch.bool)] if n > 1 else torch.tensor([])
    res["colour_distance_hellinger"] = {
        "same_design_pairs_median": float(pair_d.median()),
        "different_design_pairs_median": float(off.median()) if len(off) else None,
        "note": "if same-design distance is not clearly above ~0, the pairs may not differ much in colour"}

    # ---- verification
    pos = (B * A).sum(1).numpy()
    sim_ba = (B @ A.T)
    cross = sim_ba[~torch.eye(n, dtype=torch.bool)].numpy() if n > 1 else np.array([])
    dg = torch.cdist(hB.sqrt(), hG.sqrt())
    dg[ar, ar] = float("inf")  # exclude own design's a-image
    hard = sim[ar, dg.argmin(1)].numpy()
    ver = {"positives": {"n": n, "tpr_at_threshold": float((pos >= a.threshold).mean()),
                         "wilson95": wilson(int((pos >= a.threshold).sum()), n),
                         "mean_score": float(pos.mean())}}
    for name, neg in (("cross_design", cross), ("colour_matched", hard)):
        if len(neg) == 0:
            ver[name] = None
            continue
        y, s = np.r_[np.ones(n), np.zeros(len(neg))], np.r_[pos, neg]
        ver[name] = {"n_neg": int(len(neg)), "roc_auc": float(roc_auc_score(y, s)),
                     "fpr_at_threshold": float((neg >= a.threshold).mean()), "mean_neg_score": float(neg.mean())}
    res["verification"] = ver

    # ---- failures
    top1 = sim.argmax(1)
    paths_g = pr.path_a.tolist() + dpaths
    fail = pd.DataFrame({"design_id": pr.design_id, "query": pr.path_b, "true_rank_0based": rank.numpy(),
                         "top1_path": [paths_g[i] for i in top1.tolist()], "pair_score": pos})
    fail[fail.true_rank_0based > 0].to_csv(out / "failures.csv", index=False)

    res["warnings"] = warnings + (["n < 30: intervals are very wide, treat as a sanity check only"] if n < 30 else [])
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("failed identifications ->", out / "failures.csv")


if __name__ == "__main__":
    main()
