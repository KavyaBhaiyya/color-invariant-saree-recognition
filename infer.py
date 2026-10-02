"""Standalone inference on arbitrary image files.

  python infer.py build     --checkpoint runs/full/best.pt --split test --out gallery_test.pt
  python infer.py build     --checkpoint runs/full/best.pt --folder my_gallery/ --out gallery_my.pt
  python infer.py query     --checkpoint runs/full/best.pt --gallery gallery_test.pt --image q.jpg --topk 5
  python infer.py calibrate --checkpoint runs/full/best.pt --out threshold.json          # uses VAL split only
  python infer.py verify    --checkpoint runs/full/best.pt --a a.jpg --b b.jpg --threshold-file threshold.json
  python infer.py selftest  --checkpoint runs/full/best.pt --n 200                       # end-to-end file test

Preprocessing for any image = EXIF fix -> RGB -> resize short side 256 -> centre crop 224 (same as the gallery
preprocessing used in evaluation).
"""
import argparse
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import roc_curve

from common import (RESIZE, SIZE, Embedder, EvalSet, eval_recolor, extract, hard_neg, load_rgb, norm, seed_all, topk)
from torch.utils.data import DataLoader

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRE = T.Compose([T.Resize(RESIZE), T.CenterCrop(SIZE)])
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_model(ckpt):
    ck = torch.load(ckpt, map_location=DEV)
    m = Embedder(ck["args"]["backbone"], ck["args"]["dim"], pretrained=False).to(DEV)
    m.load_state_dict(ck["model"])
    return m.eval()


@torch.no_grad()
def embed_paths(model, paths, bs=64):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([TF.to_tensor(PRE(load_rgb(p))) for p in paths[i:i + bs]]).to(DEV)
        with torch.autocast(device_type=DEV.type, enabled=DEV.type == "cuda"):
            e = model(norm(x))
        out.append(F.normalize(e.float(), dim=1).cpu())
    return torch.cat(out)


def split_df(manifest, split):
    df = pd.read_csv(manifest)
    return df[(df.status == "ok") & (df.split == split)].reset_index(drop=True)


def cmd_build(a):
    model = load_model(a.checkpoint)
    if a.folder:
        paths = sorted(str(p) for p in Path(a.folder).rglob("*") if p.suffix.lower() in EXT)
    else:
        paths = split_df(a.manifest, a.split).path.tolist()
    emb = embed_paths(model, paths)
    torch.save({"emb": emb, "paths": paths}, a.out)
    print(f"gallery: {len(paths)} images, dim {emb.shape[1]} -> {a.out}")


def cmd_query(a):
    model, gal = load_model(a.checkpoint), torch.load(a.gallery)
    q = embed_paths(model, [a.image])
    s = (q @ gal["emb"].T)[0]
    if a.exclude_self:
        for i, p in enumerate(gal["paths"]):
            if Path(p).resolve() == Path(a.image).resolve():
                s[i] = -2
    v, idx = s.topk(min(a.topk, len(s)))
    for r, (sc, i) in enumerate(zip(v.tolist(), idx.tolist()), 1):
        print(f"{r}. score={sc:.4f}  {gal['paths'][i]}")


def cmd_calibrate(a):
    model = load_model(a.checkpoint)
    va = split_df(a.manifest, "val")
    mk = lambda mode: DataLoader(EvalSet(va, mode, a.seed), batch_size=64, num_workers=2)
    g, hg = extract(model, mk("gallery"), DEV)
    q, hq = extract(model, mk("query"), DEV)
    groups = torch.tensor(va.group.values)
    pos = (q * g).sum(1).numpy()
    neg = (q * g[hard_neg(hq, hg, groups, groups)]).sum(1).numpy()
    y, s = np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]
    fpr, tpr, thr = roc_curve(y, s)
    k = np.nanargmin(np.abs(1 - tpr - fpr))
    k1 = np.where(fpr <= 0.01)[0].max()
    res = dict(threshold_eer=float(thr[k]), threshold_1pct_fpr=float(thr[k1]), n_pos=len(pos), n_neg=len(neg),
               note="calibrated on VAL with synthetic-recolour positives vs colour-matched negatives")
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(res)


def cmd_verify(a):
    model = load_model(a.checkpoint)
    e = embed_paths(model, [a.a, a.b])
    score = float((e[0] * e[1]).sum())
    thr = a.threshold
    if a.threshold_file:
        thr = json.loads(Path(a.threshold_file).read_text())["threshold_eer"]
    msg = f"cosine={score:.4f}"
    if thr is not None:
        msg += f" threshold={thr:.4f} -> {'SAME design' if score >= thr else 'DIFFERENT design'}"
    print(msg)


def cmd_selftest(a):
    """Real file path test: write recoloured query FILES at original resolution, then run the same code a user would."""
    seed_all(a.seed)
    model = load_model(a.checkpoint)
    te = split_df(a.manifest, "test")
    if a.n and a.n < len(te):  # gallery = the sampled subset, so queries and gallery stay one-to-one
        te = te.sample(a.n, random_state=a.seed).reset_index(drop=True)
    tmp = Path(tempfile.mkdtemp(prefix="saree_q_"))
    qpaths = []
    for i, p in enumerate(te.path):
        s = 10_000 * (a.seed + 1) + i
        r = random.Random(s)
        im = load_rgb(p)
        w, h = im.size
        side = int(r.uniform(0.7, 1.0) * min(w, h))
        x0, y0 = r.randint(0, w - side), r.randint(0, h - side)
        x = eval_recolor(TF.to_tensor(im.crop((x0, y0, x0 + side, y0 + side))), s)
        f = tmp / f"q{i}.{'jpg' if a.jpeg else 'png'}"
        TF.to_pil_image(x).save(f, quality=90) if a.jpeg else TF.to_pil_image(x).save(f)
        qpaths.append(str(f))
    g = embed_paths(model, te.path.tolist())
    q = embed_paths(model, qpaths)
    res = dict(n=len(te), query_dir=str(tmp), file_based_identification=topk(q @ g.T))
    print(json.dumps(res, indent=2))
    print("Compare with run_eval numbers only loosely: gallery here is a SUBSET (smaller = easier) and queries are "
          "cropped at full resolution before the 256/224 resize.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    def add(name, fn, **kw):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--manifest", default="work/manifest.csv")
        p.add_argument("--seed", type=int, default=0)
        p.set_defaults(fn=fn)
        return p
    p = add("build", cmd_build); p.add_argument("--out", required=True)
    p.add_argument("--folder"); p.add_argument("--split", default="test")
    p = add("query", cmd_query); p.add_argument("--gallery", required=True); p.add_argument("--image", required=True)
    p.add_argument("--topk", type=int, default=5); p.add_argument("--exclude_self", action="store_true")
    p = add("calibrate", cmd_calibrate); p.add_argument("--out", default="threshold.json")
    p = add("verify", cmd_verify); p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--threshold", type=float); p.add_argument("--threshold-file")
    p = add("selftest", cmd_selftest); p.add_argument("--n", type=int, default=0, help="0 = full test split")
    p.add_argument("--jpeg", action="store_true", help="save queries as JPEG q90 instead of PNG")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
