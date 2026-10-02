"""Inspect, clean, group and split the saree datasets. No assumptions about folder structure.

Usage:
  python prepare_data.py --root deeplure=/path/a --root patterns=/path/b --out work

Outputs (in --out): manifest.csv (every file found, with status), report.json, group_check.png
"""
import argparse, hashlib, json, zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def extract_zips(root: Path):
    for z in sorted(root.rglob("*.zip")):
        dest = root / "_extracted" / z.stem
        if not dest.exists():
            print("extracting", z)
            with zipfile.ZipFile(z) as f:
                f.extractall(dest)


def scan_one(p):
    try:
        with Image.open(p) as im:
            im.verify()
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    except Exception as e:  # corrupt / unreadable
        return dict(path=str(p), status="corrupt", note=str(e)[:80])
    gray = im.convert("L")
    g = np.asarray(gray.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    dhash = "".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten())  # 64-bit, colour-blind
    gstd = float(np.asarray(gray.resize((64, 64)), dtype=np.float32).std())
    md5 = hashlib.md5(Path(p).read_bytes()).hexdigest()
    return dict(path=str(p), w=im.size[0], h=im.size[1], md5=md5, dhash=dhash, gstd=gstd, status="ok")


def group_near_duplicates(bits, ok_mask, thr):
    """Union-find over images whose grayscale dHash differ in <= thr bits. Low-texture images stay alone."""
    n = len(bits)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    idx = np.where(ok_mask)[0]
    B = bits[idx].astype(np.float32)
    for s in range(0, len(idx), 1000):
        Bc = B[s:s + 1000]
        ham = 64 - (Bc @ B.T + (1 - Bc) @ (1 - B).T)
        for a, b in np.argwhere(ham <= thr):
            a_, b_ = idx[s + a], idx[b]
            if a_ < b_:
                ra, rb = find(a_), find(b_)
                if ra != rb:
                    parent[ra] = rb
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([remap[r] for r in roots])


def contact_sheet(df, out, n_groups=8, per=6, th=128):
    multi = df[df.status == "ok"].groupby("group").filter(lambda g: len(g) >= 2)
    if multi.empty:
        return False
    sizes = multi.groupby("group").size().sort_values(ascending=False).head(n_groups)
    sheet = Image.new("RGB", (per * th, len(sizes) * th), "white")
    for r, gid in enumerate(sizes.index):
        for c, p in enumerate(multi[multi.group == gid].path.head(per)):
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB").resize((th, th))
            sheet.paste(im, (c * th, r * th))
    sheet.save(out)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="name=path (repeatable)")
    ap.add_argument("--out", default="work")
    ap.add_argument("--min_side", type=int, default=128, help="drop images with a shorter side below this")
    ap.add_argument("--hamming", type=int, default=4, help="max dHash bit difference to call near-duplicates")
    ap.add_argument("--gstd_min", type=float, default=8.0, help="below this grayscale std = plain image, not grouped")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in a.root:
        name, path = spec.split("=", 1)
        root = Path(path)
        extract_zips(root)
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXT)
        print(f"[{name}] {len(files)} image files found under {root}")
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(scan_one, files, chunksize=32))
        for p, r in zip(files, res):
            r["source"] = name
            r["folder_label"] = str(p.parent.relative_to(root))  # recorded only; NOT assumed to be a design id
            rows.append(r)
    df = pd.DataFrame(rows)
    for c, d in [("w", 0), ("h", 0), ("md5", ""), ("dhash", ""), ("gstd", 0.0), ("note", "")]:
        if c not in df:
            df[c] = d
    df = df.reset_index(drop=True)

    # ---- cleaning (each rule recorded in `status`, nothing is silently deleted)
    ok = df.status == "ok"
    df.loc[ok & (df[["w", "h"]].min(axis=1) < a.min_side), "status"] = "too_small"
    dup = (df.status == "ok") & df.duplicated("md5", keep="first")
    df.loc[dup, "status"] = "exact_duplicate"

    # ---- near-duplicate grouping (colour-blind) -> `group`; groups never cross train/val/test
    okm = (df.status == "ok").values
    bits = np.array([[c == "1" for c in h] if isinstance(h, str) and h else [False] * 64 for h in df.dhash], dtype=np.uint8)
    groupable = okm & (df.gstd.values >= a.gstd_min)
    df["group"] = group_near_duplicates(bits, groupable, a.hamming)

    # ---- group-aware split
    okdf = df[df.status == "ok"]
    gsize = okdf.groupby("group").size()
    gids = gsize.index.values.copy()
    np.random.RandomState(a.seed).shuffle(gids)
    total = gsize.sum()
    cum, split_of = 0, {}
    for g in gids:
        frac = cum / total
        split_of[g] = "test" if frac < a.test else ("val" if frac < a.test + a.val else "train")
        cum += gsize[g]
    df["split"] = df.group.map(split_of).where(df.status == "ok", "none")

    df.to_csv(out / "manifest.csv", index=False)

    rep = dict(
        counts_by_status=df.status.value_counts().to_dict(),
        counts_by_source_status=df.groupby(["source", "status"]).size().unstack(fill_value=0).to_dict("index"),
        folder_labels_top30=Counter(okdf.folder_label).most_common(30),
        n_folder_labels=int(okdf.folder_label.nunique()),
        image_size_min=int(okdf[["w", "h"]].min().min()) if len(okdf) else None,
        image_size_median_wh=[float(okdf.w.median()), float(okdf.h.median())] if len(okdf) else None,
        n_groups=int(len(gsize)),
        n_multi_image_groups=int((gsize >= 2).sum()),
        largest_group_sizes=sorted(gsize.tolist(), reverse=True)[:10],
        low_texture_images=int((okdf.gstd < a.gstd_min).sum()),
        split_counts_by_source=okdf.assign(split=df.split).groupby(["source", "split"]).size().unstack(fill_value=0).to_dict("index"),
        params=vars(a),
    )
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    made = contact_sheet(df, out / "group_check.png")
    print(json.dumps(rep, indent=2, default=str))
    print("contact sheet written:" if made else "no multi-image groups found, no contact sheet", out / "group_check.png")
    print("CHECK group_check.png by eye: rows should be the same design. If rows mix designs, lower --hamming.")


if __name__ == "__main__":
    main()
