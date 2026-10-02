"""Shared code: colour transforms, datasets, model, evaluation protocol."""
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

SIZE, RESIZE = 224, 256
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LUM = (0.299, 0.587, 0.114)
LUMT = torch.tensor(LUM).view(3, 1, 1)
_YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
_YIQ_INV = torch.linalg.inv(_YIQ)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_rgb(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def norm(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


# ---------------------------------------------------------------- colour transforms (x: 3xHxW in [0,1])
def hue_rotate(x, theta):
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=x.dtype)
    return torch.einsum("ij,jhw->ihw", _YIQ_INV @ R @ _YIQ, x).clamp(0, 1)


def train_recolor(x):
    """Training colour family: channel permutation, hue rotation, per-channel gamma/gain, random grayscale."""
    x = x[random.sample([0, 1, 2], 3)]
    x = hue_rotate(x, random.uniform(0, 2 * math.pi))
    gam = torch.tensor([math.exp(random.uniform(math.log(0.5), math.log(2.0))) for _ in range(3)]).view(3, 1, 1)
    gain = torch.tensor([random.uniform(0.6, 1.2) for _ in range(3)]).view(3, 1, 1)
    x = (x.clamp(1e-4, 1) ** gam * gain).clamp(0, 1)
    if random.random() < 0.2:
        x = (LUMT * x).sum(0, keepdim=True).expand(3, -1, -1)
    return x


def eval_recolor(x, seed):
    """Held-out colour family used ONLY for evaluation: gradient map (luminance -> random 3-colour ramp)."""
    r = random.Random(seed)
    lum = lambda c: sum(w * v for w, v in zip(LUM, c))
    for _ in range(50):
        cols = sorted(([r.random() for _ in range(3)] for _ in range(3)), key=lum)
        if lum(cols[2]) - lum(cols[0]) > 0.4:
            break
    c0, c1, c2 = [torch.tensor(c).view(3, 1, 1) for c in cols]
    l = (LUMT * x).sum(0, keepdim=True)
    return torch.where(l < 0.5, c0 + (c1 - c0) * 2 * l, c1 + (c2 - c1) * (2 * l - 1)).clamp(0, 1)


def color_hist(x):
    """64-bin RGB histogram (4x4x4) of a batch in [0,1]; used only to pick colour-matched negatives."""
    q = (F.adaptive_avg_pool2d(x, 32) * 3.999).long().clamp(0, 3)
    idx = q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2]
    return F.one_hot(idx.flatten(1), 64).float().mean(1)


# ---------------------------------------------------------------- datasets
class TrainSet(Dataset):
    """Returns two independently cropped (and, if color_aug, independently recoloured) views + group label."""

    def __init__(self, df, color_aug=True):
        self.paths, self.labels, self.color_aug = df.path.tolist(), df.group.tolist(), color_aug
        self.geo = T.Compose([T.Resize(RESIZE), T.RandomResizedCrop(SIZE, scale=(0.4, 1.0)),
                              T.RandomHorizontalFlip(), T.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = load_rgb(self.paths[i])
        views = []
        for _ in range(2):
            x = self.geo(im)
            views.append(train_recolor(x) if self.color_aug else x)
        return views[0], views[1], self.labels[i]


class EvalSet(Dataset):
    """gallery: centre crop of original. query: seeded random crop (70-100% of short side) + held-out recolour."""

    def __init__(self, df, mode, seed=0):
        self.paths, self.mode, self.seed = df.path.tolist(), mode, seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = T.Resize(RESIZE)(load_rgb(self.paths[i]))
        if self.mode == "gallery":
            return TF.to_tensor(T.CenterCrop(SIZE)(im))
        s = 10_000 * (self.seed + 1) + i
        r = random.Random(s)
        w, h = im.size
        side = int(r.uniform(0.7, 1.0) * min(w, h))
        x0, y0 = r.randint(0, w - side), r.randint(0, h - side)
        im = im.crop((x0, y0, x0 + side, y0 + side)).resize((SIZE, SIZE), Image.BICUBIC)
        return eval_recolor(TF.to_tensor(im), s)


# ---------------------------------------------------------------- model
class GeM(nn.Module):
    def __init__(self, p=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))

    def forward(self, x):
        x = x.float().clamp(min=1e-6)
        return F.avg_pool2d(x.pow(self.p), x.shape[-2:]).pow(1.0 / self.p)


class Embedder(nn.Module):
    """ImageNet-pretrained torchvision ResNet + GeM pooling + linear head. dim=0 -> no head (raw backbone)."""

    def __init__(self, backbone="resnet50", dim=256, pretrained=True):
        super().__init__()
        m = getattr(torchvision.models, backbone)(weights="IMAGENET1K_V1" if pretrained else None)
        feat = m.fc.in_features
        m.avgpool, m.fc = GeM(), nn.Identity()
        self.backbone = m
        self.head = nn.Linear(feat, dim) if dim > 0 else nn.Identity()
        self.out_dim = dim if dim > 0 else feat

    def forward(self, x):
        return self.head(self.backbone(x))


def supcon(z, labels, t=0.1):
    """Supervised contrastive loss. z: (N,d) L2-normalised float; positives = same label (incl. other view)."""
    n = z.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = (z @ z.T / t).masked_fill(eye, float("-inf"))
    pos = (labels[:, None] == labels[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    return (-(logp.masked_fill(~pos, 0).sum(1) / pos.sum(1).clamp(min=1))).mean()


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    E, H = [], []
    for x in loader:
        x = x.to(device)
        H.append(color_hist(x).cpu())
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            e = model(norm(x))
        E.append(F.normalize(e.float(), dim=1).cpu())
    return torch.cat(E), torch.cat(H)


def topk(sim, ks=(1, 5, 10)):
    """Row i's correct gallery item is i. Ties count against the model (pessimistic)."""
    rank = (sim >= sim.diag()[:, None]).sum(1) - 1
    return {f"top{k}": (rank < k).float().mean().item() for k in ks}


def hard_neg(hq, hg, gq, gg):
    """For each query, index of the colour-closest gallery image of a DIFFERENT group."""
    d = torch.cdist(hq.sqrt(), hg.sqrt()).masked_fill(gq[:, None] == gg[None, :], float("inf"))
    return d.argmin(1)


def random_neg(groups, seed):
    n, rng = len(groups), np.random.RandomState(seed)
    j = (np.arange(n) + rng.randint(1, n, n)) % n
    for _ in range(100):
        bad = (groups[j] == groups).numpy()
        if not bad.any():
            break
        j[bad] = rng.randint(0, n, bad.sum())
    return torch.tensor(j)


def verif(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    y, s = np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]
    fpr, tpr, _ = roc_curve(y, s)
    k = np.nanargmin(np.abs(1 - tpr - fpr))
    return dict(roc_auc=float(roc_auc_score(y, s)), eer=float((fpr[k] + 1 - tpr[k]) / 2),
                tpr_at_1pct_fpr=float(tpr[fpr <= 0.01].max()), n_pos=int(len(pos)), n_neg=int(len(neg)))


def natural_id(sim, groups, hd, thr, ks=(1, 5, 10)):
    """Query = each test image with a same-group partner; gallery = all other test images."""
    n = len(groups)
    eye = torch.eye(n, dtype=torch.bool)
    pos = (groups[:, None] == groups[None, :]) & ~eye
    ignore = torch.zeros_like(pos)
    if thr is not None:  # only count partners whose colours differ; drop same-colour partners from the gallery
        far = hd >= thr
        ignore, pos = pos & ~far, pos & far
    keep = pos.any(1)
    if keep.sum() == 0:
        return None
    s = sim.clone().masked_fill(eye | ignore, float("-inf"))
    best = s.masked_fill(~pos, float("-inf")).max(1).values
    rank = (s.masked_fill(pos, float("-inf")) >= best[:, None]).sum(1)
    out = {f"top{k}": (rank[keep] < k).float().mean().item() for k in ks}
    out["n_queries"] = int(keep.sum())
    return out


def run_eval(model, df, device, bs=64, workers=4, seed=0, quick=False):
    mk = lambda mode: DataLoader(EvalSet(df, mode, seed), batch_size=bs, num_workers=workers)
    g, hg = extract(model, mk("gallery"), device)
    q, hq = extract(model, mk("query"), device)
    sim = q @ g.T
    out = {"n_gallery": len(df), "identification_synthetic_recolor": topk(sim)}
    if quick:
        return out
    groups, n = torch.tensor(df.group.values), len(df)

    pos = (q * g).sum(1).numpy()
    neg_hard = (q * g[hard_neg(hq, hg, groups, groups)]).sum(1).numpy()
    neg_rand = (q * g[random_neg(groups, seed)]).sum(1).numpy()
    out["verification"] = {
        "synthetic_recolor_pos_vs_colour_matched_neg": verif(pos, neg_hard),
        "synthetic_recolor_pos_vs_random_neg": verif(pos, neg_rand),
    }

    # natural (real-image) same-group pairs, if the data contains any
    hd = torch.cdist(hg.sqrt(), hg.sqrt())
    iu = torch.triu_indices(n, n, 1)
    alld = hd[iu[0], iu[1]].numpy()
    thr = float(np.quantile(alld[np.random.RandomState(seed).choice(len(alld), min(len(alld), 200000), replace=False)], 0.75))
    gg = g @ g.T
    out["natural_identification_any_colour"] = natural_id(gg, groups, hd, None)
    out["natural_identification_different_colour"] = natural_id(gg, groups, hd, thr)
    out["different_colour_threshold_hellinger_p75"] = thr
    same = groups[iu[0]] == groups[iu[1]]
    a, b = iu[0][same], iu[1][same]
    hn_g = hard_neg(hg, hg, groups, groups)
    pos_n, neg_n = gg[a, b].numpy(), gg[a, hn_g[a]].numpy()
    far = (hd[a, b] >= thr).numpy()
    out["verification"]["natural_same_group_vs_colour_matched_neg"] = verif(pos_n, neg_n)
    out["verification"]["natural_same_group_different_colour_vs_colour_matched_neg"] = verif(pos_n[far], neg_n[far])
    return out
