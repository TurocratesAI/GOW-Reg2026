#!/usr/bin/env python3
"""
DEPRECATED / not on the shipped path. The organ head is trained jointly with the answer heads in
gow/heads/train_heads.py (which uses the shared data_split and class balancing); this standalone router is
kept only as a reference/smoke script and is not imported by the walker, eval, or container. Do not rely on
its glob-order split or its unused class-balance weights. Prefer train_heads.py.

Organ router (node-1 head) - the highest-leverage head: a wrong organ craters EdgeF1 0.69->0.43.

Gated-ABMIL over a frozen Virchow2 bag [N,2560] -> slide vector -> 7-way organ.
The pooled slide vector is also the substrate for the Mahalanobis OOD gate + FAISS retrieval.

  train:  python gow/heads/organ_router.py --features-dir data/feats --cot <train_CoT.json> --device cuda:0
  smoke:  python gow/heads/organ_router.py --smoke        # synthetic bags; validates the loop end-to-end
"""
import argparse, json, os, glob, sys
import numpy as np

ORGANS = ["prostate", "breast", "colon", "stomach", "bladder", "lung", "cervix"]
O2I = {o: i for i, o in enumerate(ORGANS)}


def build_model(in_dim=2560, hid=512, n_organ=7, dropout=0.25):
    import torch.nn as nn
    import torch

    class GatedABMIL(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(dropout))
            self.V = nn.Linear(hid, hid); self.U = nn.Linear(hid, hid); self.w = nn.Linear(hid, 1)

        def forward(self, H):                                   # H: [N, in_dim]
            h = self.fc(H)
            a = torch.softmax(self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))), dim=0)  # [N,1]
            return (a * h).sum(0), a.squeeze(-1)                # slide vec [hid], attention [N]

    class OrganRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.abmil = GatedABMIL()
            self.head = nn.Linear(hid, n_organ)

        def forward(self, H):
            z, a = self.abmil(H)
            return self.head(z), z, a

    return OrganRouter()


# ------------------------------------------------------------------ data
def bag_label_pairs(features_dir, cot_path):
    """Match each embedded bag (npz stem == slide_id) to its organ label from the CoT."""
    labels = {}
    for c in json.load(open(cot_path)):
        sid = os.path.splitext(c["id"])[0] if "id" in c else None
        if sid and c.get("organ") in O2I:
            labels[sid] = O2I[c["organ"]]
    pairs = []
    for p in glob.glob(os.path.join(features_dir, "*.npz")):
        sid = os.path.splitext(os.path.basename(p))[0]
        if sid in labels:
            pairs.append((p, labels[sid]))
    return pairs


def load_bag(npz_path):
    return np.load(npz_path)["X"].astype(np.float32)          # [N,2560]


def run(pairs, load_fn, args):
    import torch
    from torch.utils.data import Dataset, DataLoader
    dev = args.device
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()
    # class-balanced sampling weights
    counts = np.bincount([y for _, y in pairs], minlength=7) + 1
    w = 1.0 / counts
    n_tr = int(0.9 * len(pairs))
    tr, va = pairs[:n_tr], pairs[n_tr:]

    def epoch(split, train):
        model.train(train)
        tot, correct, lsum = 0, 0, 0.0
        for item, y in split:
            H = torch.from_numpy(load_fn(item)).to(dev)
            logits, _, _ = model(H)
            loss = ce(logits.unsqueeze(0), torch.tensor([y], device=dev))
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            lsum += loss.item(); tot += 1
            correct += int(logits.argmax().item() == y)
        return lsum / max(tot, 1), correct / max(tot, 1)

    for ep in range(args.epochs):
        tl, ta = epoch(tr, True)
        vl, vacc = epoch(va, False)
        print(f"epoch {ep}: train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {vacc:.3f}")
    if args.out and not args.smoke:
        torch.save(model.state_dict(), args.out)
        print(f"[saved] {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir"); ap.add_argument("--cot")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", default="gow/artifacts/organ_router.pt")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        rng = np.random.default_rng(0)
        # synthetic: each organ has a distinct mean-shift so a good model should separate them
        centers = rng.normal(0, 1, (7, 2560)).astype(np.float32)
        bags = []
        for i in range(280):
            y = i % 7
            n = int(rng.integers(120, 500))
            X = (centers[y] + rng.normal(0, 3, (n, 2560))).astype(np.float32)
            bags.append((X, y))
        run([(X, y) for X, y in bags], lambda X: X, args)
        print("\n[smoke] architecture + train loop OK (val acc should climb well above 1/7=0.14).")
        return

    assert args.features_dir and args.cot, "need --features-dir and --cot (or --smoke)"
    pairs = bag_label_pairs(args.features_dir, args.cot)
    print(f"matched {len(pairs)} embedded bags to organ labels")
    run(pairs, load_bag, args)


if __name__ == "__main__":
    main()
