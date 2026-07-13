#!/usr/bin/env python3
"""
GOW model: frozen Virchow2 bag -> organ router + question-conditioned answer/grade heads.

One shared mechanism answers EVERY node (binary presence, procedure, 46-way histologic type,
grades, diagnoses) AND stays open-vocab: a CLIP-style scorer projects the question-conditioned
evidence into CONCH text space and scores it against the candidate-answer CONCH embeddings for
that (organ,question). Restricting candidates to the node's allowed set = organ-masking for free;
adding CONCH-embedded rare/gyn/mesenchymal answers (e.g. leiomyoma) = open-vocab OOD naming.
Grades are just nodes whose candidates are the grade strings -> trains on the official CoT labels.

  smoke:  python gow/heads/gow_model.py --smoke --device cuda:1
"""
import argparse
import numpy as np


def build(in_dim=2560, q_dim=512, hid=512, n_organ=7, dropout=0.25):
    import torch, torch.nn as nn

    class GatedABMIL(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(dropout))
            self.V = nn.Linear(hid, hid); self.U = nn.Linear(hid, hid); self.w = nn.Linear(hid, 1)

        def forward(self, H):
            h = self.fc(H)
            a = torch.softmax(self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))), 0)
            return (a * h).sum(0), a.squeeze(-1)

    class QCPooler(nn.Module):
        """Question-conditioned attention (HistoSelect PatchSelector style) + focal max-pool."""
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(q_dim, in_dim)
            self.score = nn.Sequential(nn.Linear(in_dim * 2, hid), nn.GELU(), nn.Linear(hid, 1))
            self.out = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU())

        def forward(self, H, q_emb):                     # H:[N,in_dim]  q_emb:[q_dim]
            q = self.q_proj(q_emb).expand_as(H)
            s = self.score(torch.cat([H, q], -1)).squeeze(-1)          # [N] relevance
            a = torch.softmax(s, 0)
            v_soft = (a.unsqueeze(-1) * H).sum(0)                       # prevalence/context
            v_max = H[s.argmax()]                                      # focal (one tile flips a 'yes')
            return self.out(v_soft + v_max), a

    class GOWModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.abmil = GatedABMIL()
            self.organ_head = nn.Linear(hid, n_organ)
            self.pooler = QCPooler()
            self.ans_proj = nn.Sequential(nn.Linear(hid + hid + n_organ, hid), nn.GELU(),
                                          nn.Linear(hid, q_dim))
            self.logit_scale = nn.Parameter(torch.tensor(2.3))         # ~exp=10, CLIP-style temperature

        def organ(self, H):
            z, _ = self.abmil(H)
            return self.organ_head(z), z

        def answer(self, H, z, organ_oh, q_emb, cand_emb):
            """cand_emb:[C,q_dim] CONCH text embeddings of this node's allowed answers -> logits[C]."""
            v, attn = self.pooler(H, q_emb)
            p = self.ans_proj(torch.cat([v, z, organ_oh], -1))
            p = p / p.norm().clamp_min(1e-6)
            c = cand_emb / cand_emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            return self.logit_scale.exp() * (p @ c.t()), attn

    return GOWModel()


def smoke(device):
    import torch
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = build().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ce = torch.nn.CrossEntropyLoss()

    # synthetic world: 7 organs (mean-shifted bags) + one binary node whose answer = sign of bag-mean[0]
    organ_centers = torch.tensor(rng.normal(0, 1, (7, 2560)), dtype=torch.float32)
    q_emb = torch.tensor(rng.normal(0, 1, 512), dtype=torch.float32, device=device)     # the question
    cand = torch.tensor(rng.normal(0, 1, (2, 512)), dtype=torch.float32, device=device)  # 2 answers
    for step in range(400):
        y_o = int(rng.integers(0, 7)); n = int(rng.integers(120, 400))
        H = (organ_centers[y_o] + torch.randn(n, 2560) * 3).to(device)
        y_a = int(H.mean(0)[0].item() > 0)                       # learnable image->answer signal
        o_logits, z = model.organ(H)
        oh = torch.zeros(7, device=device); oh[y_o] = 1
        a_logits, _ = model.answer(H, z, oh, q_emb, cand)
        loss = ce(o_logits.unsqueeze(0), torch.tensor([y_o], device=device)) + \
               ce(a_logits.unsqueeze(0), torch.tensor([y_a], device=device))
        opt.zero_grad(); loss.backward(); opt.step()

    # eval
    oc = ac = 0
    for _ in range(200):
        y_o = int(rng.integers(0, 7)); n = int(rng.integers(120, 400))
        H = (organ_centers[y_o] + torch.randn(n, 2560) * 3).to(device)
        y_a = int(H.mean(0)[0].item() > 0)
        with torch.no_grad():
            o_logits, z = model.organ(H)
            oh = torch.zeros(7, device=device); oh[y_o] = 1
            a_logits, _ = model.answer(H, z, oh, q_emb, cand)
        oc += int(o_logits.argmax().item() == y_o); ac += int(a_logits.argmax().item() == y_a)
    print(f"[smoke] organ acc {oc/200:.3f}  |  answer acc {ac/200:.3f}  "
          f"(both should be well above chance: organ>0.14, answer>0.5)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.smoke:
        smoke(args.device)
    else:
        print("GOWModel defined. Use --smoke to validate, or import build() in the trainer.")


if __name__ == "__main__":
    main()
