# REG2026 (MICCAI 2026): Grounded Ontology Walker

Submission for the REG2026 challenge (Pathology REasoning-Guided REport Generation), MICCAI 2026.
From an H&E whole-slide image the system produces (interf1) an explicit pathologist-style
chain-of-thought over the challenge ontology plus a CAP-style pathology report, and (interf0) a
visual-grounding response for a region-of-interest crop.

## Team

- Team name: Turocrates
- Participants: Devansh Lalwani (primary), Swapnil Bhat
- Contact: founders@turocrates.ai
- Grand Challenge username: turocrates.ai
- Grand Challenge profile URL: https://grand-challenge.org/users/turocrates.ai/

## Method (summary)

We call the method the Grounded Ontology Walker (GOW). It is discriminative and LLM-free.

1. QC and tiling: GrandQC finds tissue; the slide is cut into 224 px tiles at 20x (0.5 MPP).
2. Tile encoding: each tile is embedded with Virchow2 (frozen, 2560-d). A slide becomes a bag [N, 2560].
3. Organ router: a gated-attention MIL head predicts the organ. A per-organ out-of-distribution gate
   (per-organ QDA Mahalanobis on the frozen bag-mean) detects organs not seen in training for ANY of
   the seven routes, not just cervix (leave-one-organ-out AUC 0.98). A flagged slide keeps its
   nearest in-ontology topology and only the organ and diagnosis NAMING opens to open-vocabulary
   matching over the CAP protocol library, so a false flag can only rename, never change a reasoning
   edge. The threshold is set for precision (about 2% of held-out in-distribution slides flagged),
   which still catches the dominant uterus shift and 9 of 10 novel organs on public TCGA slides.
4. The walker: starting from the organ node, at each step an answer head (a CLIP-style scorer that
   projects the question-conditioned tile evidence into CONCH text space and scores it against the
   allowed answers) picks the answer; a transition table mined from the 11,220 training chains then
   supplies the next question(s). No question string is ever invented, so every emitted edge is a
   valid ontology string.
5. The report: the final report is rendered deterministically from the collected answers using a
   per-organ CAP template (organ, procedure, ordered diagnoses, grade fields).
6. interf0: a CONCH plus GrandQC gate decides tissue vs background; the response is a deterministic
   closed-set answer (a refusal for background, a morphology assertion for tissue).

Only the official challenge dataset is used for training. Public pretrained models (Virchow2, CONCH,
KEEP, GrandQC, PubMedBERT, scispaCy) and public reference vocabularies (CAP protocol term lists) are
used as fixed frozen assets at inference. Public TCGA slides from organs outside the seven are used
solely to measure out-of-distribution generalization; no external data is used to train, fine-tune,
or select any parameter. A head-to-head of CONCH and KEEP shows open-vocabulary organ naming is
bounded by genuine morphological overlap (about four of ten novel organs open-vocabulary, six of ten
within a constrained set, similar for both encoders), so the container ships detection as the
reliable capability and open-vocabulary naming as a bounded one.

## Repository layout

```
gow/
  mining/        mine the ontology + transition table from the training chains
  extract/       WSI -> tissue tiles -> Virchow2 bag (wsi_io, grandqc_mask, extract_features)
  heads/         the model (gow_model), training (train_heads), CONCH text embeddings, OOD gate, CAP library
  walker/        the deterministic walker + report renderer (gow_walker), model-to-scorer eval (eval_real)
  interf0/       the visual-grounding responder (CONCH + GrandQC gate)
  scorer/        vendored copy of the organizers' scoring code (Haasha/REG2026), local eval only
  artifacts/     mined ontology (transitions, questions, answer_vocab, organ_meta), split, CAP library
  container/     the Grand Challenge submission container (interf0 + interf1)
```

## Environment

Python 3.12. Install the dependencies (a CUDA GPU is expected for embedding and training):

```
pip install torch timm==1.0.26 segmentation-models-pytorch==0.5.0 openslide-python \
            opencv-python-headless safetensors einops numpy Pillow tifffile imagecodecs \
            sentence-transformers spacy scispacy
pip install git+https://github.com/mahmoodlab/CONCH.git
python -m spacy download en_core_sci_lg    # scispaCy biomedical NER, for local report scoring
```

System packages for whole-slide IO: `libvips-tools`, `libvips42`, `openslide-tools`, `libopenslide0`.

## Model weights

Public backbones (download once into the HuggingFace cache or a local path):

- Virchow2: `paige-ai/Virchow2` on HuggingFace
- CONCH: `MahmoodLab/CONCH` on HuggingFace (checkpoint `conch_v1.pt`)
- GrandQC tissue and artifact UNets: from the GrandQC project
- PubMedBERT: `NeuML/pubmedbert-base-embeddings` (local scoring only)

Our trained heads (`gow_heads_final.pt`, roughly 31 MB) are included in this repository at
`gow/artifacts/gow_heads_final.pt` (well under GitHub's file-size limit), so no external download is
needed for our weights. The container build copies it to `model/gow_heads.pt` automatically via `stage.sh`.
Only the public backbones listed above are fetched (once) from HuggingFace / the GrandQC project.

## Reproduce training

```
# 1. mine the ontology + transition table from the official training chains
python gow/mining/mine_ontology.py --cot data/train_CoT_v01.json

# 2. build the patient-grouped, stratified train/val/test split
python gow/data_split.py --cot data/train_CoT_v01.json

# 3. precompute the CONCH text embeddings for the ontology answers
python gow/heads/precompute_conch_text.py

# 4. embed the training slides: WSI -> tissue tiles -> Virchow2 bags
bash gow/extract/run_embedding.sh

# 5. train the heads (organ router + answer scorer)
python gow/heads/train_heads.py --features-dir data/feats --cot data/train_CoT_v01.json \
       --text-emb gow/artifacts/text_emb.npz --fold-test --out gow/artifacts/gow_heads.pt

# 6. fit the out-of-distribution gate on the frozen features
python gow/heads/ood_gate.py --features-dir data/feats --cot data/train_CoT_v01.json
```

## Reproduce evaluation (held-out, real scorer)

```
python gow/walker/eval_real.py --heads gow/artifacts/gow_heads.pt --eval-split test
```

This runs the full WSI-to-chain pipeline on the held-out split and scores it with the vendored
official scorer (BPV, Edge-F1, MESS, final report), reporting per-organ and per-source breakdowns.

## Inference container (Grand Challenge submission)

```
cd gow/container
bash stage.sh          # bundle the source + stage weights into model/
bash do_test_run.sh    # build and run both interfaces offline on the sample inputs
bash do_save.sh        # export the image tar + model.tar.gz for upload
```

The container fills two functions:
- `src/interf1/model.py::predict_chain_of_thought(wsi_path)` returns the chain-of-thought array.
- `src/interf0/model.py::predict_visual_context_response(question_path, roi_image_path)` returns the answer.

It runs fully offline (no network), on a single 24 GB GPU, within the per-case time budget.
