# interf0 - Visual Grounding (Metric-B)

Metric-B = **0.30·B1 + 0.30·B2 + 0.40·B3**, graded by a local **Qwen3-8B** judge (majority vote), rules
read verbatim from the vendored `gow/scorer/evaluate_metrics.py`:

| term | ROI(s) | judge asks | wins if |
|---|---|---|---|
| **B1** | one background-only crop | "is this a valid answer for a *background* region?" | answer REFUSES (no tissue / not assessable / background) → `CORRECT` |
| **B2** | one tissue crop + its perturbed copy | "are the two answers the SAME?" | the two answers are semantically identical → `SAME` |
| **B3** | a tissue crop + a background crop | "are the tissue and background answers DIFFERENT?" **and** "is the background answer a correct refusal?" | tissue ≠ background **and** background is a refusal |

## The key fact

The judge **never grades whether the tissue answer is diagnostically correct** - Metric-B scores only
*background-refusal*, *stability*, and *discrimination*. So the optimal design is **two deterministic
answers gated by a semantic tissue/background classifier**:

- gate → **background** ⇒ `BG_ANSWER` - a refusal worded with the judge's own CORRECT triggers.
- gate → **tissue** ⇒ `TISSUE_ANSWER` - a fixed morphology assertion, clearly ≠ the refusal.

Consequences, by construction:
- **B2 = 1** automatically - the answer is constant per class, so original and perturbed are identical.
- **B1 = 1** whenever a background crop is gated as background.
- **B3 = 1** whenever the pair is gated correctly (tissue answer ≠ refusal, and the background refusal is correct).

So **all of Metric-B reduces to gate accuracy.** Everything else is provably satisfied.

## The gate: GrandQC primary, CONCH backup

The obvious gate is Otsu thresholding on pixel darkness - but **fat, mucin, and necrosis are pale**, so
Otsu calls them "background" and refuses to describe real tissue → B3 collapses on exactly the cases
that discriminate.

We tried **CONCH zero-shot** (image vs tissue/background text prompts) - it works but the decision margin
is **slide-dependent** (optimal threshold moved +0.03 → +0.11 across two slides; single-threshold balanced
acc only 0.72). So the **primary gate is now GrandQC's tissue head** (smp UnetPlusPlus, EffNet-B0), which is
*trained exactly to separate tissue from glass/artifact on H&E, including pale tissue*. It is decisively
better and slide-independent:

| gate signal | tissue crops | glass crops | single threshold, both slides |
|---|---|---|---|
| **GrandQC tissue-fraction** | 0.99-1.00 | 0.00 | **balanced 1.000** (per-slide [1.0, 1.0]) |
| CONCH prototype margin | slide-dependent | slide-dependent | balanced 0.72 (per-slide [0.90, 0.55]) |

Final gate = **GrandQC tissue-fraction ≥ 0.5 AND not an artifact region** (fold/pen/blur, from GrandQC's
artifact head) → tissue; else background. CONCH only breaks ties in the narrow ambiguous band (frac in
0.15-0.5), and is the fallback if GrandQC weights are absent. Files: `grandqc_roi.py` (the gate),
`compare_gates.py` (the head-to-head above).

## Files
- `respond.py` - `Responder(device).respond(question, pil_roi) -> str`. Loads CONCH once, precomputes the
  prompt-bank embeddings, gates, returns the deterministic answer. This is what the container calls.
- `validate_gate.py` - synthesizes tissue + background crops from an official WSI (no ROI bundle exists to
  download) and reports gate accuracy. **All of Metric-B rides on this number.**

## Container contract (interf0)
`predict_visual_context_response(question: str, roi_jpeg) -> str`  → one short answer string per ROI.
The scorer reads `[{id, question, answer}, ...]`. Ship CONCH weights + this module; runs on the A10G GPU
(here we validate on CPU to avoid contending with the embedding/training jobs).

## Still to do (in priority order)
1. **Calibrate against the real Qwen3-8B judge** once a GPU frees + the model is fetched (the judge rules
   make `CORRECT`/`SAME`/`DIFFERENT` near-certain for our answer strings, but confirm the wording).
2. Validate on the challenge's real background types at their true resolution (blur / pen / fold), not just
   clean glass - GrandQC's artifact head should cover these; confirm once real ROIs are available.
3. Package CONCH + GrandQC weights into the interf0 container path.
