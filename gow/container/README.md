# REG2026 Algorithm Submission Guide

How to adapt this template and submit a valid algorithm to Grand Challenge.

> 📌 **Official challenge site:** [reg2026.grand-challenge.org](https://reg2026.grand-challenge.org) - challenge overview, phase dates (Debug, Test 1, Test 2), data download, task and evaluation descriptions, rules, and platform notices. Check there for current phase status and sample data releases before you submit.

> ℹ️ **Icons in this README:** 📌 reference · ⚠️ warning · 🚫 do not · ✅ do · 📖 further reading

---

## Table of Contents

1. [Docker](#1-docker)
2. [How the container runs](#2-how-the-container-runs)
3. [What is an interface?](#3-what-is-an-interface)
4. [One case per container start](#4-one-case-per-container-start)
5. [Interface detection](#5-interface-detection)
6. [Interface 0 - Visual Grounding](#6-interface-0--visual-grounding)
7. [Interface 1 - Workflow Reasoning](#7-interface-1--workflow-reasoning)
8. [Repository layout](#8-repository-layout)
9. [Model weights](#9-model-weights)
10. [Local testing](#10-local-testing)
11. [Uploading and submitting](#11-uploading-and-submitting)
12. [Fixed paths](#12-fixed-paths)

---

## 1. Docker

Grand Challenge runs your algorithm inside a **Docker container**: your code, Python dependencies, and runtime packaged together. The same image should behave the same locally and on the platform.

| Term | Meaning |
|---|---|
| **Image** | What you build and upload (from a Dockerfile). |
| **Container** | One running instance of that image for a single inference job. |

The included [`Dockerfile`](Dockerfile) is a **sample** - it uses a PyTorch CUDA base image and installs [`requirements.txt`](requirements.txt). You may replace it with another base (e.g. a slimmer Python image, conda, or your own stack) as long as the image still runs `python inference.py` as the entrypoint and includes everything your code needs at runtime.

> 🚫 **Do not put weights in the Docker image.** [`do_build.sh`](do_build.sh) builds an image from the [`Dockerfile`](Dockerfile), which copies **code only** - not [`model/`](model/). Do not add `COPY model/ ...` to the Dockerfile.

> ✅ **Put weights in [`model/`](model/) on disk.** Pack that folder with [`do_save.sh`](do_save.sh) into **`model.tar.gz`** and upload it under **Algorithm → Models**. At run time the platform mounts those files at `/opt/ml/model/` (section 9).

You ship **one image** that supports **both** interfaces (section 3).

---

## 2. How the container runs

For each case, the platform starts a fresh container, mounts inputs read-only at `/input`, mounts `/output` for your results, and places uploaded weights at `/opt/ml/model/`. The entrypoint runs [`inference.py`](inference.py), which writes the required output file(s) and exits. The platform reads `/output` and scores it.

In practice:

- ⚠️ **No internet** at runtime - put dependencies in the image and weights in the model tarball.
- ℹ️ **One case per run** - section 4.
- ⚠️ **Write submission results only under `/output`.**

Local runs use the same layout via [`do_build.sh`](do_build.sh), [`do_test_run.sh`](do_test_run.sh), and [`do_save.sh`](do_save.sh).

---

## 3. What is an interface?

REG2026 has two task types. Grand Challenge calls each an **interface**: a fixed contract of inputs and outputs.

| Interface | Task | Metric | Inputs | Output file |
|---|---|---|---|---|
| **0** | Visual Grounding | B | ROI thumbnail (`.jpeg`) + question (JSON) | `visual-context-response.json` |
| **1** | Workflow Reasoning | A | Whole slide image (`<uid>.tiff`, opaque UUID hash) | `chain-of-thought.json` |

Your container must implement **both**. The platform picks the interface per run (section 5); you do not branch manually.

---

## 4. One case per container start

This applies to every evaluation phase - **Debug**, **Test 1**, and **Test 2** (see [Important dates](https://reg2026.grand-challenge.org/) on the challenge site).

> ⚠️ **One container = one case.** A common mistake is assuming one container processes an entire phase in a single run. It does not.

**Example (Debug phase):** suppose that phase has **10 ROI cases** (Interface 0) and **7 WSIs** (Interface 1). The platform runs your container **17 times** in total for that phase:

1. **10 runs** for Interface 0 - each with one thumbnail, one question, and one answer in `/output`.
2. **7 runs** for Interface 1 - each with one WSI and one `chain-of-thought.json` in `/output`.

Each run is independent: new container, empty `/output`, no shared state from earlier runs.

Design accordingly:

- Load the model, infer on the single case, write output, exit.
- Model load cost is paid **every** run - keep startup reasonable.
- 🚫 Do not batch or aggregate across cases in one container.

---

## 5. Interface detection

Before your code runs, the platform writes `/input/inputs.json` with the input sockets for this case. [`core.py`](core.py) reads the slugs; [`inference.py`](inference.py) dispatches to `interf0_handler` or `interf1_handler`.

> ℹ️ You do not need to edit interface detection in `core.py` or `inference.py`.

---

## 6. Interface 0 - Visual Grounding

### Platform paths

[`inference.py`](inference.py) builds these paths and passes them into your function (do not change the paths themselves):

| Role | Path |
|---|---|
| Question | `/input/visual-context-question.json` |
| ROI thumbnail | `/input/histopathology-region-of-interest-thumbnail.jpeg` |
| Interface selector | `/input/inputs.json` (read by `core.py`) |
| Your answer | `/output/visual-context-response.json` |

### Your code

Edit [`src/interf0/model.py`](src/interf0/model.py) - function `predict_visual_context_response`:

```python
def predict_visual_context_response(
    *, question_path: Path, roi_image_path: Path
) -> str:
    ...
```

The template loads inputs with [`load_json_file`](core.py) and [`load_roi_image`](core.py) (logs file size and pixel stats to verify the ROI is not empty or corrupt), sets `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`, and includes a placeholder return you should replace.

> 🚫 **Do not change the return type** (`str`). `inference.py` writes it directly to `visual-context-response.json`.

### Output format

A **plain JSON string** - not an object or array:

```json
"Yes, glandular epithelium is clearly visible."
```

---

## 7. Interface 1 - Workflow Reasoning

### Platform paths

Each Interface 1 case is one whole-slide image. The platform mounts it under a **fixed directory**, but the **filename is an opaque `<uid>`** (a long UUID-style hash), not a human-readable slide name and not a generic name like `whole-slide-image.tiff`.

| Role | Path |
|---|---|
| WSI directory | `/input/images/whole-slide-image/` |
| WSI file | `/input/images/whole-slide-image/<uid>.tiff` |
| Filename in `inputs.json` | `image.name` on the `whole-slide-image` socket (e.g. `d021e460-42d8-4a72-b83e-f07050d8468a.tiff`) |
| Interface selector | `/input/inputs.json` (read by `core.py`) |
| Your chain of thought | `/output/chain-of-thought.json` |

> ℹ️ **`<uid>`** is an **anonymous platform identifier** (typically a UUID such as `d021e460-42d8-4a72-b83e-f07050d8468a`). It is **not** the original slide filename (e.g. not `PIT_01_00020_01.tiff`). [`resolve_wsi_path()`](core.py) reads `image.name` from `/input/inputs.json` and opens `/input/images/whole-slide-image/<uid>.tiff`. Do not hard-code or guess the uid in your code.

**Example layout for one case:**

```
/input/
├── inputs.json
└── images/
    └── whole-slide-image/
        └── d021e460-42d8-4a72-b83e-f07050d8468a.tiff   ← <uid>.tiff (hash varies per case)
```

### Your code

Edit [`src/interf1/model.py`](src/interf1/model.py) - function `predict_chain_of_thought`:

```python
def predict_chain_of_thought(*, wsi_path: Path) -> list[ChainOfThoughtStep]:
    ...
```

`ChainOfThoughtStep` is defined in the same file. [`inference.py`](inference.py) resolves `wsi_path` via [`resolve_wsi_path()`](core.py), then the template loads it with [`load_wsi_array`](core.py) ([tifffile](https://pypi.org/project/tifffile/)) and logs array shape, dtype, min/max/mean/std, and sample values to verify the slide is not empty or corrupt. For large slides you may use `memmap`, OpenSlide, or cuCIM in your own code; keep `wsi_path` as the entry point.

> 🚫 **Do not change the return type** (`list[ChainOfThoughtStep]`). `inference.py` writes the list directly to `chain-of-thought.json`.

[`inference.py`](inference.py) also calls `show_torch_cuda_info()` for this interface so GPU availability appears in the logs.

### Output format

A **bare JSON array** of steps. Each step:

```json
{
    "question": "...",
    "answer": "...",
    "next_question": "..."
}
```

Example:

```json
[
    {
        "question": "What type of specimen is this?",
        "answer": "The specimen is a surgically resected tissue section.",
        "next_question": "What is the predominant tissue architecture observed in this specimen?"
    },
    {
        "question": "What is the predominant tissue architecture observed in this specimen?",
        "answer": "The tissue shows predominantly glandular structures.",
        "next_question": "Are there morphological features suggestive of malignancy?"
    },
    {
        "question": "Are there morphological features suggestive of malignancy?",
        "answer": "There are features including nuclear pleomorphism and increased mitotic figures.",
        "next_question": ""
    }
]
```

**Rules:**

- ⚠️ Use the **exact canonical** `question` / `next_question` strings from training annotations where the metric requires them.
- ℹ️ Last step: `"next_question": ""`.
- 🚫 Do **not** wrap the array in `{"id": ...}` - the platform adds that.

---

## 8. Repository layout

```
reg2026_algorithm_submission_template/
├── inference.py          # entrypoint - dispatches by interface
├── core.py               # paths, I/O helpers, interface detection (do not edit)
├── Dockerfile            # sample - swap base image or layout if you prefer
├── requirements.txt      # add dependencies here
├── do_build.sh           # build Docker image
├── do_test_run.sh        # local test (both interfaces)
├── do_save.sh            # export image + model.tar.gz for upload
├── README.md             # this file
├── model/                # local weights → /opt/ml/model/ (see model/README.md)
├── src/
│   ├── interf0/model.py  # predict_visual_context_response()
│   └── interf1/model.py  # predict_chain_of_thought()
└── test/                 # mirrors platform /input and /output layout
    ├── input/
    │   ├── interf0/      # ROI .jpeg + question JSON
    │   └── interf1/
    │       ├── inputs.json              # image.name → <uid>.tiff (UUID hash)
    │       └── images/whole-slide-image/
    │           └── d021e460-42d8-4a72-b83e-f07050d8468a.tiff
    └── output/           # written by do_test_run.sh (gitignored)
```

Implement inference in the `src/interf*/model.py` files or import from a package under `src/`. List any new packages in [`requirements.txt`](requirements.txt).

---

## 9. Model weights

Weights are **not** part of the Docker image. They live in the **[`model/`](model/)** directory in this repo and are uploaded to Grand Challenge as a **separate** `model.tar.gz` under **Algorithm → Models**. Before each run, the platform extracts that tarball to **`/opt/ml/model/`** inside the container.

| Location | When | Purpose |
|---|---|---|
| [`model/`](model/) (this repo) | Development | Store checkpoints and other large files locally |
| `/opt/ml/model/` (container) | Every run | Runtime path - use `MODEL_PATH` from [`core.py`](core.py) |

**✅ Do**

- Put all checkpoints and large model assets under [`model/`](model/) (any layout you like, e.g. `model/weights.pt`, `model/my_model/config.json`).
- Load weights at runtime from `MODEL_PATH` (maps to `/opt/ml/model/`).
- Run [`do_save.sh`](do_save.sh) to build `model.tar.gz` for upload alongside the container image.

**🚫 Do not**

- Add `COPY model/ …` (or similar) to the [`Dockerfile`](Dockerfile) - weights are provided via `model.tar.gz`, not the image.

```python
from core import MODEL_PATH

model.load_state_dict(torch.load(MODEL_PATH / "weights.pt", map_location=device))
```

For local testing, [`do_test_run.sh`](do_test_run.sh) bind-mounts [`model/`](model/) read-only to `/opt/ml/model/`, matching the platform. See [`model/README.md`](model/README.md).

---

## 10. Local testing

Run [`do_test_run.sh`](do_test_run.sh) from this directory. It:

1. Builds the image ([`do_build.sh`](do_build.sh)).
2. Mounts [`test/input/interf0/`](test/input/interf0/) and [`test/input/interf1/`](test/input/interf1/) as `/input` (same layout as the platform).
3. Runs Interface 0, then Interface 1, with no internet (GPU if available).

Keep sample cases under `test/input/interf0/` and `test/input/interf1/` in the repo. Outputs go to `test/output/` (not tracked).

**Interface 1 test fixture:** the sample WSI must be named `<uid>.tiff` under `test/input/interf1/images/whole-slide-image/`, and [`test/input/interf1/inputs.json`](test/input/interf1/inputs.json) must set the matching `image.name` (bundled example: `d021e460-42d8-4a72-b83e-f07050d8468a.tiff`). Use the same opaque hash for both the file on disk and `image.name` - not a slide basename. Do not use a fixed name like `whole-slide-image.tiff`.

Expected files after a successful run:

- [`test/output/interf0/visual-context-response.json`](test/output/interf0/visual-context-response.json)
- [`test/output/interf1/chain-of-thought.json`](test/output/interf1/chain-of-thought.json)

---

## 11. Uploading and submitting

1. Run [`do_save.sh`](do_save.sh) - produces `reg2026_algorithm_<timestamp>.tar.gz` (code image) and `model.tar.gz` (everything under [`model/`](model/)).
2. On Grand Challenge: upload the **container image** under **Algorithm → Container images** and **`model.tar.gz`** under **Algorithm → Models**.

   > 🚫 **Two separate uploads** - weights must not be embedded only in the container image.
3. Use **Try out algorithm** with sample inputs before a phase submission.
4. Submit the algorithm to the challenge phase when ready.

**📖 Further reading**

- 📌 [REG2026 challenge site](https://reg2026.grand-challenge.org) - registration, data, rules, evaluation method, and phase deadlines
- 📖 [Making a Challenge Submission](https://grand-challenge.org/documentation/making-a-challenge-submission/) - Grand Challenge container upload walkthrough

---

## 12. Fixed paths

These paths are defined in [`core.py`](core.py) and wired in [`inference.py`](inference.py).

> 🚫 **Do not change the directory layout or socket filenames** in `core.py` or `inference.py`. The WSI **basename** `<uid>.tiff` changes per case; resolve it with [`resolve_wsi_path()`](core.py), do not hard-code a single filename.

| Path | Interface | Purpose |
|---|---|---|
| `/input/inputs.json` | Both | Interface selector |
| `/input/visual-context-question.json` | 0 | Question |
| `/input/histopathology-region-of-interest-thumbnail.jpeg` | 0 | ROI thumbnail |
| `/input/images/whole-slide-image/` | 1 | WSI directory (one file per case) |
| `/input/images/whole-slide-image/<uid>.tiff` | 1 | WSI file (`<uid>` = opaque UUID in `image.name`) |
| `/output/visual-context-response.json` | 0 | Your answer |
| `/output/chain-of-thought.json` | 1 | Your reasoning steps |
| `/opt/ml/model/` | Both | Model weights |
