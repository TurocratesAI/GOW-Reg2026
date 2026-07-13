# Root Dockerfile for Grand Challenge's "link a GitHub repository" build (GC requires the Dockerfile at the
# repo root; build context = repo root). Produces the SAME algorithm image as gow/container/Dockerfile, but with
# repo-root-relative COPY paths and a committed gow/container/gowsrc/ so the build is self-contained (no staging).
# Model weights are NOT baked in -> they are attached separately as a Grand Challenge "Model" (model.tar.gz),
# mounted at /opt/ml/model at run time.
FROM --platform=linux/amd64 pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime AS reg2026_algorithm_amd64
# CUDA 11.8 (not 12.6): runs on any A10G driver (450+). See gow/container/Dockerfile for the rationale.

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/app
ENV GOW_SRC=/opt/app/gowsrc
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# System deps for WSI IO: openslide (tiled tiffs) + libvips (striped BigTIFF -> tiled).
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libvips-tools libvips42 openslide-tools libopenslide0 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app

COPY --chown=user:user gow/container/requirements.txt /opt/app/
RUN python -m pip install --user --no-cache-dir --no-color --requirement /opt/app/requirements.txt

# App code + the bundled GOW source tree (committed at gow/container/gowsrc/: extract, heads, walker, interf0,
# data_split.py, and the SMALL artifacts - transitions/questions/answer_vocab/organ_meta/text_emb; NO weights).
COPY --chown=user:user gow/container/core.py      /opt/app/
COPY --chown=user:user gow/container/gowcfg.py    /opt/app/
COPY --chown=user:user gow/container/inference.py /opt/app/
COPY --chown=user:user gow/container/src/         /opt/app/src/
COPY --chown=user:user gow/container/gowsrc/      /opt/app/gowsrc/

# Version label: bumps the image config so Grand Challenge imports this as a new, distinct image (its checksum
# differs from any previously-uploaded/built image). No functional effect.
LABEL org.opencontainers.image.title="reg2026-gow-algorithm" \
      org.opencontainers.image.version="0.2"

ENTRYPOINT ["python", "inference.py"]
