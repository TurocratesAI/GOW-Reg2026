FROM --platform=linux/amd64 pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime AS reg2026_algorithm_amd64
# PyTorch + CUDA base so GPU is available at inference time. CUDA 11.8 (NOT the template's 12.6): the metrics
# from a Grand Challenge run showed GPU util 0 with the container crawling on 1 CPU thread -> torch could not
# see the A10G because the 12.6 runtime needs a very new NVIDIA driver (~560). CUDA 11.8 runs on driver 450+,
# so the A10G is usable regardless of GC's exact driver. A10G is Ampere (sm_86), fully supported by 11.8.

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/app
ENV GOW_SRC=/opt/app/gowsrc
# Fully offline at inference: weights are baked (image) or mounted (/opt/ml/model); never fetch.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# System deps for WSI IO: openslide (reads tiled tiffs) + libvips (converts PIT_03 striped BigTIFFs).
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libvips-tools libvips42 openslide-tools libopenslide0 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# App code + our GOW source tree (gowsrc/ = extract, heads, walker, interf0, data_split.py, artifacts).
COPY --chown=user:user core.py      /opt/app/
COPY --chown=user:user gowcfg.py    /opt/app/
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user src/         /opt/app/src/
COPY --chown=user:user gowsrc/      /opt/app/gowsrc/

ENTRYPOINT ["python", "inference.py"]
