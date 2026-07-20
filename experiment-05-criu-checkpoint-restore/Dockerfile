# Reproducible environment for Experiment 05 — CRIU checkpoint/restore.
#
# Captures the exact toolchain used on the VMs: Ubuntu 24.04 + criu 4.2 (from the
# CRIU PPA, since criu was dropped from Noble's repos) + NVIDIA cuda-checkpoint +
# CUDA torch + transformers.
#
# NOTE: this image is for *reproducing the setup*. Actually running CRIU inside a
# container needs extra privileges at runtime, and the GPU stage needs the NVIDIA
# runtime + a recent host driver (r550+). The experiments themselves were run
# directly on VMs, not in this container.
#
# Build:  docker build -t <you>/criu-coldstart:latest .
# Push:   docker login && docker push <you>/criu-coldstart:latest
# CPU checkpoint run (needs privileges):
#   docker run --rm -it --privileged --pid=host <you>/criu-coldstart:latest bash
# GPU stage (needs nvidia runtime + host driver r550+):
#   docker run --rm -it --gpus all --privileged <you>/criu-coldstart:latest bash

# CUDA runtime base so torch finds CUDA libs; matches the H100 box (driver r580).
FROM nvidia/cuda:12.6.2-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

# --- criu (from the CRIU PPA: not in Ubuntu 24.04 repos, Launchpad #2066148) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl python3 python3-venv python3-pip \
    && add-apt-repository -y ppa:criu/ppa \
    && apt-get update && apt-get install -y --no-install-recommends criu \
    && rm -rf /var/lib/apt/lists/*

# --- NVIDIA cuda-checkpoint (note the bin/x86_64_Linux/ path) ---
RUN curl -fsSL -o /usr/local/bin/cuda-checkpoint \
      https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint \
    && chmod +x /usr/local/bin/cuda-checkpoint

# --- Python deps in a venv ---
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch transformers accelerate safetensors

WORKDIR /work
COPY scripts/ /work/scripts/

# Sanity: criu present, cuda plugin present, cuda-checkpoint present.
RUN criu --version && ls -l /usr/lib/criu/cuda_plugin.so && cuda-checkpoint --version || true

CMD ["bash"]
