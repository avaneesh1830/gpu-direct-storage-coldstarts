# 🛠️ Complete Setup & Experiment Guide

> End-to-end walkthrough: how the CPU and GPU instances were used, why the CPU
> instance was only needed once, and every command used across the H100, H200,
> and B200 experiments. (B100 was dropped — see note below.)

---

## 🧭 The Big Picture — Why Two Instances?

This project used **two kinds of cloud instances** on Brev, each with a distinct job:

| Instance | Job | How often |
|---|---|---|
| **CPU instance** | Build the x86_64 Docker image **once** and push it to Docker Hub | One time |
| **GPU instance** (H100 / H200 / B200) | Pull that image and actually run + benchmark the models | Every experiment |

### Why was the CPU instance only needed once?

The Docker image is **architecture-specific**. Our image is built on top of
`vllm/vllm-openai`, which is compiled for **x86_64 (amd64)** CPUs — the same
architecture as all of Brev's cloud GPU instances.

Key insight:

> **Building the image and running the image are separate steps.**
> The image only needs to be *built* once. After it's pushed to Docker Hub,
> every GPU instance just *pulls* the finished image — no rebuilding needed.

So the workflow is:

```
[CPU instance]  build image  ──push──►  [Docker Hub]
                                              │
                                              ├──pull──►  [H100 instance]  run + benchmark
                                              ├──pull──►  [H200 instance]  run + benchmark
                                              └──pull──►  [B200 instance]  run + benchmark
```

The CPU instance is cheap (~$1.60/hr) compared to GPU instances ($2-5+/hr), so
doing the slow image build on the CPU box saves money — we don't burn expensive
GPU-hours on a build step that doesn't need a GPU at all.

> **Note:** We *could* have built the image on a GPU instance too (and in fact
> did, as a fallback). But the clean approach is: build once on CPU → push →
> pull everywhere. The image is identical regardless of where it's pulled.

---

## 📦 What's in the Docker Image?

A **single reusable image** (`avaneesharoor/ml-experiments:latest`) that works for
every model size and every GPU. The model is chosen at *runtime*, not baked into
the image. This is why one image covers all 16 experiments (4 GPUs × 4 model sizes).

```dockerfile
FROM vllm/vllm-openai:latest

RUN pip install --no-cache-dir \
    huggingface_hub accelerate sentencepiece protobuf einops autoawq

ENV HF_HOME=/model-cache
ENV TRANSFORMERS_CACHE=/model-cache
ENV HF_HUB_CACHE=/model-cache

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

CMD ["--model", "Qwen/Qwen2.5-1.5B-Instruct", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--dtype", "bfloat16", \
     "--gpu-memory-utilization", "0.90", \
     "--max-model-len", "8192"]
```

---

# PART 1 — CPU Instance (build the image once)

## Step 1.1 — Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker          # reload group without logging out
```

## Step 1.2 — Create the project folder and Dockerfile

```bash
mkdir -p ~/project && cd ~/project

cat > Dockerfile << 'DOCKEREOF'
FROM vllm/vllm-openai:latest

RUN pip install --no-cache-dir \
    huggingface_hub accelerate sentencepiece protobuf einops autoawq

ENV HF_HOME=/model-cache
ENV TRANSFORMERS_CACHE=/model-cache
ENV HF_HUB_CACHE=/model-cache

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

CMD ["--model", "Qwen/Qwen2.5-1.5B-Instruct", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--dtype", "bfloat16", \
     "--gpu-memory-utilization", "0.90", \
     "--max-model-len", "8192"]
DOCKEREOF
```

## Step 1.3 — Build and push to Docker Hub

```bash
docker login                                            # enter Docker Hub credentials
docker build -t avaneesharoor/ml-experiments:latest .   # ~10-15 min (pulls ~10GB base)
docker push avaneesharoor/ml-experiments:latest         # uploads to Docker Hub
```

**That's it for the CPU instance.** You can shut it down now to stop billing —
the image lives on Docker Hub permanently.

---

# PART 2 — GPU Instance Setup (same for H100 / H200 / B200)

Every GPU instance needs Docker **and** the NVIDIA Container Toolkit (so the
container can see the GPU). Run this once per GPU instance.

## Step 2.1 — Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

## Step 2.2 — Install NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update -y
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Step 2.3 — Reload docker group and verify GPU

```bash
newgrp docker
nvidia-smi             # should show the GPU (H100 / H200 / B200)
```

## Step 2.4 — Pull the image and prepare folders

```bash
docker pull avaneesharoor/ml-experiments:latest
mkdir -p ~/project && cd ~/project
mkdir -p /ephemeral/model-cache    # weights cached here (large disk)
```

## Step 2.5 — Add the benchmark script

Copy `benchmark.py` into `~/project/` (drag-drop in VS Code, or scp it from your Mac).
Then install its dependencies:

```bash
pip install aiohttp gputil psutil
```

> **If pip fails** (some instances ship a locked virtualenv — seen on the B200
> Shadeform image), use one of these instead:
> ```bash
> # Option A: bypass the system-package lock
> pip install --break-system-packages aiohttp gputil psutil
>
> # Option B: install to a local folder and point Python at it
> pip install --break-system-packages --target=$HOME/pylibs aiohttp gputil psutil
> echo 'export PYTHONPATH=$HOME/pylibs:$PYTHONPATH' >> ~/.bashrc
> export PYTHONPATH=$HOME/pylibs:$PYTHONPATH
> ```
> If `GPUtil` specifically can't install, that's fine — the benchmark catches the
> missing import and simply skips GPU VRAM/util readings (the throughput numbers
> still confirm the GPU is working). Remember to set `PYTHONPATH` in **every new
> terminal** if you used Option B and didn't add it to `~/.bashrc`.

---

# PART 3 — Running Experiments

## How each run works

You need **two terminals** on the GPU instance:
- **Terminal 1** runs the vLLM server (stays running — never close it)
- **Terminal 2** runs the benchmark against it

Optionally a **Terminal 3** with `watch -n1 nvidia-smi` to watch GPU live.

> **Entrypoint note:** The image on Docker Hub was built with a `python`
> entrypoint that some instances don't recognize. To be safe, every run command
> below overrides it with `--entrypoint python3` and calls the vLLM server module
> directly. (If you rebuild the image with the corrected `python3` entrypoint,
> you can drop the `--entrypoint python3 ... -m vllm.entrypoints.openai.api_server`
> override and just pass the flags.)

> **Port already allocated?** If you Ctrl+C a server and the port stays held,
> clear it with: `docker stop $(docker ps -q)`

---

## 🟢 H100 80GB Experiment

The H100 has 80GB VRAM, so the 32B and 70B models **must** use AWQ INT4
quantization to fit.

### 1B — Qwen2.5-1.5B (BF16)
```bash
# Terminal 1
docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /ephemeral/model-cache:/model-cache -p 8000:8000 \
  --entrypoint python3 avaneesharoor/ml-experiments:latest \
  -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --max-model-len 8192

# Terminal 2
python3 benchmark.py --size 1B --model Qwen/Qwen2.5-1.5B-Instruct
```

### 10B — Qwen2.5-7B (BF16)
```bash
# Terminal 1
docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /ephemeral/model-cache:/model-cache -p 8000:8000 \
  --entrypoint python3 avaneesharoor/ml-experiments:latest \
  -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --max-model-len 8192

# Terminal 2
python3 benchmark.py --size 10B --model Qwen/Qwen2.5-7B-Instruct
```

### 30B — Qwen2.5-32B-AWQ (AWQ INT4 — required to fit on 80GB)
```bash
# Terminal 1
docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /ephemeral/model-cache:/model-cache -p 8000:8000 \
  --entrypoint python3 avaneesharoor/ml-experiments:latest \
  -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --quantization awq_marlin \
  --host 0.0.0.0 --port 8000 --dtype float16 \
  --gpu-memory-utilization 0.90 --max-model-len 8192

# Terminal 2
python3 benchmark.py --size 30B --model Qwen/Qwen2.5-32B-Instruct-AWQ
```

### 110B — Llama-3.1-70B-AWQ (AWQ INT4 — required to fit on 80GB)
```bash
# Terminal 1
docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /ephemeral/model-cache:/model-cache -p 8000:8000 \
  --entrypoint python3 avaneesharoor/ml-experiments:latest \
  -m vllm.entrypoints.openai.api_server \
  --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
  --quantization awq_marlin \
  --host 0.0.0.0 --port 8000 --dtype float16 \
  --gpu-memory-utilization 0.92 --max-model-len 4096

# Terminal 2
python3 benchmark.py --size 110B --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
```

---

## 🔵 H200 141GB Experiment

The H200 has 141GB VRAM, so the 32B model runs at **full BF16** (no quantization
needed — better quality). The 123B model still uses AWQ INT4.

### 1B — Qwen2.5-1.5B (BF16)
```bash
docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /ephemeral/model-cache:/model-cache -p 8000:8000 \
  --entrypoint python3 avaneesharoor/ml-experiments:latest \
  -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --max-model-len 8192

python3 benchmark.py --size 1B --model Qwen/Qwen2.5-1.5B-Instruct
```

### 10B — Qwen2.5-7B (BF16)
```bash
docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /ephemeral/model-cache:/model-cache -p 8000:8000 \
  --entrypoint python3 avaneesharoor/ml-experiments:latest \
  -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --max-model-len 8192

python3 benchmark.py --size 10B --model Qwen/Qwen2.5-7B-Instruct
```

### 30B — Qwen2.5-32B (full BF16 — fits on 141GB, no quantization)
```bash
docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /ephemeral/model-cache:/model-cache -p 8000:8000 \
  --entrypoint python3 avaneesharoor/ml-experiments:latest \
  -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-32B-Instruct \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --max-model-len 8192

python3 benchmark.py --size 30B --model Qwen/Qwen2.5-32B-Instruct
```

### 110B — Mistral-Large-2407 123B (AWQ INT4)
```bash
docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /ephemeral/model-cache:/model-cache -p 8000:8000 \
  --entrypoint python3 avaneesharoor/ml-experiments:latest \
  -m vllm.entrypoints.openai.api_server \
  --model casperhansen/mistral-large-instruct-2407-awq \
  --quantization awq_marlin \
  --host 0.0.0.0 --port 8000 --dtype float16 \
  --gpu-memory-utilization 0.92 --max-model-len 4096

python3 benchmark.py --size 110B --model casperhansen/mistral-large-instruct-2407-awq
```

---

## 🟡 B100 — Skipped (not available)

> **B100 was dropped from this study.** Virtually no cloud provider stocks the
> B100 — most went straight from the Hopper series (H100/H200) to the B200.
> Brev had no B100 inventory available. Since the B100 and B200 share the same
> 192GB VRAM and 8 TB/s memory bandwidth (the B200 only adds higher FP4 compute,
> which this BF16/AWQ benchmark doesn't exercise), the **B200 results fully
> represent the Blackwell tier** for this bandwidth-bound workload.

---

## 🔴 B200 192GB Experiment ✅ Complete

B200 (Blackwell, 192GB VRAM, 8 TB/s bandwidth). Delivered the fastest TPOT of all
three GPUs — confirming the bandwidth-bound nature of decode. The 32B ran at full
BF16 and still beat the H100's quantized version. The exact commands used:

```bash
# 1B
docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /ephemeral/model-cache:/model-cache -p 8000:8000 \
  --entrypoint python3 avaneesharoor/ml-experiments:latest \
  -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --max-model-len 8192
python3 benchmark.py --size 1B --model Qwen/Qwen2.5-1.5B-Instruct

# 10B
docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /ephemeral/model-cache:/model-cache -p 8000:8000 \
  --entrypoint python3 avaneesharoor/ml-experiments:latest \
  -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --max-model-len 8192
python3 benchmark.py --size 10B --model Qwen/Qwen2.5-7B-Instruct

# 30B (full BF16)
docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /ephemeral/model-cache:/model-cache -p 8000:8000 \
  --entrypoint python3 avaneesharoor/ml-experiments:latest \
  -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-32B-Instruct \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --max-model-len 8192
python3 benchmark.py --size 30B --model Qwen/Qwen2.5-32B-Instruct

# 110B — Mistral-Large 123B (AWQ INT4)
docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /ephemeral/model-cache:/model-cache -p 8000:8000 \
  --entrypoint python3 avaneesharoor/ml-experiments:latest \
  -m vllm.entrypoints.openai.api_server \
  --model casperhansen/mistral-large-instruct-2407-awq \
  --quantization awq_marlin \
  --host 0.0.0.0 --port 8000 --dtype float16 \
  --gpu-memory-utilization 0.92 --max-model-len 4096
python3 benchmark.py --size 110B --model casperhansen/mistral-large-instruct-2407-awq
```

---

# PART 4 — Saving Results & Cleanup

## Copy results to your Mac (run on your Mac terminal, not SSH)

```bash
scp shadeform@<gpu-instance-ip>:~/project/benchmark_results.json \
    ~/Desktop/brev-bottleneck-loading_models-project/results/
```

## Shut down instances to stop billing

Always shut down both CPU and GPU instances from the Brev dashboard when done —
GPU instances cost $2-5+/hr.

---

# 📋 Quick Reference — Model Map

| Size label | H100 | H200 | B200 |
|---|---|---|---|---|
| 1B | Qwen2.5-1.5B (BF16) | same | same |
| 10B | Qwen2.5-7B (BF16) | same | same |
| 30B | Qwen2.5-32B **AWQ** | Qwen2.5-32B **BF16** | BF16 |
| 110B | Llama-3.1-70B AWQ | Mistral-Large-123B AWQ | Mistral-Large-123B AWQ |

> The difference between GPUs is **how much quantization is forced**:
> H100 (80GB) needs AWQ for 30B+. H200/B200 (141-192GB) can run larger
> models at full precision. This is the core story of the whole project.

---

## ⚠️ Known Measurement Caveats (documented honestly)

1. **GPU util shows 0%** — GPUtil samples the host between requests, missing the
   active-inference window. Throughput numbers confirm the GPU is working.
2. **TTFT is queuing-dominated** — the benchmark fires 5 concurrent requests, so
   reported TTFT reflects queue wait, not pure single-request prefill. For true
   prefill timing, run requests sequentially (concurrency=1).
3. **System RAM low (~8-10GB)** — confirms the model lives in GPU VRAM, not
   system memory, as expected for GPU inference.
4. **Prefix caching effect** — running the same benchmark twice back-to-back
   (observed on B200, 32B) dropped TTFT from 473ms to 43ms with identical TPOT.
   vLLM caches the KV for previously-seen prompts. For consistency, all reported
   numbers use the **cold-cache first run** of each model.
5. **B200 GPUtil unavailable** — on the B200 instance GPUtil couldn't install
   (locked venv), so its VRAM/util fields read 0. This is a tooling artifact;
   throughput confirms the GPU was fully engaged.

