#!/usr/bin/env python3
"""
Experiment 2: LLM cold-start / loading bottleneck analysis.

Goal: decompose where model-loading time goes (disk -> page cache -> PCIe ->
VRAM -> engine ready -> first token) as the baseline for GPU Direct Storage /
CRIU / cuda-checkpoint work.

Modes:
  load  (default) : time vLLM engine init + first-token latency + decode sanity check
  disk            : raw sequential read of the model's safetensors files (no GPU)

Protocol (see README.md): run each model 3 ways and diff the numbers:
  1. cold page cache  -> disk + H2D + init + compile
  2. warm page cache  -> H2D + init + compile        (cold - warm ~= disk time)
  3. disk mode, cold  -> raw disk bandwidth ceiling
"""
import argparse
import glob
import json
import os
import statistics
import subprocess
import time
from datetime import datetime

# Same artifacts on every GPU so loading comparisons are apples-to-apples.
# (30B/110B use AWQ everywhere, even where BF16 would fit.)
MODELS = {
    "1b":   ("Qwen/Qwen2.5-1.5B-Instruct",                      None,         "bfloat16"),
    "10b":  ("Qwen/Qwen2.5-7B-Instruct",                        None,         "bfloat16"),
    "30b":  ("Qwen/Qwen2.5-32B-Instruct-AWQ",                   "awq_marlin", "float16"),
    "110b": ("casperhansen/mistral-large-instruct-2407-awq",    "awq_marlin", "float16"),
}


def gpu_info():
    raw = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current",
         "--format=csv,noheader,nounits"],
        text=True,
    ).strip().split("\n")[0]
    name, total, gen, width = [p.strip() for p in raw.split(",")]
    return {"name": name, "vram_total_mb": int(total),
            "pcie_gen": gen, "pcie_width": width}


def model_files(model_id):
    # Candidate cache roots: works with our image (HF_HOME=/model-cache) and
    # the NGC vLLM image (default ~/.cache/huggingface)
    hf_home = os.environ.get("HF_HOME", "/model-cache")
    bases = [
        os.environ.get("HF_HUB_CACHE") or os.path.join(hf_home, "hub"),
        hf_home,
        os.path.expanduser("~/.cache/huggingface/hub"),
    ]
    slug = "models--" + model_id.replace("/", "--")
    for base in bases:
        files = sorted(glob.glob(os.path.join(base, slug, "snapshots", "*", "*.safetensors")))
        if files:
            return files
    return []


def bench_disk(files, block_mb=64):
    """Sequential read of all weight files. With a cold page cache this is the
    raw disk bandwidth ceiling that GDS/streaming loaders are trying to hit."""
    block = block_mb * 1024 * 1024
    total = 0
    t0 = time.perf_counter()
    for path in files:
        with open(path, "rb", buffering=0) as f:
            while True:
                chunk = f.read(block)
                if not chunk:
                    break
                total += len(chunk)
    elapsed = time.perf_counter() - t0
    return total, elapsed


def bench_load(model_id, quant, dtype, enforce_eager, load_format, prompt_len, gen_tokens):
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=model_id,
        dtype=dtype,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enforce_eager=enforce_eager,
        enable_prefix_caching=False,   # so warm TTFT is a real prefill, not a cache hit
    )
    if quant:
        kwargs["quantization"] = quant
    if load_format != "auto":
        kwargs["load_format"] = load_format

    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    init_s = time.perf_counter() - t0

    prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_len // 10 + 1)
    sp_one = SamplingParams(temperature=0, max_tokens=1)

    # First request after init — the tail end of the cold start
    t0 = time.perf_counter()
    llm.generate([prompt], sp_one)
    ttft_first_ms = (time.perf_counter() - t0) * 1000

    # Steady-state TTFT (sequential, prefix cache off)
    warm = []
    for _ in range(3):
        t0 = time.perf_counter()
        llm.generate([prompt], sp_one)
        warm.append((time.perf_counter() - t0) * 1000)
    ttft_warm_ms = statistics.mean(warm)

    # Decode sanity check (single run — inference speed was exp 1's job)
    t0 = time.perf_counter()
    llm.generate([prompt], SamplingParams(temperature=0, max_tokens=gen_tokens))
    total_s = time.perf_counter() - t0
    decode_tps = (gen_tokens - 1) / (total_s - ttft_warm_ms / 1000)

    return {
        "llm_init_s": round(init_s, 2),
        "ttft_first_ms": round(ttft_first_ms, 1),
        "ttft_warm_ms": round(ttft_warm_ms, 1),
        "decode_tps": round(decode_tps, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), required=True)
    parser.add_argument("--mode", choices=["load", "disk"], default="load")
    parser.add_argument("--label", required=True,
                        help="cache state for this run, e.g. first_download / cold / warm / disk_cold")
    parser.add_argument("--model-id", default=None, help="override the model map")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="skip CUDA graph capture (diff vs default isolates compile time)")
    parser.add_argument("--load-format", default="auto",
                        choices=["auto", "fastsafetensors", "runai_streamer", "tensorizer", "instanttensor"],
                        help="fastsafetensors = GDS/cuFile path (needs NGC image or pip pkg)")
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--gen-tokens", type=int, default=100)
    parser.add_argument("--output", default="/tmp/results/results.jsonl")
    args = parser.parse_args()

    model_id, quant, dtype = MODELS[args.model]
    if args.model_id:
        model_id = args.model_id

    gpu = gpu_info()
    files = model_files(model_id)
    disk_bytes = sum(os.path.getsize(f) for f in files) if files else 0

    print(f"GPU    : {gpu['name']} | PCIe gen{gpu['pcie_gen']} x{gpu['pcie_width']}")
    print(f"Model  : {model_id} ({disk_bytes / 1e9:.1f} GB on disk, {len(files)} shards)")
    print(f"Mode   : {args.mode} | label: {args.label}")

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gpu": gpu["name"],
        "pcie": f"gen{gpu['pcie_gen']} x{gpu['pcie_width']}",
        "model_tier": args.model,
        "model_id": model_id,
        "quant": quant or "none",
        "mode": args.mode,
        "label": args.label,
        "enforce_eager": args.enforce_eager,
        "load_format": args.load_format,
        "model_disk_gb": round(disk_bytes / 1e9, 2),
    }

    if args.mode == "disk":
        if not files:
            raise SystemExit("No cached safetensors found — run a load first to download.")
        total, elapsed = bench_disk(files)
        result.update({
            "disk_read_s": round(elapsed, 2),
            "disk_read_gbps": round(total / elapsed / 1e9, 2),
        })
    else:
        metrics = bench_load(model_id, quant, dtype, args.enforce_eager,
                             args.load_format, args.prompt_len, args.gen_tokens)
        result.update(metrics)
        if disk_bytes:
            # The headline number for the GDS work: effective bytes/sec from
            # disk to engine-ready. Compare against disk_read_gbps and PCIe peak.
            result["effective_load_gbps"] = round(disk_bytes / metrics["llm_init_s"] / 1e9, 2)

    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
