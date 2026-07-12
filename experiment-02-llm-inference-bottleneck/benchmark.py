import argparse, asyncio, json, time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List
import aiohttp, psutil

try:
    import GPUtil
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

BASE_URL = "http://localhost:8000"
HEADERS  = {"Content-Type": "application/json"}
TEST_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to merge two sorted lists.",
    "What are the main differences between supervised and unsupervised learning?",
    "Summarize the causes of World War I in 3 sentences.",
    "How does a transformer neural network work?",
]

@dataclass
class RequestResult:
    prompt: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0
    tpot_ms: float = 0.0
    tokens_per_sec: float = 0.0
    success: bool = True
    error: str = ""

@dataclass
class BenchmarkReport:
    model: str
    model_size: str
    timestamp: str
    server_startup_ms: float = 0.0
    gpu_vram_used_gb: float = 0.0
    gpu_vram_total_gb: float = 0.0
    gpu_utilization_pct: float = 0.0
    cpu_pct: float = 0.0
    system_ram_used_gb: float = 0.0
    results: List[RequestResult] = field(default_factory=list)
    mean_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    mean_latency_ms: float = 0.0
    mean_tokens_per_sec: float = 0.0
    total_throughput_tps: float = 0.0
    bottleneck: str = ""

def get_gpu_stats():
    if not GPU_AVAILABLE:
        return 0.0, 0.0, 0.0
    try:
        gpus = GPUtil.getGPUs()
        if not gpus: return 0.0, 0.0, 0.0
        g = gpus[0]
        return g.memoryUsed/1024, g.memoryTotal/1024, g.load*100
    except:
        return 0.0, 0.0, 0.0

def get_system_stats():
    return psutil.cpu_percent(interval=0.5), psutil.virtual_memory().used/(1024**3)

async def wait_for_server(timeout=60):
    print(f"\n[startup] Checking vLLM server...")
    start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        while time.perf_counter() - start < timeout:
            try:
                async with session.get(f"{BASE_URL}/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        elapsed = (time.perf_counter() - start)*1000
                        print(f"[startup] Server responded in {elapsed:.0f} ms")
                        return elapsed
            except:
                pass
            await asyncio.sleep(1)
    raise TimeoutError("Server not responding")

async def run_single_request(session, model, prompt):
    result = RequestResult(prompt=prompt[:60]+"...")
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "stream": True, "temperature": 0.0}
    t_start = time.perf_counter()
    t_first_token = None
    token_times = []
    output_text = ""
    try:
        async with session.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"): continue
                chunk = line[len("data:"):].strip()
                if chunk == "[DONE]": break
                try:
                    data = json.loads(chunk)
                    delta = data["choices"][0]["delta"].get("content","")
                    if delta:
                        now = time.perf_counter()
                        if t_first_token is None: t_first_token = now
                        else: token_times.append(now)
                        output_text += delta
                except: continue
        t_end = time.perf_counter()
        result.total_latency_ms = (t_end - t_start)*1000
        result.ttft_ms = (t_first_token - t_start)*1000 if t_first_token else result.total_latency_ms
        result.output_tokens = len(output_text.split())
        result.prompt_tokens = len(prompt.split())
        if t_first_token and len(token_times) > 1:
            result.tpot_ms = ((t_end - t_first_token)*1000)/len(token_times)
        if result.total_latency_ms > 0:
            result.tokens_per_sec = result.output_tokens/(result.total_latency_ms/1000)
    except Exception as e:
        result.success = False
        result.error = str(e)
    return result

async def run_benchmark(model, model_size):
    report = BenchmarkReport(model=model, model_size=model_size, timestamp=datetime.now().isoformat())
    report.server_startup_ms = await wait_for_server()
    vram_used, vram_total, gpu_util = get_gpu_stats()
    cpu_pct, ram_used = get_system_stats()
    report.gpu_vram_used_gb = round(vram_used,2)
    report.gpu_vram_total_gb = round(vram_total,2)
    report.gpu_utilization_pct = round(gpu_util,1)
    report.cpu_pct = round(cpu_pct,1)
    report.system_ram_used_gb = round(ram_used,2)
    print(f"[gpu]  VRAM: {vram_used:.1f}/{vram_total:.1f} GB  util: {gpu_util:.0f}%")
    print(f"[sys]  CPU: {cpu_pct:.0f}%  RAM: {ram_used:.1f} GB")
    print(f"\n[bench] Running {len(TEST_PROMPTS)} prompts...")
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[run_single_request(session, model, p) for p in TEST_PROMPTS])
    report.results = list(results)
    ok = [r for r in results if r.success]
    if ok:
        ttfts = sorted(r.ttft_ms for r in ok)
        tpots = [r.tpot_ms for r in ok if r.tpot_ms > 0]
        lats  = [r.total_latency_ms for r in ok]
        tps   = [r.tokens_per_sec for r in ok]
        def pct(lst,p): return lst[max(0,int(len(lst)*p/100)-1)]
        report.mean_ttft_ms = round(sum(ttfts)/len(ttfts),1)
        report.p50_ttft_ms  = round(pct(ttfts,50),1)
        report.p99_ttft_ms  = round(pct(ttfts,99),1)
        report.mean_tpot_ms = round(sum(tpots)/len(tpots),1) if tpots else 0
        report.mean_latency_ms = round(sum(lats)/len(lats),1)
        report.mean_tokens_per_sec = round(sum(tps)/len(tps),1)
        total_tok  = sum(r.output_tokens for r in ok)
        total_time = sum(r.total_latency_ms for r in ok)/1000
        report.total_throughput_tps = round(total_tok/total_time,1) if total_time > 0 else 0
        if report.mean_ttft_ms > 2000:   report.bottleneck = "HIGH TTFT — prefill slow."
        elif report.mean_tpot_ms > 100:  report.bottleneck = "HIGH TPOT — decode slow, bandwidth saturated."
        elif report.gpu_utilization_pct < 30 and report.gpu_vram_used_gb > 0: report.bottleneck = "LOW GPU UTIL — CPU is bottleneck."
        elif report.gpu_vram_used_gb/max(report.gpu_vram_total_gb,1) > 0.95:  report.bottleneck = "VRAM NEAR FULL — reduce max-model-len."
        else: report.bottleneck = "No obvious bottleneck. System looks healthy."
    return report

def print_report(r):
    sep = "="*55
    print(f"\n{sep}\n  BENCHMARK REPORT — {r.model_size} model\n{sep}")
    print(f"  Model        : {r.model}")
    print(f"  Timestamp    : {r.timestamp}")
    print(f"\n--- Infrastructure ---")
    print(f"  Server ready : {r.server_startup_ms:.0f} ms")
    print(f"  GPU VRAM     : {r.gpu_vram_used_gb:.1f} / {r.gpu_vram_total_gb:.1f} GB")
    print(f"  GPU util     : {r.gpu_utilization_pct:.0f}%")
    print(f"  CPU usage    : {r.cpu_pct:.0f}%")
    print(f"  System RAM   : {r.system_ram_used_gb:.1f} GB")
    print(f"\n--- Latency ---")
    print(f"  Mean TTFT    : {r.mean_ttft_ms:.1f} ms  <- time to first token")
    print(f"  P50  TTFT    : {r.p50_ttft_ms:.1f} ms")
    print(f"  P99  TTFT    : {r.p99_ttft_ms:.1f} ms")
    print(f"  Mean TPOT    : {r.mean_tpot_ms:.1f} ms  <- ms per output token")
    print(f"  Mean E2E     : {r.mean_latency_ms:.1f} ms")
    print(f"\n--- Throughput ---")
    print(f"  tok/s/req    : {r.mean_tokens_per_sec:.1f}")
    print(f"  Total tok/s  : {r.total_throughput_tps:.1f}")
    print(f"\n--- Per request ---")
    for i,res in enumerate(r.results):
        s = "OK" if res.success else "FAIL"
        print(f"  [{s}] #{i+1} | TTFT {res.ttft_ms:6.0f}ms | TPOT {res.tpot_ms:5.1f}ms | {res.tokens_per_sec:5.1f} tok/s")
        if not res.success: print(f"       ERROR: {res.error}")
    print(f"\n--- Bottleneck ---")
    print(f"  {r.bottleneck}")
    print(sep)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",  default="1B", choices=["1B","10B","30B","110B"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--out",   default="benchmark_results.json")
    args = parser.parse_args()
    model = args.model or await get_model_name()
    print(f"[info] Model: {model}  size: {args.size}")
    report = await run_benchmark(model=model, model_size=args.size)
    print_report(report)
    with open(args.out,"w") as f:
        json.dump(asdict(report),f,indent=2)
    print(f"\n[saved] {args.out}")

if __name__ == "__main__":
    asyncio.run(main())
