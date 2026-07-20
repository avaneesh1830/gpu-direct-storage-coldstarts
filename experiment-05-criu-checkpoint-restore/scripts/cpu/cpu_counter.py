import time, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def rss_gb():
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith("VmRSS"):
                return int(l.split()[1]) / 1e6
    return 0.0

print(f"[{time.time():.1f}] loading {MODEL_ID} ...", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()
print(f"[{time.time():.1f}] LOADED in {time.time()-t0:.1f}s  rss={rss_gb():.2f}GB", flush=True)

prompt = "The capital of France is"
ids = tok(prompt, return_tensors="pt").input_ids
counter = 0.0

i = 0
while True:
    i += 1
    counter += 1.0
    line = f"tick={i} counter={int(counter)} rss={rss_gb():.2f}GB"
    if i % 5 == 0:
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=3, do_sample=False)
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().replace("\n"," ")
        line += f"  infer='{gen}'"
    print(line, flush=True)
    time.sleep(1)
