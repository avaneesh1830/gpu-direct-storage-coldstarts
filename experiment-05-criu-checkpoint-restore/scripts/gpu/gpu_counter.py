import time, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

def vram_gb():
    return torch.cuda.memory_allocated() / 1e9

print(f"[{time.time():.1f}] loading {MODEL_ID} on GPU ...", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda()
model.eval()
torch.cuda.synchronize()
print(f"[{time.time():.1f}] LOADED in {time.time()-t0:.1f}s  vram={vram_gb():.2f}GB", flush=True)

gpu_c = torch.zeros(1, device="cuda")
prompt = "The capital of France is"
ids = tok(prompt, return_tensors="pt").input_ids.cuda()

i = 0
while True:
    i += 1
    gpu_c += 1  # real GPU op each tick
    line = f"tick={i} gpu_counter={int(gpu_c.item())} vram={vram_gb():.2f}GB"
    if i % 5 == 0:
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=3, do_sample=False)
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().replace("\n"," ")
        line += f"  infer='{gen}'"
    print(line, flush=True)
    time.sleep(1)
