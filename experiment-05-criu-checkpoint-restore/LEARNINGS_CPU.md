# Experiment 5 — CRIU Checkpoint/Restore: Complete Learnings

**What this is:** a full teaching walkthrough of Stage 1 (CPU) — every script, every
flag, why each choice was made, every issue hit, and the transferable insights.
Companion to [RESULTS.md](RESULTS.md) (the numbers) and [README.md](README.md) (the recipe).

---

## 0. The big picture — what CRIU does and why we care

CRIU = **C**heckpoint/**R**estore **I**n **U**serspace. A running program is just *state*:
the contents of its memory (RAM), its CPU registers, and its open files/sockets. CRIU
freezes a running process, copies all of that state into files on disk (a "checkpoint"),
and can later rebuild an identical process from those files (a "restore") — even after
the original was killed or the machine rebooted.

**Why it matters to this project:** experiments 2–4 showed that starting a model server
has a fixed ~33 s "engine-init" cost that no loader or faster storage could remove. CRIU
lets you pay that cost **once**, snapshot the ready process, then "start" future copies by
*restoring the snapshot* instead of re-initializing. This is the mechanism NVIDIA
productized as Dynamo Snapshot and Doubleword demonstrated in Cloudburst.

Everything in Stage 1 was to prove that mechanism, from the simplest possible case
(a counter) up to a real model resident in RAM.

---

## 1. How CRIU works under the hood (the mental model)

When you run `criu dump -t <PID>`:

1. **Seize** — CRIU uses `ptrace` (the same kernel API debuggers use) to freeze the target
   process and all its threads mid-execution.
2. **Inject parasite code** — it injects a tiny bit of code into the frozen process's own
   address space so it can read that process's memory and kernel state from the inside.
3. **Collect state** — it walks `/proc/<pid>/` to enumerate memory maps, open file
   descriptors, sockets, registers, credentials, etc.
4. **Write images** — it serializes everything into a set of `.img` files in the images
   directory. The big one is `pages-*.img` (the actual memory contents).
5. **Kill (by default)** — the frozen process is terminated. The checkpoint on disk is now
   the only copy of that process's state.

On `criu restore`:

1. CRIU forks a new process, restores its PID, memory maps, and file descriptors, copies
   the saved memory pages back in, reloads the CPU registers, and resumes execution at the
   exact instruction where it was frozen. With `-d` it then detaches and lets the process
   run on its own.

**The key intuition:** restore is not "run the program again." It's "reconstruct a memory
image from disk." That single fact explains every performance result below.

### What's in the images directory (`ckpt/`)
| File | Holds |
|---|---|
| `pages-*.img` | the process's memory contents (the bulk — 2.2 GB for the model) |
| `pagemap-*.img` | which virtual addresses map to which saved pages |
| `core-*.img` | CPU registers, thread state, FPU/AVX state |
| `mm-*.img` | memory-mapping layout (what's mapped where) |
| `files.img` / `fdinfo-*.img` | open file descriptors and their positions |
| `fs-*.img` | working directory, root, umask |
| `pstree.img` | the process/thread tree structure |
| `inventory.img` | metadata/version; restore reads this first (an empty one = failed dump) |

---

## 2. Every script, line by line

### `counter.py` — the toy workload
```python
import time
i = 0
while True:
    i += 1
    print(f"count = {i}", flush=True)
    time.sleep(1)
```
- `i = 0` — the state we want to survive death. If restore works, `i` returns at its frozen value.
- `while True` — runs forever so we can freeze it at an arbitrary moment.
- `print(..., flush=True)` — **flush is critical.** Python buffers stdout; without flushing,
  lines sit in memory and you can't tell in real time what count it's on. Flushing writes
  each line to the log immediately, so the log is a faithful record of the internal state.
- `time.sleep(1)` — one tick/second: slow enough to watch and to checkpoint at a known value.

**Why so trivial?** If checkpoint/restore can't preserve a single integer, nothing bigger
matters. Prove the mechanism at the smallest scale first.

### `start.sh` — launching it *checkpointably*
```bash
setsid python3 counter.py < /dev/null &> counter.log &
echo "counter PID: $!"
```
- `setsid` — **the most important word here.** Starts the program in a new *session* with
  **no controlling terminal**. CRIU refuses to dump a TTY-attached process unless you add
  `--shell-job`, because a terminal is shared kernel state it can't safely snapshot.
  `setsid` avoids that entire class of failure.
- `< /dev/null` — detaches stdin (nothing to read from a terminal).
- `&> counter.log` — redirects stdout+stderr into a file. The process's output is now a
  plain file CRIU can snapshot cleanly, and it's our evidence log.
- `&` — background it. `$!` = PID of the last backgrounded process, which CRIU needs.

### The checkpoint command
```bash
criu dump -t "$PID" -D ckpt -v4 -o dump.log
```
- `dump` — the checkpoint subcommand.
- `-t $PID` (`--tree`) — target root process; CRIU walks its whole process/thread tree.
- `-D ckpt` (`--images-dir`) — where to write the `.img` files.
- `-v4` — verbosity level 4 (max). Overkill when it works, essential for debugging — this
  is how we *found* the TCP-socket error.
- `-o dump.log` — log filename, written **relative to `-D`** (so it landed in `ckpt/dump.log`,
  a small surprise the first time).
- **Default behavior: it KILLS the process after snapshotting.** Intentional — the process
  must be genuinely dead so a successful restore proves resurrection.

### Dropping the page cache (before restore)
```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```
- `sync` — flush pending disk writes so the cache-drop is clean.
- `echo 3 > /proc/sys/vm/drop_caches` — tell the kernel to empty its page cache (the RAM
  copy of recently-read files). This forces restore to read the image **from disk, cold** —
  the real serverless condition. Without it, restore reads the image out of RAM and gives a
  misleadingly fast number. (This is the same "manufacture a cold start" trick from exp 3–4.)

### The restore command
```bash
criu restore -D ckpt -d -v4 -o restore.log
```
- `restore` — the restore subcommand.
- `-D ckpt` — read images from here.
- `-d` (`--restore-detached`) — rebuild the process, then CRIU exits, leaving the restored
  process running independently (re-parented to init). Without `-d`, CRIU stays attached as
  the parent and blocks.
- The restored process gets the **same PID**, reopens its log at the same position, and
  continues from the exact instruction it was frozen on.
- **Pass condition:** log shows `count = N` then `count = N+1` after a real time gap.

### `cpu_counter.py` — adding realism (a real model in RAM)
```python
def rss_gb():
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith("VmRSS"):
                return int(l.split()[1]) / 1e6
```
- We swapped the toy integer for **actual model weights held in RAM** — the realism jump.
  Now the state CRIU must preserve is ~2.4 GB of weights, not one number.
- `rss_gb()` reads `VmRSS` (Resident Set Size = physical RAM the process actually uses) from
  `/proc/self/status`, so we can *see* the weights are resident and confirm the same amount
  returns after restore (proof they came back, not reloaded).

```python
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()
```
- `torch.float32` — deliberate: on CPU, float16 is slow and some ops are unsupported, and
  the box had only 7 GB RAM. float32 for a 0.5B model ≈ 2 GB, which fits. (GPU uses float16.)
- `model.eval()` — inference mode (no dropout etc.).

```python
prompt = "The capital of France is"
ids = tok(prompt, return_tensors="pt").input_ids
...
if i % 5 == 0:
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=3, do_sample=False)
```
- The fixed prompt is a **weight-integrity test**: if after restore the model still answers
  "Paris," all 2.2 GB of weights came back byte-correct. Garbled output would mean corruption.
- `torch.no_grad()` — no gradient tracking (forward-only), saves memory/time.
- `max_new_tokens=3, do_sample=False` — short, greedy, deterministic → identical every time,
  so "still says Paris" is an unambiguous check.

This one process demonstrates **three things surviving a checkpoint at once**: a counter
(live compute state), RSS (weights resident), and inference output (weights correct).

### `start_model.sh` — the offline fix baked in
```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
setsid ~/venv/bin/python cpu_counter.py < /dev/null &> model.log &
```
- Same `setsid … < /dev/null &> log &` detachment pattern.
- The two `export`s force HuggingFace to load the **cached** model without touching the
  network — preventing the live TCP socket CRIU couldn't dump (see Issue 5). Also the honest
  serverless config: weights pre-baked, load offline.
- `~/venv/bin/python` — a virtualenv, because Ubuntu 24.04 blocks system-wide pip
  (PEP-668 "externally managed environment").

---

## 3. The issues we hit — and what each taught

Five distinct problems; each is a real lesson about CRIU and the Linux/ML stack.

**1. `criu` not installable on Ubuntu 24.04.** `apt install criu` → "no installation
candidate." The package **failed to build** on 24.04's toolchain and was dropped from the
repos (Launchpad bug #2066148) — not our error. **Fix:** the CRIU team PPA (`ppa:criu/ppa`).
**Lesson:** "package missing" can mean the distro dropped it, not that you misconfigured
apt — check upstream before fighting the sources list.

**2. `criu check` "failed."** It printed `CRIU needs CAP_SYS_ADMIN`. Not a failure — CRIU
manipulates other processes' memory and PIDs, which requires **root**. **Fix:** `sudo criu
check`. **Lesson:** checkpoint/restore is inherently privileged.

**3. Root-owned files broke re-runs.** After `sudo` runs, `ckpt/` and the timing files were
root-owned, so the next run's `rm`/redirects failed with "Permission denied" and the script
silently ran on stale state (the bogus 711/712 result). **Fix:** `sudo rm`/`chown` before
re-running. **Lesson:** mixing sudo and non-sudo in one directory leaves ownership landmines;
clean deliberately between runs. (This was mistake #4 from the earlier SESSION_LOG, recurring.)

**4. A PID collision produced a fake "restore."** `pkill` raced; the old process was still
alive holding its PID, so `criu restore` failed instantly (exit 1) while the untouched
original kept counting — and `/usr/bin/time` clocked the *fast failure* as if it were a fast
restore. **Fix:** verify the process is actually dead (`pgrep` empty) *before* restoring, and
check the restore **exit code**. **Lesson:** confirm the precondition (dead) and postcondition
(exit 0), or you measure the wrong thing.

**5. The live TCP socket (the most instructive).** The model dump failed with
`inet: Connected TCP socket`. Cause: `huggingface_hub` keeps a **keep-alive HTTPS connection**
to the Hub in its connection pool after downloading. CRIU won't snapshot a live TCP
connection by default, because the remote end won't exist after restore. **Fix:** load the
cached model with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` so no socket is opened.
**Lesson (a keeper):** CRIU can only checkpoint state it fully controls — memory and files,
yes; a live connection to a remote server, no. **A checkpointable process must be "closed":
no live external connections.** This matters enormously for vLLM, which opens sockets everywhere.

---

## 4. What was achieved

- Proved CRIU checkpoint/restore end-to-end on Linux: a killed process resurrected to the
  exact instruction it died on.
- Scaled it from a 4 MB toy to a **2.2 GB real-model process**, verifying not just resumption
  but that the **weights returned byte-correct** (still answers "Paris"), with no disk reload.
- Produced clean, honest timings under true cold-cache conditions (see RESULTS.md).

## 5. What was learned (transferable insights)

**(a) CRIU restore is disk-bound.** The image is a memory snapshot; restoring is a sequential
disk read. The 2.2 GB image restored at 0.12 GB/s — the *exact* virtio speed from exp 3–4.
This phase plugs straight into the existing storage analysis: restore time ≈ image size ÷
disk speed.

**(b) The decision rule for whether snapshotting is worth it:**
> snapshot wins  ⟺  (initialization time skipped) − (image read time) > 0

For 0.5B on slow disk this is **negative** — a fresh warm load (~2–3 s) beats an 18.6 s image
read, because 0.5B has almost no init to skip. This doesn't undercut the thesis; it *defines*
when the technique pays off: **large init cost + fast storage** — precisely the 110B vLLM /
GPU case with the ~33 s engine floor.

**(c) A checkpointable process must be self-contained** — no live TCP connections, no terminal
attachment, no ownership surprises. These constraints are exactly why productizing this
(Dynamo Snapshot, Cloudburst) is real engineering, not one `criu` command.

**(d) CRIU vs fast loaders are different levers.** InstantTensor (exp 3–4) attacks the
*weight-load* portion of a cold start. CRIU attacks the *whole thing including engine init* —
by not doing init at all. They're complementary: a loader speeds the one-time snapshot
creation; CRIU removes init from every subsequent start.

---

## 6. Why the GPU stage is harder (the on-ramp to Stage 2)

A normal process image contains its RAM. A GPU process holds state CRIU **cannot** see or
snapshot on its own:
- **VRAM** — the model weights live in GPU memory, not in `/proc/<pid>/` RAM maps.
- **Open device handles** — `/dev/nvidia*` file descriptors the CUDA driver owns.

If you point plain CRIU at a CUDA process, it chokes on those device FDs (the GPU analog of
the TCP-socket problem). NVIDIA's **`cuda-checkpoint`** tool solves it: before `criu dump`, it
**locks** the process's CUDA calls, **copies VRAM down to host RAM**, and **releases the GPU
handles** — turning the process into an ordinary CRIU-dumpable one whose (now larger) RAM
image includes the former GPU state. After `criu restore`, it re-acquires the GPU, copies the
state back up to VRAM, and **unlocks**. Recent CRIU ships a CUDA plugin that calls
`cuda-checkpoint` automatically during dump/restore.

**The gate:** `cuda-checkpoint` needs a recent driver (r550+, ideally r570+). That's the first
thing to verify on any GPU box (`nvidia-smi`) — it's the GPU-stage equivalent of "is criu even
installable."

**Why the GPU numbers should finally go positive:** on an H100 box the two terms in the
decision rule both flip favorably — fast local NVMe makes the image read cheap, and there's a
real engine-init floor to skip. That's the head-to-head we want: **restore time vs. a fresh
cold start**, measured on 10B and 30B.

---

## 7. Reproducibility checklist (do this every run)

1. Launch the target detached: `setsid … < /dev/null &> log &` (no TTY).
2. For model workloads: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` (no live sockets).
3. Note the last log line before dump (the known state to resume from).
4. `sudo criu dump …` → confirm process is **dead** (`pgrep` empty) and exit 0.
5. `sync; echo 3 > /proc/sys/vm/drop_caches` (true cold restore).
6. `sudo criu restore -d …` → confirm exit 0.
7. Verify: log resumes at **N+1**, RSS/VRAM matches pre-dump, inference output unchanged.
8. Record: dump time, restore time, image size (`du -sh`), and the resume excerpt.
9. `sudo chown -R $USER ckpt*` before the next run (avoid the root-owned-file trap).

---

## 8. Glossary

- **Checkpoint / dump** — freezing a process's state to files on disk.
- **Restore** — rebuilding a live process from those files.
- **Image / images directory** — the set of `.img` files a checkpoint produces.
- **RSS (Resident Set Size)** — physical RAM a process currently occupies; our proxy for
  "weights are resident."
- **Page cache** — the kernel's RAM copy of recently-read files; dropping it forces a true
  cold (disk) read.
- **ptrace** — the kernel API (used by debuggers) CRIU uses to freeze and inspect a process.
- **Cold vs warm restore** — cold = image read from disk (cache dropped), the serverless
  reality; warm = image served from RAM.
- **cuda-checkpoint** — NVIDIA tool that drains VRAM to host RAM and releases GPU handles so a
  CUDA process becomes CRIU-dumpable; reverses on restore.
- **Engine-init floor** — the ~33 s fixed startup cost (torch.compile, CUDA graphs, NCCL,
  worker spawn) from exp 2–4 that CRIU restore skips by not re-initializing.
