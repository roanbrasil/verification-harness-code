"""What verification costs versus generation.

Phase A: generate N_RUNS solutions per task (same prompt, different seeds) and measure tokens and time.
Phase B: run each output through the verification layers and measure the cost of each one.
Phase C: LLM judge (the same model) on each output, direct verdict.

Everything goes to results.jsonl (one line per output); the script resumes where it stopped.
"""
import json, os, re, sys, time
from llama_cpp import Llama
from common import preamble, extract_code, static_check, sandbox

MODEL = "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
N_RUNS = 5
GEN = dict(temperature=0.7, top_p=0.8, top_k=20, max_tokens=320)
OUT = "results.jsonl"

GEN_SYS = ("You are a Python programmer. Reply with the complete function in a single ```python code block. "
           "No explanation.")
tasks = json.load(open("tasks.json"))
rows = {}
if os.path.exists(OUT):
    for ln in open(OUT):
        r = json.loads(ln)
        rows[(r["task_id"], r["run"])] = r


def save():
    with open(OUT + ".tmp", "w") as f:
        for k in sorted(rows, key=lambda k: (k[1], int(k[0].split("/")[1]))):
            f.write(json.dumps(rows[k]) + "\n")
    os.replace(OUT + ".tmp", OUT)


llm = Llama(MODEL, n_ctx=2048, n_threads=2, verbose=False)

# ---------- Phase A: generation (outer loop over runs, so the prefix cache is not reused) ----------
for run in range(N_RUNS):
    for ti, t in enumerate(tasks):
        key = (t["task_id"], run)
        if key in rows:
            continue
        msgs = [{"role": "system", "content": GEN_SYS},
                {"role": "user", "content": "Complete this function:\n\n" + t["prompt"]}]
        t0 = time.perf_counter()
        o = llm.create_chat_completion(msgs, seed=1000 * run + ti, **GEN)
        dt = time.perf_counter() - t0
        rows[key] = {"task_id": t["task_id"], "run": run, "seed": 1000 * run + ti,
                     "raw": o["choices"][0]["message"]["content"],
                     "finish": o["choices"][0]["finish_reason"],
                     "gen_prompt_tok": o["usage"]["prompt_tokens"],
                     "gen_completion_tok": o["usage"]["completion_tokens"], "gen_s": dt}
        save()
        print(f"[gen] run {run} {t['task_id']} {o['usage']['completion_tokens']} tok {dt:.1f}s", flush=True)

# ---------- Phase B: deterministic layers + ground truth ----------
by_id = {t["task_id"]: t for t in tasks}
for key, r in rows.items():
    if "hidden_ok" in r:
        continue
    t = by_id[r["task_id"]]
    code = extract_code(r["raw"])
    program = preamble(t["prompt"], t["entry_point"]) + "\n\n" + code
    r["code"] = code
    r["static_ok"], r["static_s"] = static_check(program, t["entry_point"])
    base = {"program": program, "entry_point": t["entry_point"]}
    ex, r["examples_s"] = sandbox({**base, "stage": "examples", "spec": t["spec_doc"]})
    r["examples_ok"] = bool(ex.get("ok"))
    pr, r["probes_s"] = sandbox({**base, "stage": "probes", "test": t["test"], "probes": t["probes"]})
    r["signature"] = pr.get("signature") if pr.get("ok") else ["INVALID"]
    hid, r["hidden_s"] = sandbox({**base, "stage": "hidden", "test": t["test"]})
    r["hidden_ok"] = bool(hid.get("ok"))
    r["hidden_err"] = hid.get("error")
save()
print("[det] deterministic layers done", flush=True)

# ---------- Phase C: LLM judge ----------
JUDGE_DIRECT_SYS = ("You are a strict code reviewer. You receive a function specification and a candidate "
                    "implementation. Decide whether the implementation is correct for ALL inputs the specification "
                    "allows, including edge cases. Reply with exactly one line: 'VERDICT: PASS' or 'VERDICT: FAIL'. "
                    "Nothing else.")
VARIANTS = [("judge_d", JUDGE_DIRECT_SYS, 8)]
order = sorted(rows, key=lambda k: (k[1], int(k[0].split("/")[1])))
for tag, sysmsg, max_tok in VARIANTS:
    for key in order:
        r = rows[key]
        if tag + "_raw" in r:
            continue
        t = by_id[r["task_id"]]
        user = ("SPECIFICATION:\n" + t["prompt"].strip() + "\n\nCANDIDATE IMPLEMENTATION:\n```python\n" +
                r["code"].strip() + "\n```")
        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]
        t0 = time.perf_counter()
        o = llm.create_chat_completion(msgs, temperature=0.0, max_tokens=max_tok, seed=7)
        dt = time.perf_counter() - t0
        txt = o["choices"][0]["message"]["content"]
        m = re.findall(r"VERDICT:\s*(PASS|FAIL)", txt)
        r[tag + "_raw"] = txt
        r[tag + "_verdict"] = m[-1] if m else "NONE"
        r[tag + "_prompt_tok"] = o["usage"]["prompt_tokens"]
        r[tag + "_completion_tok"] = o["usage"]["completion_tokens"]
        r[tag + "_s"] = dt
        save()
        print(f"[{tag}] run {key[1]} {key[0]} {r[tag + '_verdict']} truth={r['hidden_ok']} "
              f"{o['usage']['completion_tokens']} tok {dt:.1f}s", flush=True)
print("DONE", flush=True)
