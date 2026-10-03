"""Read results.jsonl and compute every number the article cites. Output: stats.json + a terminal summary."""
import json, collections, re, statistics as st

rows = [json.loads(l) for l in open("results.jsonl")]


def verdict(raw, strict):
    """Verdict from the raw text. The direct judge sometimes answers just 'PASS'; the parser accepts that."""
    pat = r"VERDICT:\s*\**\s*(PASS|FAIL)" if strict else r"\b(PASS|FAIL)\b"
    m = re.findall(pat, raw)
    return m[-1] if m else "NONE"


for r in rows:
    r["jd"] = verdict(r["judge_d_raw"], strict=False)

tasks = json.load(open("tasks.json"))
N = len(rows)
T = len(tasks)
RUNS = sorted({r["run"] for r in rows})
K = len(RUNS)
by_task = collections.defaultdict(list)
for r in rows:
    by_task[r["task_id"]].append(r)
for v in by_task.values():
    v.sort(key=lambda r: r["run"])

S = {"n_outputs": N, "n_tasks": T, "n_runs": K}

# ---------- generation ----------
gen_tok = [r["gen_prompt_tok"] + r["gen_completion_tok"] for r in rows]
S["gen"] = {
    "prompt_tok_mean": st.mean(r["gen_prompt_tok"] for r in rows),
    "completion_tok_mean": st.mean(r["gen_completion_tok"] for r in rows),
    "total_tok_mean": st.mean(gen_tok),
    "s_mean": st.mean(r["gen_s"] for r in rows),
    "s_median": st.median(r["gen_s"] for r in rows),
    "s_total": sum(r["gen_s"] for r in rows),
    "truncated": sum(r["finish"] == "length" for r in rows),
}
G_S = S["gen"]["s_mean"]
G_TOK = S["gen"]["total_tok_mean"]

# ---------- ground truth and consistency ----------
defect = lambda r: not r["hidden_ok"]
n_def = sum(defect(r) for r in rows)
passes = {t: sum(r["hidden_ok"] for r in v) for t, v in by_task.items()}
hist = collections.Counter(passes.values())
per_run = [st.mean(r["hidden_ok"] for r in rows if r["run"] == k) for k in RUNS]
S["truth"] = {
    "defects": n_def, "correct": N - n_def,
    "mean_at_k": (N - n_def) / N,
    "pass_hat_k": sum(v == K for v in passes.values()) / T,
    "tasks_all_pass": sum(v == K for v in passes.values()),
    "tasks_all_fail": sum(v == 0 for v in passes.values()),
    "tasks_mixed": sum(0 < v < K for v in passes.values()),
    "hist_k": {str(k): hist.get(k, 0) for k in range(K + 1)},
    "per_run_pass": per_run,
    "per_run_min": min(per_run), "per_run_max": max(per_run),
}
S["truth"]["gap_pp"] = 100 * (S["truth"]["mean_at_k"] - S["truth"]["pass_hat_k"])
S["truth"]["normalized_consistency"] = S["truth"]["pass_hat_k"] / S["truth"]["mean_at_k"]
S["truth"]["mixed_task_ids"] = {t: passes[t] for t in passes if 0 < passes[t] < K}

# ---------- layer 4: agreement across runs ----------
def agree_flags(runs_used, need):
    """flag[(task, run)] = True if the output is NOT in the majority cluster (>= need) of the runs used."""
    flags = {}
    for t, v in by_task.items():
        sub = [r for r in v if r["run"] in runs_used]
        sigs = [tuple(r["signature"]) for r in sub]
        cnt = collections.Counter(s for s in sigs if s != ("INVALID",))
        top, size = (cnt.most_common(1)[0] if cnt else (None, 0))
        unique_top = size >= need and list(cnt.values()).count(size) == 1
        for r, s in zip(sub, sigs):
            flags[(t, r["run"])] = not (unique_top and s == top)
    return flags


agree5 = agree_flags(set(RUNS), 3)
agree3 = agree_flags({0, 1, 2}, 2)

# ---------- layers in isolation ----------
def layer(name, flagged, cost_s, cost_tok, subset=None):
    sub = [r for r in rows if subset is None or (r["task_id"], r["run"]) in subset]
    fl = [r for r in sub if flagged(r)]
    d = sum(defect(r) for r in sub)
    caught = sum(defect(r) for r in fl)
    false_rej = len(fl) - caught
    return {"name": name, "n": len(sub), "defects": d, "flagged": len(fl), "caught": caught,
            "recall": caught / d if d else None, "false_rejections": false_rej,
            "false_rejection_rate": false_rej / (len(sub) - d) if len(sub) - d else None,
            "precision": caught / len(fl) if fl else None, "escaped": d - caught,
            "cost_s": cost_s, "cost_tok": cost_tok, "cost_s_pct_gen": 100 * cost_s / G_S,
            "cost_tok_pct_gen": 100 * cost_tok / G_TOK}


probes_s = st.mean(r["probes_s"] for r in rows)
jd_tok = st.mean(r["judge_d_prompt_tok"] + r["judge_d_completion_tok"] for r in rows)
jd_s = st.mean(r["judge_d_s"] for r in rows)
L = {
    "static": layer("static", lambda r: not r["static_ok"], st.mean(r["static_s"] for r in rows), 0),
    "examples": layer("examples", lambda r: not r["examples_ok"], st.mean(r["examples_s"] for r in rows), 0),
    "judge_d": layer("judge_d", lambda r: r["jd"] != "PASS", jd_s, jd_tok),
    "agree3": layer("agree3", lambda r: agree3[(r["task_id"], r["run"])], 2 * G_S + 3 * probes_s, 2 * G_TOK,
                    subset=set(agree3)),
    "agree5": layer("agree5", lambda r: agree5[(r["task_id"], r["run"])], 4 * G_S + 5 * probes_s, 4 * G_TOK),
}
S["layers"] = L
def judge_stats(tag, key):
    return {
        "prompt_tok_mean": st.mean(r[tag + "_prompt_tok"] for r in rows),
        "completion_tok_mean": st.mean(r[tag + "_completion_tok"] for r in rows),
        "truncated": sum(r[key] == "NONE" for r in rows),
        "verdicts": dict(collections.Counter(r[key] for r in rows)),
        "pass_on_defect": sum(r[key] == "PASS" and defect(r) for r in rows),
        "fail_on_defect": sum(r[key] != "PASS" and defect(r) for r in rows),
        "fail_on_correct": sum(r[key] != "PASS" and not defect(r) for r in rows),
        "pass_on_correct": sum(r[key] == "PASS" and not defect(r) for r in rows),
        "accuracy": st.mean((r[key] == "PASS") == r["hidden_ok"] for r in rows),
    }


S["judge_d"] = judge_stats("judge_d", "jd")
S["always_pass_accuracy"] = (N - n_def) / N
S["probes_s_mean"] = probes_s
S["hidden_s_mean"] = st.mean(r["hidden_s"] for r in rows)

# ---------- cascade: each layer only runs on what the previous one let through ----------
def agree_cost(r):
    """Measured cost of verifying r by agreement: the 4 sibling generations + the probes of all 5 runs."""
    sib = by_task[r["task_id"]]
    s = sum(x["gen_s"] for x in sib if x["run"] != r["run"]) + sum(x["probes_s"] for x in sib)
    tok = sum(x["gen_prompt_tok"] + x["gen_completion_tok"] for x in sib if x["run"] != r["run"])
    return s, tok


COST = {"static": lambda r: (r["static_s"], 0), "examples": lambda r: (r["examples_s"], 0),
        "judge_d": lambda r: (r["judge_d_s"], r["judge_d_prompt_tok"] + r["judge_d_completion_tok"]),
        "agree5": agree_cost}


def cascade(judge_name, judge_key):
    order = [("static", lambda r: not r["static_ok"]), ("examples", lambda r: not r["examples_ok"]),
             (judge_name, lambda r: r[judge_key] != "PASS"),
             ("agree5", lambda r: agree5[(r["task_id"], r["run"])])]
    alive = list(rows)
    casc = []
    cum_s = cum_tok = 0.0
    cum_caught = cum_false = 0
    for name, flagged in order:
        n_in = len(alive)
        fl = [r for r in alive if flagged(r)]
        caught = sum(defect(r) for r in fl)
        false_rej = len(fl) - caught
        layer_s = sum(COST[name](r)[0] for r in alive)     # measured cost over what reached this layer
        layer_tok = sum(COST[name](r)[1] for r in alive)
        cum_s += layer_s
        cum_tok += layer_tok
        cum_caught += caught
        cum_false += false_rej
        alive = [r for r in alive if not flagged(r)]
        casc.append({"name": name, "n_in": n_in, "caught": caught, "false_rejections": false_rej,
                     "layer_s_total": layer_s, "layer_tok_total": layer_tok,
                     "s_per_defect": layer_s / caught if caught else None,
                     "tok_per_defect": layer_tok / caught if caught else None,
                     "cum_caught": cum_caught, "cum_false": cum_false,
                     "cum_s_per_output": cum_s / N, "cum_tok_per_output": cum_tok / N,
                     "cum_s_pct_gen": 100 * (cum_s / N) / G_S, "cum_tok_pct_gen": 100 * (cum_tok / N) / G_TOK,
                     "escaped_after": n_def - cum_caught, "accepted_after": len(alive),
                     "residual_defect_rate": sum(defect(r) for r in alive) / len(alive) if alive else None})
    return casc, alive


casc, alive = cascade("judge_d", "jd")

# variant without the judge: examples -> agreement
al = [r for r in rows if r["static_ok"] and r["examples_ok"]]
fl = [r for r in al if agree5[(r["task_id"], r["run"])]]
S["cascade_no_judge"] = {
    "n_in_agree": len(al), "caught_agree": sum(defect(r) for r in fl),
    "false_agree": sum(not defect(r) for r in fl),
    "escaped": sum(defect(r) for r in al) - sum(defect(r) for r in fl),
    "cum_s_per_output": (L["static"]["cost_s"] * N + L["examples"]["cost_s"] * N + sum(agree_cost(r)[0] for r in al)) / N,
}
S["cascade_no_judge"]["cum_s_pct_gen"] = 100 * S["cascade_no_judge"]["cum_s_per_output"] / G_S
S["judge_waste"] = {"judge_s_total": casc[2]["layer_s_total"], "false_rejections": casc[2]["false_rejections"],
                    "regen_s": casc[2]["false_rejections"] * G_S}
S["timing"] = {k: {"mean": st.mean(r[k] for r in rows), "median": st.median(r[k] for r in rows),
                   "max": max(r[k] for r in rows)} for k in ["static_s", "examples_s", "probes_s", "judge_d_s", "gen_s"]}
S["price_equiv"] = {str(ratio): {
    "gen": S["gen"]["prompt_tok_mean"] + ratio * S["gen"]["completion_tok_mean"],
    "judge": S["judge_d"]["prompt_tok_mean"] + ratio * S["judge_d"]["completion_tok_mean"]} for ratio in (1, 4, 5)}
for v in S["price_equiv"].values():
    v["judge_pct_gen"] = 100 * v["judge"] / v["gen"]
S["cascade"] = casc

# cheap cascade: only the two deterministic layers
S["cheap_cascade"] = {"caught": casc[1]["cum_caught"], "share_of_defects": casc[1]["cum_caught"] / n_def if n_def else None,
                      "cost_s": casc[1]["cum_s_per_output"], "pct_gen": casc[1]["cum_s_pct_gen"]}

# ---------- consistency is not correctness ----------
unan = [t for t, v in by_task.items() if len({tuple(r["signature"]) for r in v}) == 1]
S["agreement"] = {
    "tasks_unanimous": len(unan),
    "tasks_unanimous_correct": sum(passes[t] == K for t in unan),
    "tasks_unanimous_wrong": sum(passes[t] == 0 for t in unan),
    "tasks_unanimous_wrong_ids": [t for t in unan if passes[t] == 0],
    "distinct_signatures_mean": st.mean(len({tuple(r["signature"]) for r in v}) for v in by_task.values()),
}
# defects that get through ALL layers
esc = [r for r in alive if defect(r)]
S["escapes"] = {"n": len(esc), "tasks": dict(collections.Counter(r["task_id"] for r in esc))}

json.dump(S, open("stats.json", "w"), indent=1, default=str)

print(f"outputs {N} | tasks {T} | runs {K}")
g = S["gen"]
print(f"generation: {g['prompt_tok_mean']:.0f}+{g['completion_tok_mean']:.0f} tok, {g['s_mean']:.2f}s mean, truncated {g['truncated']}")
tr = S["truth"]
print(f"defects {tr['defects']}/{N} | Mean@{K} {tr['mean_at_k']:.3f} | Pass^{K} {tr['pass_hat_k']:.3f} | gap {tr['gap_pp']:.1f}pp | hist {tr['hist_k']}")
print(f"per run: {[round(x, 3) for x in per_run]}")
for k, v in L.items():
    print(f"{k:9s} n={v['n']:3d} flag={v['flagged']:3d} caught={v['caught']:3d}/{v['defects']} "
          f"false_rej={v['false_rejections']:3d} cost={v['cost_s']:.4f}s ({v['cost_s_pct_gen']:.2f}% of gen) tok={v['cost_tok']:.0f} ({v['cost_tok_pct_gen']:.0f}%)")
print("direct judge:", S["judge_d"])

for c in casc:
    print(f"cascade {c['name']:9s} in={c['n_in']:3d} caught={c['caught']:3d} false_rej={c['false_rejections']:3d} "
          f"s/defect={c['s_per_defect']} cum_caught={c['cum_caught']} cum_cost={c['cum_s_per_output']:.3f}s "
          f"({c['cum_s_pct_gen']:.1f}%) escaped={c['escaped_after']} residual={c['residual_defect_rate']}")
print("no judge:", S["cascade_no_judge"])
print("judge waste:", S["judge_waste"])
print("price-equivalent:", S["price_equiv"])
print("timings:", S["timing"])
print("agreement:", S["agreement"])
print("escapes:", S["escapes"])
