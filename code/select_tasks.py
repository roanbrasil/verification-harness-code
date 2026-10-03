"""Pick the tasks: HumanEval problems with >>> examples in the prompt, consistent ground truth and >= 3 probes."""
import json, random
import pandas as pd
from common import preamble, sandbox, probes_from_test, n_examples, spec_doc

N_TASKS = 32
df = pd.read_parquet("data/openai_humaneval/test-00000-of-00001.parquet")
eligible = []
for _, r in df.iterrows():
    if n_examples(r.prompt) < 1:
        continue
    program = r.prompt + r.canonical_solution
    ex, _ = sandbox({"stage": "examples", "program": program, "entry_point": r.entry_point, "spec": spec_doc(r.prompt)})
    hid, _ = sandbox({"stage": "hidden", "program": program, "entry_point": r.entry_point, "test": r.test})
    if not (ex.get("ok") and hid.get("ok")):
        continue
    probes = probes_from_test(r.test)
    pr, _ = sandbox({"stage": "probes", "program": program, "entry_point": r.entry_point, "test": r.test,
                     "probes": probes})
    if not pr.get("ok"):
        continue
    good = [p for p, s in zip(probes, pr["signature"]) if not s.startswith("ERR:")]
    if len(good) < 3:
        continue
    eligible.append({"task_id": r.task_id, "prompt": r.prompt, "entry_point": r.entry_point, "test": r.test,
                     "probes": good, "n_examples": n_examples(r.prompt), "spec_doc": spec_doc(r.prompt)})

print("eligible:", len(eligible), "of", len(df))
random.Random(42).shuffle(eligible)
tasks = sorted(eligible[:N_TASKS], key=lambda t: int(t["task_id"].split("/")[1]))
json.dump(tasks, open("tasks.json", "w"), indent=1)
print([t["task_id"] for t in tasks])
print("examples per task (mean):", sum(t["n_examples"] for t in tasks) / len(tasks))
print("probes per task (mean):", sum(len(t["probes"]) for t in tasks) / len(tasks))
