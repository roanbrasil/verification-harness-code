# What verification costs versus generation

Code and data for the article "What Verification Actually Costs: I Put a Stopwatch on Four Layers of an Agent Harness".

Four verification layers run over 160 generated outputs (32 HumanEval tasks x 5 runs), with time and tokens measured
per call: a static check, the examples from the task prompt, an LLM judge, and agreement across runs.

## Run

```bash
pip install llama-cpp-python huggingface_hub pandas pyarrow matplotlib graphviz
python3 download.py        # GGUF model (~1 GB) and HumanEval -> models/, data/
python3 select_tasks.py    # picks the 32 tasks (seed 42) -> tasks.json
python3 experiment.py      # generation, deterministic layers and judge -> results.jsonl (resumable)
python3 analyze.py         # every number the article cites -> stats.json
python3 make_figs.py en    # figures and tables -> images/ (needs the Graphviz `dot` binary; `pt` also works)
```

`results.jsonl`, `tasks.json` and `stats.json` from the published run are included, so `analyze.py` and
`make_figs.py` work without downloading the model. `experiment.py` resumes from `results.jsonl`: delete that file to
regenerate everything from scratch (about 35 minutes on 2 vCPUs).

## Files

- `download.py`: fetches the model and the dataset.
- `select_tasks.py`: filters HumanEval down to the eligible tasks and samples 32.
- `experiment.py`: phase A (generation), phase B (deterministic layers + ground truth), phase C (LLM judge).
- `analyze.py`: per-layer and cascade statistics, consistency metrics.
- `make_figs.py`: the article's 4 figures and 3 tables, white background.
- `common.py`: code extraction, static layer, sandbox call, probe extraction.
- `runner.py`: subprocess sandbox (spec examples, hidden suite, probes).
- `results.jsonl`: the 160 outputs of the published run, with tokens and timings per call.
- `tasks.json`: the 32 sampled tasks.
- `stats.json`: the aggregated numbers.

## Notes on the data

- Hardware of the published run: 2 vCPUs, no GPU, `n_threads=2`.
- Fields `judge_raw`, `judge_verdict` and `judge_c_*` in `results.jsonl` come from two abandoned attempts at a
  reasoning judge (96- and 220-token limits). The article uses only `judge_d_*`.
- The judge verdict is recomputed in `analyze.py` from `judge_d_raw` (last occurrence of PASS or FAIL); the stored
  `judge_d_verdict` field uses a stricter regex and should be ignored.
- To swap the model, change `MODEL` in `experiment.py`. To use a judge different from the generator, load a second
  `Llama` in phase C.
