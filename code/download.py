"""Download the model and the tasks. Run once before select_tasks.py."""
from huggingface_hub import hf_hub_download

hf_hub_download("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", local_dir="models")
hf_hub_download("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet", repo_type="dataset",
                local_dir="data")
