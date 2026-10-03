import ast, json, re, subprocess, sys, time, doctest, os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "runner.py")


def preamble(prompt, entry_point):
    """Everything the prompt defines before the target function (imports and helpers)."""
    lines = prompt.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith("def " + entry_point + "("):
            return "\n".join(lines[:i])
    return ""


def extract_code(text):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if m:
        return max(m, key=len)
    m = re.search(r"```(?:python|py)?\s*\n(.*)", text, flags=re.S)  # truncated block
    return m.group(1) if m else text


def static_check(code, entry_point):
    """Layer 1: parse the code and check that the target function exists. Returns (ok, seconds)."""
    t = time.perf_counter()
    ok = False
    try:
        tree = ast.parse(code)
        ok = any(isinstance(n, ast.FunctionDef) and n.name == entry_point for n in ast.walk(tree))
    except SyntaxError:
        ok = False
    return ok, time.perf_counter() - t


def sandbox(job, timeout=12):
    """Run one stage in an isolated subprocess. Returns (result, wall-clock seconds)."""
    t = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, RUNNER], input=json.dumps(job), capture_output=True,
                           text=True, timeout=timeout)
        res = json.loads(p.stdout) if p.stdout.strip() else {"ok": False, "error": "Crash"}
    except subprocess.TimeoutExpired:
        res = {"ok": False, "error": "Timeout"}
    return res, time.perf_counter() - t


def probes_from_test(test_src):
    """Extract the candidate(...) calls from the hidden suite: INPUTS only, never the expected output."""
    out = []
    for node in ast.walk(ast.parse(test_src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "candidate":
            s = ast.unparse(node)
            if s not in out:
                out.append(s)
    return out


def spec_doc(prompt):
    """The prompt's docstrings, cleaned (so the closing quotes do not contaminate the last example)."""
    try:
        tree = ast.parse(prompt + "\n    pass\n")
    except SyntaxError:
        return ""
    docs = [ast.get_docstring(n) or "" for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    return "\n\n".join(d for d in docs if d)


def n_examples(prompt):
    try:
        return len(doctest.DocTestParser().get_examples(spec_doc(prompt)))
    except ValueError:  # malformed doctest in the prompt: the task is left out
        return 0
