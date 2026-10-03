"""Verification sandbox. Reads a JSON job from stdin, writes the result as JSON.

stage = "examples" -> runs the >>> examples from the prompt (layer 2)
stage = "hidden"   -> runs the hidden suite (ground truth, not a harness layer)
stage = "probes"   -> evaluates the probe calls and returns the behavioral signature (layer 4)
"""
import sys, json, doctest, io, signal, time, contextlib


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)


def load(program):
    g = {"__name__": "candidate_module"}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        signal.setitimer(signal.ITIMER_REAL, 3.0)
        try:
            exec(program, g)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    return g


def main():
    job = json.loads(sys.stdin.read())
    out = {"ok": False}
    t0 = time.perf_counter()
    try:
        g = load(job["program"])
        fn = g[job["entry_point"]]
        if job["stage"] == "examples":
            examples = doctest.DocTestParser().get_examples(job["spec"])
            test = doctest.DocTest(examples, g, "spec", None, 0, None)
            runner = doctest.DocTestRunner(verbose=False, optionflags=doctest.NORMALIZE_WHITESPACE)
            buf = io.StringIO()
            signal.setitimer(signal.ITIMER_REAL, 4.0)
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    res = runner.run(test, out=buf.write, clear_globs=False)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            out = {"ok": res.failed == 0, "failed": res.failed, "attempted": res.attempted}
        elif job["stage"] == "hidden":
            signal.setitimer(signal.ITIMER_REAL, 4.0)
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    tg = {"__name__": "test_module"}
                    exec(job["test"], tg)
                    tg["check"](fn)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            out = {"ok": True}
        elif job["stage"] == "probes":
            tg = {"__name__": "test_module"}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(job["test"], tg)
            tg["candidate"] = fn
            sig = []
            for p in job["probes"]:
                signal.setitimer(signal.ITIMER_REAL, 1.0)
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        v = eval(p, tg)
                    sig.append(repr(v)[:200])
                except Timeout:
                    sig.append("ERR:Timeout")
                except BaseException as e:  # noqa
                    sig.append("ERR:" + type(e).__name__)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            out = {"ok": True, "signature": sig}
    except Timeout:
        out = {"ok": False, "error": "Timeout"}
    except BaseException as e:  # noqa
        out = {"ok": False, "error": type(e).__name__}
    out["inner_s"] = time.perf_counter() - t0
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
