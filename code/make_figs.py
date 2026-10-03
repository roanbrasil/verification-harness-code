"""Figures and tables for the article, generated from stats.json. Usage: python3 make_figs.py en|pt"""
import json, os, sys, textwrap
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import graphviz

LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
OUT = "images"
os.makedirs(OUT, exist_ok=True)
S = json.load(open("stats.json"))
L, C, TR, G = S["layers"], S["cascade"], S["truth"], S["gen"]
K = S["n_runs"]

BG = '#ffffff'; INK = '#20202e'   # white background; the rest of the palette is unchanged
BLUE = '#3b5b92'; BLUEF = '#cddaeb'
GREEN = '#4a7c59'; GREENF = '#cbdccf'
RED = '#bf4f4a'; REDF = '#ecccc9'
AMBER = '#c98a2b'; AMBERF = '#eddcc0'
GREY = '#8d8d8d'
mpl.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
    'font.family': 'serif', 'font.size': 13, 'axes.edgecolor': INK,
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
    'axes.grid': True, 'axes.axisbelow': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.8, 'axes.linewidth': 1.1,
})


def num(x, nd=1):
    s = f"{x:,.{nd}f}"
    if LANG == "pt":
        s = s.replace(",", "§").replace(".", ",").replace("§", ".")
    return s


def dur(s):
    if s < 1e-3:
        return f"{num(s * 1e6, 0)} µs"
    if s < 1:
        return f"{num(s * 1e3, 0)} ms"
    return f"{num(s, 1)} s"


def pct(p):
    if p < 0.1:
        return f"{num(p, 3)}%"
    if p < 100:
        return f"{num(p, 1)}%"
    return f"{num(p, 0)}%"


T = {
    "pt": dict(
        gen="Gerador\n(LLM)", l1="Camada 1\nestática", l2="Camada 2\nexemplos\ndo enunciado", l3="Camada 3\njuiz LLM",
        l4=f"Camada 4\nconcordância\nentre {K} execuções", ok="Aceita", rej="Rejeita\n(regenera ou escala)",
        f2_title="Custo de verificar uma saída, por camada", f2_x="segundos por saída (escala log)",
        f2_ref="gerar uma saída", rows2=["Camada 1: estática", "Camada 2: exemplos do enunciado", "Camada 3: juiz LLM",
                                         "Camada 4: concordância, N=3", f"Camada 4: concordância, N={K}"],
        of_gen="da geração",
        f3_title="O que cada camada pega, rodando em cascata", f3_caught="defeitos pegos",
        f3_false="saídas corretas rejeitadas", f3_y="saídas", f3_esc="defeitos que escapam de tudo",
        rows3=["1 estática", "2 exemplos", "3 juiz LLM", f"4 concordância\n(N={K})"],
        f4_title=f"Mesma tarefa, {K} execuções: quantas passam?", f4_x=f"execuções corretas (de {K})",
        f4_y="tarefas", f4_all=f"passam nas {K}", f4_none="falham nas {K}".replace("{K}", str(K)),
        f4_mixed="instáveis",
    ),
    "en": dict(
        gen="Generator\n(LLM)", l1="Layer 1\nstatic", l2="Layer 2\nspec\nexamples", l3="Layer 3\nLLM judge",
        l4=f"Layer 4\nagreement\nacross {K} runs", ok="Accept", rej="Reject\n(regenerate or escalate)",
        f2_title="Cost of verifying one output, by layer", f2_x="seconds per output (log scale)",
        f2_ref="generating one output", rows2=["Layer 1: static", "Layer 2: spec examples", "Layer 3: LLM judge",
                                               "Layer 4: agreement, N=3", f"Layer 4: agreement, N={K}"],
        of_gen="of generation",
        f3_title="What each layer catches, run as a cascade", f3_caught="defects caught",
        f3_false="correct outputs rejected", f3_y="outputs", f3_esc="defects that escape every layer",
        rows3=["1 static", "2 examples", "3 LLM judge", f"4 agreement\n(N={K})"],
        f4_title=f"Same task, {K} runs: how many pass?", f4_x=f"correct runs (out of {K})",
        f4_y="tasks", f4_all=f"pass all {K}", f4_none=f"fail all {K}", f4_mixed="unstable",
    ),
}[LANG]


# ---------- Figure 1: the pipeline (Graphviz) ----------
def fig1():
    g = graphviz.Digraph(format='png')
    g.attr(rankdir='TB', bgcolor=BG, nodesep='0.45', ranksep='0.9', fontname='DejaVu Serif', dpi='170')
    g.attr('node', shape='box', style='rounded,filled', fontname='DejaVu Serif', fontsize='13', color=INK,
           penwidth='1.4', margin='0.20,0.14')
    g.attr('edge', color=INK, penwidth='1.4', fontname='DejaVu Serif', fontsize='11')
    g.node('gen', T["gen"] + f"\n{dur(G['s_mean'])}", fillcolor=BLUEF, color=BLUE)
    g.node('l1', T["l1"] + f"\n{dur(L['static']['cost_s'])}", fillcolor=GREENF, color=GREEN)
    g.node('l2', T["l2"] + f"\n{dur(L['examples']['cost_s'])}", fillcolor=GREENF, color=GREEN)
    g.node('l3', T["l3"] + f"\n{dur(L['judge_d']['cost_s'])}", fillcolor=AMBERF, color=AMBER)
    g.node('l4', T["l4"] + f"\n{dur(L['agree5']['cost_s'])}", fillcolor=AMBERF, color=AMBER)
    g.node('ok', T["ok"], fillcolor='#eceadf')
    g.node('rej', T["rej"], fillcolor=REDF, color=RED)
    with g.subgraph() as s:
        s.attr(rank='same')
        for n in ['gen', 'l1', 'l2', 'l3', 'l4', 'ok']:
            s.node(n)
    for a, b in [('gen', 'l1'), ('l1', 'l2'), ('l2', 'l3'), ('l3', 'l4'), ('l4', 'ok')]:
        g.edge(a, b)
    for n in ['l1', 'l2', 'l3', 'l4']:
        g.edge(n, 'rej', color=RED, style='dashed')
    g.render(f"{OUT}/fig1_{LANG}", cleanup=True)


# ---------- Figure 2: cost per layer (log) ----------
def fig2():
    keys = ["static", "examples", "judge_d", "agree3", "agree5"]
    vals = [G["s_mean"]] + [L[k]["cost_s"] for k in keys]
    labels = [T["f2_ref"].capitalize()] + T["rows2"]
    cols = [RED, GREEN, GREEN, AMBER, AMBER, AMBER]
    notes = [dur(G["s_mean"])] + [f"{dur(L[k]['cost_s'])}  ·  {pct(L[k]['cost_s_pct_gen'])} {T['of_gen']}" for k in keys]
    fig, ax = plt.subplots(figsize=(11, 5.4), layout='constrained')
    y = list(range(len(vals)))[::-1]
    lo = 10 ** -5
    ax.barh(y, [v - lo for v in vals], left=lo, color=cols, edgecolor=INK, linewidth=1.0, height=0.62)
    ax.set_xscale('log')
    ax.set_xlim(lo, 10 ** 5)
    ax.set_yticks(y, labels)
    for yi, v, n in zip(y, vals, notes):
        ax.text(v * 1.4, yi, n, va='center', fontsize=12,
                bbox=dict(facecolor=BG, edgecolor='none', pad=1.5))
    ax.set_xlabel(T["f2_x"])
    ax.set_title(T["f2_title"], loc='left', fontsize=15, pad=12)
    ax.grid(axis='y', visible=False)
    fig.savefig(f"{OUT}/fig2_{LANG}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------- Figure 3: cascade ----------
def fig3():
    caught = [c["caught"] for c in C]
    false = [c["false_rejections"] for c in C]
    fig, ax = plt.subplots(figsize=(10.5, 5.2), layout='constrained')
    x = range(len(C))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], caught, w, color=GREEN, edgecolor=INK, label=T["f3_caught"])
    b2 = ax.bar([i + w / 2 for i in x], false, w, color=RED, edgecolor=INK, label=T["f3_false"])
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, str(int(b.get_height())), ha='center',
                    va='bottom', fontsize=12)
    ax.set_xticks(list(x), T["rows3"])
    ax.set_ylabel(T["f3_y"])
    top = max(caught + false)
    ax.set_ylim(0, top * 1.28)
    ax.set_title(T["f3_title"], loc='left', fontsize=15, pad=12)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper right', frameon=False)
    esc = C[-1]["escaped_after"]
    ax.text(0.02, 0.93, f"{T['f3_esc']}: {esc} / {TR['defects']}", transform=ax.transAxes, ha='left', va='top',
            fontsize=12, color=INK)
    fig.savefig(f"{OUT}/fig3_{LANG}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------- Figure 4: consistency ----------
def fig4():
    h = [TR["hist_k"][str(k)] for k in range(K + 1)]
    cols = [RED] + [AMBER] * (K - 1) + [GREEN]
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    bars = ax.bar(range(K + 1), h, color=cols, edgecolor=INK, width=0.7)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, str(int(b.get_height())), ha='center',
                va='bottom', fontsize=12)
    ax.set_xticks(range(K + 1))
    ax.set_xlabel(T["f4_x"])
    ax.set_ylabel(T["f4_y"])
    ax.set_ylim(0, max(h) * 1.2)
    ax.set_title(T["f4_title"], loc='left', fontsize=15, pad=12)
    ax.grid(axis='x', visible=False)
    txt = (f"Mean@{K} = {num(100 * TR['mean_at_k'])}%\nPass^{K} = {num(100 * TR['pass_hat_k'])}%\n"
           f"gap = {num(TR['gap_pp'], 2)} pp")
    ax.text(0.33, 0.93, txt, transform=ax.transAxes, va='top', ha='left', fontsize=13, linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#eceadf', edgecolor=INK))
    fig.savefig(f"{OUT}/fig4_{LANG}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------- Tables (white background) ----------
def table(name, header, rows, widths, wrap):
    mpl.rcParams.update({'font.family': 'sans-serif'})
    lines = [[textwrap.wrap(str(c), w) or [""] for c, w in zip(r, wrap)] for r in [header] + rows]
    heights = [max(len(c) for c in r) * 0.30 + 0.26 for r in lines]
    W, H = sum(widths), sum(heights)
    fig, ax = plt.subplots(figsize=(W, H))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    y = H
    for ri, (r, h) in enumerate(zip(lines, heights)):
        x = 0
        for c, w in zip(r, widths):
            ax.add_patch(plt.Rectangle((x, y - h), w, h, facecolor='#eef1f6' if ri == 0 else 'white',
                                       edgecolor='#b9bec9', linewidth=0.9))
            ax.text(x + 0.12, y - h / 2, "\n".join(c), va='center', ha='left', fontsize=11.5,
                    fontweight='bold' if ri == 0 else 'normal', color='#1a1a1a', linespacing=1.35)
            x += w
        y -= h
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(f"{OUT}/{name}_{LANG}.png", dpi=170, facecolor='white', bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    mpl.rcParams.update({'font.family': 'serif'})


def tables():
    def row(k, name, needs):
        v = L[k]
        tok = "0" if v["cost_tok"] == 0 else num(v["cost_tok"], 0)
        return [name, needs, dur(v["cost_s"]), pct(v["cost_s_pct_gen"]), tok, f"{v['caught']} / {v['defects']}",
                f"{v['false_rejections']} / {v['n'] - v['defects']}"]
    if LANG == "pt":
        header = ["Camada", "Do que ela precisa", "Tempo por saída", "% do tempo de gerar", "Tokens por saída",
                  "Defeitos pegos", "Corretas rejeitadas"]
        rows = [row("static", "1. Estática", "Só o código gerado"),
                row("examples", "2. Exemplos do enunciado", "Exemplos de entrada e saída já escritos na especificação"),
                row("judge_d", "3. Juiz LLM", "Especificação + código, sem executar nada"),
                row("agree3", "4. Concordância, N=3", "Entradas de sonda, sem saída esperada, e 2 gerações extras"),
                row("agree5", f"4. Concordância, N={K}", "Entradas de sonda, sem saída esperada, e 4 gerações extras")]
        h2 = ["Fonte", "Agente e tarefa", "Média por execução", "Passa em todas as k", "Gap"]
        r2 = [["IBM Research (Duesterwald et al., set. 2026), citado", "ReAct com GPT-4.1, AppWorld, 168 tarefas, k=5",
               "77,4%", "53,0%", "24,4 pp"],
              ["τ-bench (Yao et al., 2024), citado", "gpt-4o com function calling, domínio retail, k=8",
               "abaixo de 50%", "abaixo de 25%", "não informado no resumo"],
              ["Este artigo, medido", f"Qwen2.5-Coder 1.5B Q4_K_M, {S['n_tasks']} tarefas HumanEval, k={K}",
               f"{num(100 * TR['mean_at_k'])}%", f"{num(100 * TR['pass_hat_k'])}%", f"{num(TR['gap_pp'], 2)} pp"]]
    else:
        header = ["Layer", "What it needs", "Time per output", "% of generation time", "Tokens per output",
                  "Defects caught", "Correct outputs rejected"]
        rows = [row("static", "1. Static", "Only the generated code"),
                row("examples", "2. Spec examples", "Input/output examples already written in the spec"),
                row("judge_d", "3. LLM judge", "Spec + code, nothing executed"),
                row("agree3", "4. Agreement, N=3", "Probe inputs, no expected output, and 2 extra generations"),
                row("agree5", f"4. Agreement, N={K}", "Probe inputs, no expected output, and 4 extra generations")]
        h2 = ["Source", "Agent and task", "Mean per run", "Passes all k runs", "Gap"]
        r2 = [["IBM Research (Duesterwald et al., Sep 2026), cited", "ReAct with GPT-4.1, AppWorld, 168 tasks, k=5",
               "77.4%", "53.0%", "24.4 pp"],
              ["τ-bench (Yao et al., 2024), cited", "gpt-4o with function calling, retail domain, k=8",
               "below 50%", "below 25%", "not given in the abstract"],
              ["This article, measured", f"Qwen2.5-Coder 1.5B Q4_K_M, {S['n_tasks']} HumanEval tasks, k={K}",
               f"{num(100 * TR['mean_at_k'])}%", f"{num(100 * TR['pass_hat_k'])}%", f"{num(TR['gap_pp'], 2)} pp"]]
    table("tab1", header, rows, [2.3, 3.3, 1.35, 1.5, 1.3, 1.3, 1.45], [20, 30, 10, 12, 10, 10, 12])
    names = {"pt": {"static": "1. Estática", "examples": "2. Exemplos do enunciado", "judge_d": "3. Juiz LLM",
                    "agree5": f"4. Concordância, N={K}"},
             "en": {"static": "1. Static", "examples": "2. Spec examples", "judge_d": "3. LLM judge",
                    "agree5": f"4. Agreement, N={K}"}}[LANG]
    hc = {"pt": ["Camada", "Saídas que entram", "Defeitos pegos", "Corretas rejeitadas", "Defeitos que seguem",
                 "Custo acumulado por saída", "% do tempo de gerar", "Segundos por defeito pego"],
          "en": ["Layer", "Outputs in", "Defects caught", "Correct outputs rejected", "Defects still in",
                 "Cumulative cost per output", "% of generation time", "Seconds per defect caught"]}[LANG]
    rc = [[names[c["name"]], str(c["n_in"]), str(c["caught"]), str(c["false_rejections"]), str(c["escaped_after"]),
           dur(c["cum_s_per_output"]), pct(c["cum_s_pct_gen"]),
           "—" if c["s_per_defect"] is None else num(c["s_per_defect"], 2 if c["s_per_defect"] < 10 else (1 if c["s_per_defect"] < 100 else 0))]
          for c in C]
    table("tab2", hc, rc, [2.6, 1.25, 1.25, 1.35, 1.35, 1.7, 1.45, 1.65], [22, 10, 10, 11, 11, 14, 12, 14])
    table("tab3", h2, r2, [3.2, 4.0, 1.7, 1.7, 1.9], [29, 38, 14, 14, 16])


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); tables()
    print("ok", LANG)
