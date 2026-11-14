"""Write Table 1 (meta-evaluation) LaTeX from meta_eval.json.

    python make_table.py   -> results/eval/table1.tex

Metrics present in meta_eval.json get full rows; metrics listed in PENDING keep their earlier tau
and are marked with a double dagger until they are recomputed.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "meta_eval.json")))

GROUPS = [
    ("Surface overlap", ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "chrF", "Token F1"]),
    ("Embedding similarity", ["BERTScore (heBERT)", "BERTScore (AlephBERT)"]),
    ("Entity hallucination", ["ENFS"]),
    ("NLI", ["SummaC-ZS"]),
]
DISPLAY = {"SummaC-ZS": "SummaC-ZS (mDeBERTa)", "ENFS": "ENFS (entity overlap)"}
PENDING = [("NLI", "SummaC-ZS (mDeBERTa)", "0.006"), ("Trained classifier", "mFACT-He", "0.529"),
           ("LLM judge (GPT-4o-mini)", "Zero-shot judge", "0.599"), ("LLM judge (GPT-4o-mini)", "G-EVAL", "0.242")]


def row(m, name=None, bold=False):
    a, f, h = R["all"][m], R["MAFAT-Aug"][m], R["mFACT-He"][m]
    b = (lambda x: f"\\textbf{{{x}}}") if bold else (lambda x: x)
    auc = f"{a['auc']:.3f}\\,{{\\scriptsize[{a['auc_ci'][0]:.2f},{a['auc_ci'][1]:.2f}]}}"
    return " & ".join([name or DISPLAY.get(m, m), b(auc), b(f"{a['bacc_cv']:.3f}"), b(f"{a['tau_b']:.3f}"),
                       b(f"{f['auc']:.3f}"), b(f"{f['bacc_cv']:.3f}"), b(f"{h['auc']:.3f}"), b(f"{h['bacc_cv']:.3f}")]) + " \\\\"


L = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
     r"\begin{tabular}{lccc|cc|cc}", r"\toprule",
     r" & \multicolumn{3}{c|}{\textbf{All} ($n{=}4{,}024$)} & \multicolumn{2}{c|}{\textbf{MAFAT-Aug} ($n{=}1{,}889$)} & \multicolumn{2}{c}{\textbf{mFACT-He} ($n{=}2{,}135$)} \\",
     r"\textbf{Metric} & AUC [95\% CI] & BAcc & $\tau_b$ & AUC & BAcc & AUC & BAcc \\", r"\midrule"]
shown_groups = set()
for g, ms in GROUPS:
    present = [m for m in ms if m in R["all"]]
    if present:
        L.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{g}}}}} \\")
        shown_groups.add(g)
        L += [row(m) for m in present]
for g, name, tau in PENDING:
    key = [k for k, v in DISPLAY.items() if v == name]
    if key and key[0] in R["all"]:
        continue
    if g not in shown_groups:
        L.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{g}}}}} \\")
        shown_groups.add(g)
    L.append(f"{name}$^{{\\ddagger}}$ & -- & -- & {tau} & -- & -- & -- & -- \\\\")
pending_left = any(not ([k for k, v in DISPLAY.items() if v == n] and [k for k, v in DISPLAY.items() if v == n][0] in R["all"]) for _, n, _ in PENDING)
cap = (r"Meta-evaluation on the Hebrew benchmark. AUC and Kendall's $\tau_b$ are threshold-free; balanced accuracy "
       r"(BAcc) uses a decision threshold tuned on held-out folds (Section~\ref{sec:protocol}). Brackets give 95\% bootstrap "
       r"confidence intervals over articles. MorphEvalHeb outperforms every recomputed baseline ($p<0.001$, paired "
       r"bootstrap on AUC, Holm-corrected).")
if pending_left:
    cap += r" $^{\ddagger}$Not yet recomputed: $\tau$ from our earlier run; the remaining columns are pending."
L += [r"\midrule", row("MorphEvalHeb", r"\textbf{MorphEvalHeb} (ours)", bold=True), r"\bottomrule", r"\end{tabular}",
      r"\caption{" + cap + "}", r"\label{tab:main_results}", r"\end{table*}"]
open(os.path.join(HERE, "table1.tex"), "w").write("\n".join(L) + "\n")
print("table1.tex written,", sum(1 for x in L if x.endswith("\\\\")), "rows incl. headers")
