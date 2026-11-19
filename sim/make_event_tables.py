"""Build the LaTeX results tables of final-skeleton.tex (M1 and M3, with and without events)
from the CSVs written by run_m1_table2.py and the printout of run_m1_events.py.

Usage: python3 make_event_tables.py <m1_events_printout.txt> > tables.tex
Reads results_m1_table2.csv and results_m3_table2.csv in this folder.
A cell is red (\\tie) when the full method does not have the lowest mean.
"""
import csv
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EV = ["full", "no_ghost_credit", "fixed_launch_nodes", "nearest_ugv_recovery"]
ALL = EV + ["tier0_ugv_only"]
NAME = {"full": "full method", "no_ghost_credit": "no air-only credit",
        "fixed_launch_nodes": "fixed launch nodes", "nearest_ugv_recovery": "nearest-UGV recovery",
        "tier0_ugv_only": "UGV-only (floor)"}
SIG = (1, 2, 3)
TCRIT = 2.05     # 5 %, 29 degrees of freedom (30 seeds)


def load(name):
    rows = list(csv.DictReader(open(os.path.join(HERE, name))))
    for r in rows:
        r["seed"], r["severity"] = int(r["seed"]), int(r["severity"])
        r["frozen"] = r["frozen"] == "True"
    return rows


def fleets(rows):
    fl = sorted({r["fleet"] for r in rows}, key=lambda f: tuple(map(int, f.split("-"))))
    return fl


def get(rows, fl, v, sig, key, frozen=False):
    xs = {r["seed"]: float(r[key]) for r in rows if r["fleet"] == fl and r["variant"] == v
          and r["severity"] == sig and r["frozen"] == frozen and r.get(key) not in (None, "")}
    return xs


def paired(a, b):
    s = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in s]
    mu = st.mean(d)
    se = st.stdev(d) / len(d) ** .5
    return mu, mu / se if se else 0.0, sum(x < 0 for x in d), len(d)


def fmt_f(fl):
    return fl.replace("-", " / ")


def closest(rows, fl, sig):
    """baseline with the lowest mean combined; paired margin full minus it"""
    m = {v: st.mean(get(rows, fl, v, sig, "combined").values()) for v in EV}
    b = min((v for v in EV if v != "full"), key=lambda v: m[v])
    mu, t, w, n = paired(get(rows, fl, "full", sig, "combined"), get(rows, fl, b, sig, "combined"))
    return b, mu, t, w, n, m


def table_noevent_full(rows, mapname, label, caption):
    out = [r"\begin{table}[H]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           r"\begin{tabular}{llrrrrrr}", r"\toprule",
           r"Fleet & Variant & Road & Air mean & Air peak & Combined & Road peak & Prio.\ peak \\", r"\midrule"]
    for i, fl in enumerate(fleets(rows)):
        if i:
            out.append(r"\addlinespace")
        means = {v: {k: st.mean(get(rows, fl, v, 0, k).values()) for k in
                     ("mean_idleness", "ghost_mean_idleness", "ghost_max_idleness", "combined",
                      "road_max_idleness", "priority_max_idleness")} for v in ALL}
        best = min(ALL, key=lambda v: means[v]["combined"])
        for j, v in enumerate(ALL):
            m = means[v]
            comb = f"{m['combined']:.1f}"
            if v == best:
                comb = rf"\textbf{{{comb}}}"
            cells = [fmt_f(fl) if j == 0 else "", NAME[v], f"{m['mean_idleness']:.1f}", f"{m['ghost_mean_idleness']:.1f}",
                     f"{m['ghost_max_idleness']:.1f}", comb, f"{m['road_max_idleness']:.1f}", f"{m['priority_max_idleness']:.1f}"]
            if v == "full" and best != "full":
                cells = [rf"\tie{{{c}}}" if c else c for c in cells]
            out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def table_event(rows, label, caption):
    out = [r"\begin{table}[H]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           r"\begin{tabular}{llrrrr}", r"\toprule",
           r"Fleet & Method & No event & $\sigma = 1$ & $\sigma = 2$ & $\sigma = 3$ \\", r"\midrule"]
    for i, fl in enumerate(fleets(rows)):
        if i:
            out.append(r"\addlinespace")
        m = {(v, s): st.mean(get(rows, fl, v, s, "combined").values()) for v in EV for s in (0,) + SIG}
        best = {s: min(EV, key=lambda v: m[(v, s)]) for s in (0,) + SIG}
        for j, v in enumerate(EV):
            cells = [fmt_f(fl) if j == 0 else "", NAME[v]]
            for s in (0,) + SIG:
                c = f"{m[(v, s)]:.1f}"
                if s:
                    p = 100 * (m[(v, s)] / m[(v, 0)] - 1)
                    c += f" ({'+' if p >= 0 else '$-$'}{abs(p):.1f}\\,\\%)"
                if best[s] == v:
                    c = rf"\textbf{{{c.split(' ')[0]}}}" + (" " + " ".join(c.split(" ")[1:]) if s else "")
                if v == "full" and best[s] != "full":
                    c = rf"\tie{{{c}}}"
                cells.append(c)
            out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def table_margin(rows, label, caption):
    out = [r"\begin{table}[H]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Fleet & No event & $\sigma = 1$ & $\sigma = 2$ & $\sigma = 3$ \\", r"\midrule"]
    stats = []
    for fl in fleets(rows):
        cells = [fmt_f(fl)]
        for s in (0,) + SIG:
            b, mu, t, w, n, _ = closest(rows, fl, s)
            stats.append((fl, s, b, mu, t, w, n))
            c = f"${mu:+.1f}$ (${t:.2f}$) {w}"
            if mu > 0:
                c = rf"\tie{{{c}}}"
            cells.append(c)
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out), stats


def table_frozen(rows, label, caption):
    out = [r"\begin{table}[H]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
           r"& \multicolumn{3}{c}{Response window} & \multicolumn{3}{c}{Whole run} \\",
           r"Fleet & $\sigma = 1$ & $\sigma = 2$ & $\sigma = 3$ & $\sigma = 1$ & $\sigma = 2$ & $\sigma = 3$ \\", r"\midrule"]
    stats = []
    for fl in fleets(rows):
        a, b = [], []
        for s in SIG:
            ew, fw = get(rows, fl, "full", s, "resp_window"), get(rows, fl, "full", s, "resp_window", True)
            mu, t, _, _ = paired(ew, fw)
            pct = 100 * mu / st.mean(fw.values())
            a.append(f"${pct:+.1f}$\\,\\% (${t:.2f}$)")
            mu2, t2, _, _ = paired(get(rows, fl, "full", s, "combined"), get(rows, fl, "full", s, "combined", True))
            b.append(f"${mu2:+.1f}$ (${t2:.2f}$)")
            stats.append((fl, s, pct, t, mu2, t2))
        out.append(" & ".join([fmt_f(fl)] + a + b) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out), stats


def table_dispatch(txt, label, caption):
    """parse run_m1_events.py printout"""
    lines = open(txt).read().splitlines()
    rows, fl = [], None
    for i, ln in enumerate(lines):
        mf = re.match(r"M1 (\d+-\d+),", ln)
        if mf:
            fl = mf.group(1)
            continue
        mr = re.match(r"\s+(\d)\s+(full|V4_single|V5_random)\s+(.*)", ln)
        if mr:
            f = mr.group(3).split()
            # dJ dJ% dRoad dAir clear sxclr D_resp first short | ... | resp red served
            resp = re.search(r"\|\s+([+-]?\d+\.\d)%\s+([+-]?\d+\.\d)%\s+served", ln).group(1)
            post = re.search(r"post-recovery dJ% \(after clear \+ T\): ([+-]?\d+\.\d)%", lines[i + 1]).group(1)
            rows.append((fl, int(mr.group(1)), mr.group(2), float(resp), float(post), float(f[4]), float(f[7]),
                         float(f[6]), float(f[8])))
    nm = {"full": r"Alg.~\ref{alg:respond}", "V4_single": "V4 single", "V5_random": "V5 random"}
    out = [r"\begin{table}[H]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           r"\begin{tabular}{llrrrrrr}", r"\toprule",
           r"Fleet, $\sigma$ & Selection & $\Delta J_{\mathrm{resp}}$ & $\Delta J_{\mathrm{post}}$ & $\tau_{\mathrm{clear}}$ & First & $D_{\mathrm{resp}}$ & Short \\",
           r"\midrule"]
    prev = None
    for fl, s, v, resp, post, clr, first, d, short in rows:
        if prev and prev != fl:
            out.append(r"\addlinespace")
        prev = fl
        head = f"{fmt_f(fl)}, {s}" if v == "full" else ""
        sg = lambda x: f"{'+' if x >= 0 else '$-$'}{abs(x):.1f}\\,\\%"
        out.append(f"{head} & {nm[v]} & {sg(resp)} & {sg(post)} & {clr:.0f} & {first:.1f} & {d:.0f} & {short:.2f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out), rows


if __name__ == "__main__":
    m1, m3 = load("results_m1_table2.csv"), load("results_m3_table2.csv")
    parts = {}
    parts["m1"] = table_noevent_full(m1, "M1", "tab:m1",
        r"M1, no event, covering walk set, 500\,m recovery radius, mean over seeds 0--29, 2-hour horizon, idleness in seconds.")
    parts["m3"] = table_noevent_full(m3, "M3", "tab:m3",
        r"M3 (Boston), no event, covering walk set, 500\,m recovery radius, $T = 421$\,s ($\geq T^\star$), mean over seeds 0--29, 2-hour horizon, idleness in seconds.")
    mm1, s_m1 = table_margin(m1, "tab:m1evmargin",
        r"M1, margin of the full method over the closest baseline (paired over seeds 0--29): difference in combined idleness (s), $t$ in brackets, and the number of seeds where the full method is lower. Negative: the full method is lower. $|t| > 2.05$ is significant at 5\,\%.")
    mm3, s_m3 = table_margin(m3, "tab:m3evmargin",
        r"M3, margin of the full method over the closest baseline, same layout as Table~\ref{tab:m1evmargin}.")
    parts["m1evmargin"], parts["m3evmargin"] = mm1, mm3
    parts["m1ev"] = table_event(m1, "tab:m1ev",
        r"M1, combined idleness (s) over the whole 2-hour horizon, without an event and with one event at $t_0 = 3600$\,s, mean over seeds 0--29. In brackets: $\Delta J_{\mathrm{event}}$, the rise against the same method's run without the event. Bold: lowest in each column of a fleet.")
    parts["m3ev"] = table_event(m3, "tab:m3ev",
        r"M3, combined idleness (s), same layout as Table~\ref{tab:m1ev}.")
    f1, fs1 = table_frozen(m1, "tab:m1frozen",
        r"M1, frozen check, full method: event run minus frozen run (same seed, same quota stopped for the same $\tau_{\mathrm{clear}}$), mean over seeds 0--29, $t$ in brackets. Response: combined idleness from onset to $\tau_{\mathrm{clear}}$, relative to the frozen run. Whole run: difference in combined idleness over the horizon (s).")
    f3, fs3 = table_frozen(m3, "tab:m3frozen",
        r"M3, frozen check, full method, same layout as Table~\ref{tab:m1frozen}.")
    parts["m1frozen"], parts["m3frozen"] = f1, f3
    if len(sys.argv) > 1:
        parts["m1dispatch"], drows = table_dispatch(sys.argv[1], "tab:m1dispatch",
            r"M1, responder selection, patrol = full method, mean over seeds 0--29. $\Delta J_{\mathrm{resp}}$: rise in combined idleness from onset to $\tau_{\mathrm{clear}}$; $\Delta J_{\mathrm{post}}$: rise from $\tau_{\mathrm{clear}} + T$ to the horizon; both against the same seed without the event. First: onset to first arrival (s). Short: quota slots not filled at onset.")
    for k, v in parts.items():
        print(f"%%%% BEGIN {k}\n{v}\n%%%% END {k}")
    # summary for the prose (stderr)
    for nm_, ss in (("M1", s_m1), ("M3", s_m3)):
        sig_w = sum(1 for x in ss if x[3] < 0 and abs(x[4]) > TCRIT)
        low = sum(1 for x in ss if x[3] < 0)
        print(f"{nm_}: full lowest {low}/{len(ss)}, significant {sig_w}", file=sys.stderr)
        for x in ss:
            print("   ", x[0], x[1], x[2], f"{x[3]:+.1f} t {x[4]:.2f} {x[5]}/{x[6]}", file=sys.stderr)
    for nm_, ss in (("M1", fs1), ("M3", fs3)):
        print(nm_, "frozen", [(x[0], x[1], round(x[2], 1), round(x[3], 2), round(x[4], 1), round(x[5], 2)) for x in ss], file=sys.stderr)
