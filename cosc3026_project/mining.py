"""
COSC-3026 Final Project - Density-Aware Frequent Itemset Mining
Apriori vs FP-Growth, with a TID-list Apriori variant (Addition A)
and a density-based selector (Addition B).

Usage:
    python mining.py verify                      # correctness checks on a small dataset
    python mining.py bench DATA_DIR OUT_DIR      # full benchmark on FIMI-format files
        DATA_DIR must contain: retail.dat, chess.dat, mushroom.dat (FIMI format)
        and optionally groceries.dat (one basket per line, items separated by spaces
        or commas). Results are written to OUT_DIR/results.csv and charts (png).

All three miners return exactly the same frequent itemsets {frozenset: count}.
Support is ALWAYS computed over the ORIGINAL number of transactions.
"""
import sys, os, time, math, csv, itertools, tracemalloc, statistics
from collections import defaultdict

# --------------------------------------------------------------------------
# Loading / preprocessing
# --------------------------------------------------------------------------
def load_transactions(path):
    """Read a basket file (FIMI style or comma separated).
    Preprocessing: remove duplicate items in a basket, drop empty baskets,
    map item labels to integer IDs."""
    raw = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:      # CSV baskets, e.g. "whole milk,yogurt" (items may contain spaces)
                items = {x.strip().strip('"') for x in line.split(",")}
                items.discard("")
            else:                # FIMI style: integers separated by spaces
                items = set(line.split())
            if items:
                raw.append(items)
    ids = {}
    tx = []
    for items in raw:
        row = set()
        for it in items:
            if it not in ids:
                ids[it] = len(ids)
            row.add(ids[it])
        tx.append(frozenset(row))
    return tx


def density(tx):
    """Average transaction length divided by number of distinct items."""
    n_items = len({i for t in tx for i in t})
    avg_len = sum(len(t) for t in tx) / len(tx)
    return avg_len / n_items


def dataset_stats(tx):
    n_items = len({i for t in tx for i in t})
    avg_len = sum(len(t) for t in tx) / len(tx)
    return {"transactions": len(tx), "items": n_items,
            "avg_len": round(avg_len, 2), "density": round(avg_len / n_items, 4)}


def _min_count(n, min_sup):
    return max(1, math.ceil(min_sup * n - 1e-9))


# --------------------------------------------------------------------------
# Baseline Apriori (Agrawal & Srikant [1]) - every candidate is checked
# against every transaction in every pass k >= 2.
# --------------------------------------------------------------------------
def _gen_candidates(prev_frequent, k):
    """Join step + prune step. prev_frequent: set of sorted tuples of size k-1."""
    prev = sorted(prev_frequent)
    prev_set = set(prev)
    cands = []
    for i in range(len(prev)):
        a = prev[i]
        for j in range(i + 1, len(prev)):
            b = prev[j]
            if a[:-1] != b[:-1]:
                break
            c = a + (b[-1],)
            # prune: all (k-1)-subsets must be frequent
            if all(c[:m] + c[m + 1:] in prev_set for m in range(k - 2)):
                cands.append(c)
    return cands


def apriori(tx, min_sup):
    n = len(tx)
    mc = _min_count(n, min_sup)
    stats = {"candidates": 0, "checks": 0, "passes": 1}
    c1 = defaultdict(int)
    for t in tx:
        for i in t:
            c1[i] += 1
    freq = {frozenset([i]): c for i, c in c1.items() if c >= mc}
    keep = {next(iter(s)) for s in freq}
    # shared preprocessing for ALL miners: restrict to frequent items
    tx2 = [t & keep for t in tx]
    tx2 = [t for t in tx2 if len(t) >= 1]
    prev = {(i,) for i in keep}
    k = 2
    while prev:
        cands = _gen_candidates(prev, k)
        if not cands:
            break
        stats["candidates"] += len(cands)
        stats["passes"] += 1
        counts = defaultdict(int)
        for t in tx2:                      # scan ALL transactions
            for c in cands:
                stats["checks"] += 1
                if t.issuperset(c):
                    counts[c] += 1
        prev = set()
        for c, v in counts.items():
            if v >= mc:
                freq[frozenset(c)] = v
                prev.add(c)
        k += 1
    return freq, stats


# --------------------------------------------------------------------------
# Addition A: TID-list Apriori.  Candidates are counted only inside the TID list
# of their least-frequent item. Single-item transactions are never needed for
# k >= 2 counting; 1-itemset counts and the support denominator use the original data.
# Optional early pruning: skip a candidate whose rarest-item TID list is shorter
# than the minimum count (cannot be frequent).
# --------------------------------------------------------------------------
def apriori_tid(tx, min_sup, early_prune=True):
    n = len(tx)
    mc = _min_count(n, min_sup)
    stats = {"candidates": 0, "checks": 0, "passes": 1, "pruned_early": 0}
    c1 = defaultdict(int)
    for t in tx:
        for i in t:
            c1[i] += 1
    freq = {frozenset([i]): c for i, c in c1.items() if c >= mc}
    keep = {next(iter(s)) for s in freq}
    tx2 = [t & keep for t in tx]            # same shared preprocessing as baseline
    tidlist = defaultdict(list)             # item -> transaction indexes (len >= 2 only)
    for idx, t in enumerate(tx2):
        if len(t) >= 2:                     # single-item transactions are skipped
            for i in t:
                tidlist[i].append(idx)
    prev = {(i,) for i in keep}
    k = 2
    while prev:
        cands = _gen_candidates(prev, k)
        if not cands:
            break
        stats["candidates"] += len(cands)
        stats["passes"] += 1
        counts = {}
        for c in cands:
            rare = min(c, key=lambda i: len(tidlist[i]))
            tl = tidlist[rare]
            if early_prune and len(tl) < mc:
                stats["pruned_early"] += 1
                continue
            v = 0
            cs = frozenset(c)
            for tid in tl:
                stats["checks"] += 1
                if tx2[tid].issuperset(cs):
                    v += 1
            counts[c] = v
        prev = set()
        for c, v in counts.items():
            if v >= mc:
                freq[frozenset(c)] = v
                prev.add(c)
        k += 1
    return freq, stats


# --------------------------------------------------------------------------
# FP-Growth (Han, Pei, Yin [2])
# --------------------------------------------------------------------------
class _Node:
    __slots__ = ("item", "count", "parent", "children", "link")
    def __init__(self, item, parent):
        self.item, self.count, self.parent = item, 0, parent
        self.children, self.link = {}, None


def _build_tree(weighted_tx, order):
    root = _Node(None, None)
    header = {}
    tails = {}
    for items, w in weighted_tx:
        items = sorted((i for i in items if i in order), key=order.get)
        node = root
        for i in items:
            ch = node.children.get(i)
            if ch is None:
                ch = _Node(i, node)
                node.children[i] = ch
                if i in header:
                    tails[i].link = ch
                else:
                    header[i] = ch
                tails[i] = ch
            ch.count += w
            node = ch
    return header


def _mine(header, counts, suffix, mc, out, stats):
    # process items from least frequent to most frequent
    for item in sorted(counts, key=lambda i: counts[i]):
        new = suffix | {item}
        out[frozenset(new)] = counts[item]
        stats["nodes_visited"] += 0
        # conditional pattern base
        base = []
        node = header[item]
        while node is not None:
            path = []
            p = node.parent
            while p is not None and p.item is not None:
                path.append(p.item)
                p = p.parent
            if path:
                base.append((path, node.count))
            node = node.link
        cc = defaultdict(int)
        for path, w in base:
            for i in path:
                cc[i] += w
        cc = {i: c for i, c in cc.items() if c >= mc}
        if cc:
            order = {i: r for r, i in enumerate(sorted(cc, key=lambda x: (-cc[x], x)))}
            stats["cond_trees"] += 1
            h2 = _build_tree(base, order)
            _mine(h2, cc, new, mc, out, stats)


def fpgrowth(tx, min_sup):
    n = len(tx)
    mc = _min_count(n, min_sup)
    stats = {"candidates": 0, "checks": 0, "passes": 2, "cond_trees": 0, "nodes_visited": 0}
    c1 = defaultdict(int)
    for t in tx:
        for i in t:
            c1[i] += 1
    counts = {i: c for i, c in c1.items() if c >= mc}
    order = {i: r for r, i in enumerate(sorted(counts, key=lambda x: (-counts[x], x)))}
    header = _build_tree(((t, 1) for t in tx), order)
    out = {}
    _mine(header, counts, frozenset(), mc, out, stats)
    return out, stats


# --------------------------------------------------------------------------
# Association rules (for the rule-count comparison, mirroring [4])
# --------------------------------------------------------------------------
def count_rules(freq, n, thresholds=(0.1, 0.2, 0.3, 0.4)):
    """Number of rules A->B (A, B non-empty, disjoint) whose confidence >= each
    threshold. Support of every itemset is count / n, n = ORIGINAL transaction count."""
    res = {t: 0 for t in thresholds}
    for itemset, cnt in freq.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for ante in itertools.combinations(items, r):
                a = freq.get(frozenset(ante))
                if a:
                    conf = cnt / a
                    for t in thresholds:
                        if conf >= t - 1e-12:
                            res[t] += 1
    return res


def reduced_denominator_experiment(tx, min_sup, thresholds=(0.1, 0.2, 0.3, 0.4)):
    """Qiu et al. [4]-style: delete single-item transactions, then compute support
    over the REDUCED transaction count. Compare with the fixed-denominator version."""
    fixed, _ = apriori_tid(tx, min_sup)
    reduced_tx = [t for t in tx if len(t) >= 2]
    red, _ = apriori(reduced_tx, min_sup)
    return {
        "n_original": len(tx), "n_reduced": len(reduced_tx),
        "freq_fixed": len(fixed), "freq_reduced": len(red),
        "rules_fixed": count_rules(fixed, len(tx), thresholds),
        "rules_reduced": count_rules(red, len(reduced_tx), thresholds),
    }


# --------------------------------------------------------------------------
# Addition B: density-based selector
# --------------------------------------------------------------------------
def select_algorithm(dens, min_sup, tau_d, tau_s):
    """Apriori-TID when data is sparse AND support is high; otherwise FP-Growth."""
    return "apriori_tid" if (dens < tau_d and min_sup >= tau_s) else "fpgrowth"


def calibrate(records, train_names):
    """records: list of dicts {dataset, density, min_sup, t_apriori_tid, t_fpgrowth}.
    Grid-search (tau_d, tau_s) on training datasets; return best pair and accuracy."""
    train = [r for r in records if r["dataset"] in train_names]
    ds = sorted({r["density"] for r in train})
    ss = sorted({r["min_sup"] for r in train})
    cand_d = [0.0] + [(a + b) / 2 for a, b in zip(ds, ds[1:])] + [ds[-1] * 2]
    cand_s = [0.0] + [(a + b) / 2 for a, b in zip(ss, ss[1:])] + [ss[-1] * 2]
    best = (None, None, -1)
    for td in cand_d:
        for ts in cand_s:
            acc = accuracy(train, td, ts)
            if acc > best[2]:
                best = (td, ts, acc)
    return best


def accuracy(recs, tau_d, tau_s):
    if not recs:
        return float("nan")
    ok = 0
    for r in recs:
        fast = "apriori_tid" if r["t_apriori_tid"] <= r["t_fpgrowth"] else "fpgrowth"
        ok += select_algorithm(r["density"], r["min_sup"], tau_d, tau_s) == fast
    return ok / len(recs)


# --------------------------------------------------------------------------
# Measurement helpers
# --------------------------------------------------------------------------
def timed(fn, tx, min_sup, runs=5):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        res, st = fn(tx, min_sup)
        times.append(time.perf_counter() - t0)
    return res, st, statistics.mean(times), (statistics.stdev(times) if runs > 1 else 0.0)


def peak_mem(fn, tx, min_sup):
    tracemalloc.start()
    fn(tx, min_sup)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6  # MB


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
def brute_force(tx, min_sup, max_size=6):
    n = len(tx)
    mc = _min_count(n, min_sup)
    items = sorted({i for t in tx for i in t})
    out = {}
    for k in range(1, max_size + 1):
        found = False
        for c in itertools.combinations(items, k):
            cs = frozenset(c)
            v = sum(1 for t in tx if cs <= t)
            if v >= mc:
                out[cs] = v
                found = True
        if not found:
            break
    return out


def reference_mlxtend(tx, min_sup):
    """Use mlxtend (if installed) as an independent reference implementation."""
    import pandas as pd
    from mlxtend.frequent_patterns import apriori as mx_apriori
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    df = pd.DataFrame(te.fit(tx).transform(tx), columns=te.columns_)
    r = mx_apriori(df, min_support=min_sup, use_colnames=True)
    return {frozenset(s): round(sup * len(tx)) for sup, s in zip(r["support"], r["itemsets"])}


def verify():
    import random
    random.seed(1)
    ok_all = True
    for trial in range(5):
        items = list(range(12))
        tx = [frozenset(random.sample(items, random.randint(1, 6))) for _ in range(120)]
        for ms in (0.05, 0.1, 0.2):
            ref = brute_force(tx, ms)
            for name, fn in (("apriori", apriori), ("apriori_tid", apriori_tid),
                             ("fpgrowth", fpgrowth)):
                res, _ = fn(tx, ms)
                same = res == ref
                ok_all &= same
                if not same:
                    print("MISMATCH", name, trial, ms)
    try:
        tx = [frozenset(random.sample(range(15), random.randint(1, 7))) for _ in range(200)]
        ref = reference_mlxtend(tx, 0.1)
        res, _ = fpgrowth(tx, 0.1)
        print("mlxtend reference match:", res == ref)
    except ImportError:
        print("mlxtend not installed - checked against brute force only")
    print("ALL MINERS IDENTICAL TO REFERENCE:", ok_all)


# --------------------------------------------------------------------------
# Benchmark driver
# --------------------------------------------------------------------------
DEFAULT_SUPPORTS = {
    # Start with these. If a dataset is too slow, delete its LOWEST support; if it is
    # very fast, add a lower one. Pure Python gets slow at low supports on dense data.
    "groceries": [0.01, 0.007, 0.005, 0.003],
    "retail":    [0.06, 0.04, 0.03, 0.02],
    "chess":     [0.95, 0.9, 0.85, 0.8],
    "mushroom":  [0.5, 0.4, 0.35, 0.3],
}
# Rule counting enumerates all subsets of every itemset, which explodes on dense data,
# so the rule experiment (Table IV) is run only on these (sparse) datasets.
RULE_DATASETS = ("groceries", "retail")


def bench(data_dir, out_dir, runs=3, supports=None, train=("groceries", "chess"),
          heldout=("retail", "mushroom")):
    supports = supports or DEFAULT_SUPPORTS
    os.makedirs(out_dir, exist_ok=True)
    records, stats_rows, rule_rows = [], [], []
    for name, sups in supports.items():
        path = os.path.join(data_dir, name + ".dat")
        if not os.path.exists(path):
            print("skip (missing):", path)
            continue
        tx = load_transactions(path)
        ds = dataset_stats(tx)
        print(name, ds)
        stats_rows.append({"dataset": name, **ds})
        for ms in sups:
            row = {"dataset": name, "density": ds["density"], "min_sup": ms}
            ref_set = None
            for key, fn in (("apriori", apriori), ("apriori_tid", apriori_tid),
                            ("fpgrowth", fpgrowth)):
                res, st, mt, sd = timed(fn, tx, ms, runs)
                if ref_set is None:
                    ref_set = res
                assert res == ref_set, f"OUTPUT MISMATCH {key} {name} {ms}"
                row[f"t_{key}"] = mt
                row[f"sd_{key}"] = sd
                row[f"mem_{key}"] = peak_mem(fn, tx, ms)
                row[f"cand_{key}"] = st["candidates"]
                row[f"checks_{key}"] = st["checks"]
            row["n_frequent"] = len(ref_set)
            records.append(row)
            if name in RULE_DATASETS:
                rr = reduced_denominator_experiment(tx, ms)
                rule_rows.append({"dataset": name, "min_sup": ms, **{
                    k: v for k, v in rr.items() if not isinstance(v, dict)},
                    **{f"fixed_conf{t}": v for t, v in rr["rules_fixed"].items()},
                    **{f"reduced_conf{t}": v for t, v in rr["rules_reduced"].items()}})
            print("  done", name, ms, flush=True)
    _write_csv(os.path.join(out_dir, "dataset_stats.csv"), stats_rows)
    _write_csv(os.path.join(out_dir, "results.csv"), records)
    _write_csv(os.path.join(out_dir, "rules_denominator.csv"), rule_rows)

    # selector: calibrate on train datasets, validate on held-out datasets
    td, ts, acc_train = calibrate(records, set(train))
    held = [r for r in records if r["dataset"] in heldout]
    acc_held = accuracy(held, td, ts)
    always_a = sum(r["t_apriori_tid"] <= r["t_fpgrowth"] for r in held) / max(1, len(held))
    always_f = 1 - always_a
    with open(os.path.join(out_dir, "selector.txt"), "w") as f:
        f.write(f"tau_d={td}\ntau_s={ts}\ntrain_accuracy={acc_train}\n"
                f"heldout_accuracy={acc_held}\nalways_apriori_tid_heldout={always_a}\n"
                f"always_fpgrowth_heldout={always_f}\n")
    print("selector:", td, ts, acc_train, acc_held)
    make_charts(records, out_dir)


def _write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def make_charts(records, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted({r["dataset"] for r in records})
    for metric, ylabel, fname, logy in (
            ("t", "Runtime (s)", "runtime", True),
            ("mem", "Peak memory (MB)", "memory", False),
            ("checks", "Candidate-transaction checks", "checks", True)):
        fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 2.8))
        if len(names) == 1:
            axes = [axes]
        for ax, nm in zip(axes, names):
            rs = sorted([r for r in records if r["dataset"] == nm], key=lambda r: r["min_sup"])
            xs = [r["min_sup"] for r in rs]
            keys = ("apriori", "apriori_tid", "fpgrowth") if metric != "checks" else ("apriori", "apriori_tid")
            for key in keys:
                ax.plot(xs, [r[f"{metric}_{key}"] for r in rs], marker="o", label=key)
            ax.set_title(nm, fontsize=9)
            ax.set_xlabel("min support", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            if logy:
                ax.set_yscale("log")
            ax.tick_params(labelsize=7)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname + ".png"), dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "verify":
        verify()
    elif len(sys.argv) >= 4 and sys.argv[1] == "bench":
        bench(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
