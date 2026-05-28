import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


try:
    BASE_DIR = Path(__file__).resolve().parent   # running as a .py script
except NameError:
    BASE_DIR = Path.cwd()                        # running inside a notebook

OUTDIR = BASE_DIR / "output"
OUTDIR.mkdir(parents=True, exist_ok=True)

H   = 52
EPS = 1e-9

TOP_BLACK = ["Node 4", "Node 6", "Node 31"]
HARD      = ["Node 4", "Node 6", "Node 31", "Node 2", "Node 14",
             "Node 23", "Node 28", "Node 20", "Node 30", "Node 12"]
MID_HARD  = ["Node 17", "Node 21", "Node 24", "Node 29",
             "Node 10", "Node 19", "Node 15", "Node 22"]



def find_file(name):
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    for loc in [BASE_DIR / name, OUTDIR / name, Path(name)]:
        if loc.exists():
            return loc
    hits = glob.glob(str(BASE_DIR / "**" / name), recursive=True)
    return Path(hits[0]) if hits else None


def _node_cols(df):
    cols = []
    for c in df.columns:
        parts = str(c).strip().replace("_", " ").split()
        if len(parts) == 2 and parts[0].lower() == "node" and parts[1].isdigit():
            cols.append(c)
    return sorted(cols, key=lambda c: int(str(c).strip().replace("_", " ").split()[1]))


def load_csv(name, expected=None):
    path = find_file(name)
    if path is None:
        raise FileNotFoundError("Cannot find: " + name)
    df   = pd.read_csv(path).loc[:, lambda d: ~d.columns.duplicated()]
    cols = _node_cols(df)
    out  = df[cols].copy()
    out.columns = ["Node %d" % int(c.split()[1]) for c in cols]
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0).clip(lower=0.0)
    if expected:
        for n in expected:
            if n not in out.columns:
                out[n] = 0.0
        out = out[expected]
    return out.iloc[:H].reset_index(drop=True)


def save_csv(df, fname, nodes):
    out = df[nodes].clip(lower=0.0).round(6).reset_index(drop=True)
    out.insert(0, "Id", range(1, H + 1))
    path = OUTDIR / fname
    out.to_csv(path, index=False)
    print("  saved -> " + path.name)
    return path



# Seasonal model


def seasonal_one(series, period=52, scale_window=8, phase=0,
                 trend=0.0, damp=0.0, asym=0.0, robust=False):
    y = np.asarray(series, dtype=float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).clip(0.0)
    n = len(y)
    if n == 0:
        return np.zeros(H)

    y_fit = np.minimum(y, np.percentile(y, 98.5)) if (robust and n >= 30) else y.copy()

    if n <= period + abs(phase):
        template = np.full(H, np.mean(y_fit[-min(8, n):]))
    else:
        template = np.array([
            y_fit[int(np.clip(n - period + (h + phase) % max(1, period), 0, n - 1))]
            for h in range(H)
        ], dtype=float)

    sw     = min(scale_window, n)
    recent = np.mean(y[-sw:]) + EPS
    lag    = max(0, n - period)
    old    = np.mean(y[max(0, lag - sw):lag]) + EPS if lag > 0 else np.mean(y[:sw]) + EPS
    template *= np.clip(recent / old, 0.45, 2.20)

    if trend and n >= 16:
        slope     = (np.mean(y[-min(8, n):]) - np.mean(y[-min(24, n):])) / max(1, min(24, n))
        template += trend * slope * np.arange(1, H + 1)

    if damp:
        t        = np.linspace(0.0, 1.0, H)
        w        = np.clip(1.0 - damp * t, 0.55, 1.15)
        template = w * template + (1.0 - w) * recent

    if asym:
        level           = np.mean(y[-min(6, n):])
        below           = template < level
        template[below] = (1.0 - asym) * template[below] + asym * level

    return template.clip(0.0)


def apply_model(train_df, nodes, params):
    skip = {"name"}
    return pd.DataFrame(
        {n: seasonal_one(train_df[n].values,
                         **{k: v for k, v in params.items() if k not in skip})
         for n in nodes},
        columns=nodes
    ).clip(lower=0.0)



# Candidate bank

def build_candidates():
    cands = []

    for period in [52, 104, 156, 26, 13]:
        for sw in [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30]:
            cands.append({"name": "p%d_s%d" % (period, sw), "period": period, "scale_window": sw})
            if period in [52, 104, 156] and sw in [5, 6, 8, 10, 12, 16]:
                cands.append({"name": "p%d_s%d_rob" % (period, sw), "period": period,
                              "scale_window": sw, "robust": True})

    for phase in range(-6, 7):
        if phase == 0:
            continue
        for period in [52, 104, 156]:
            for sw in [5, 6, 8, 10, 12, 16]:
                cands.append({"name": "p%d_s%d_ph%d" % (period, sw, phase),
                              "period": period, "scale_window": sw, "phase": phase})

    for tr in [0.06, 0.10, 0.15, 0.25, 0.40, 0.55, 0.75, 1.00,
               -0.06, -0.10, -0.15, -0.25, -0.40]:
        for period in [52, 104, 156]:
            for sw in [5, 6, 8, 12, 16]:
                cands.append({"name": "p%d_s%d_tr%s" % (period, sw, tr),
                              "period": period, "scale_window": sw, "trend": tr})

    for dm in [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20,
               -0.02, -0.03, -0.05, -0.08, -0.10]:
        for period in [52, 104, 156]:
            for sw in [5, 6, 8, 12, 16]:
                cands.append({"name": "p%d_s%d_d%s" % (period, sw, dm),
                              "period": period, "scale_window": sw, "damp": dm})

    for asym in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
        for period in [52, 104, 156]:
            for sw in [5, 6, 8, 12, 16]:
                cands.append({"name": "p%d_s%d_asym%s" % (period, sw, asym),
                              "period": period, "scale_window": sw, "asym": asym})

    return cands



# Cross-validated seasonal selection


def cv_select(train_df, nodes, candidates, top_k=3):
    history = train_df.iloc[:-H].reset_index(drop=True)
    holdout = train_df.iloc[-H:].reset_index(drop=True)

    print("    evaluating %d candidates ..." % len(candidates), flush=True)
    val_fc  = [apply_model(history,  nodes, p) for p in candidates]
    test_fc = [apply_model(train_df, nodes, p) for p in candidates]

    out  = pd.DataFrame(index=range(H), columns=nodes, dtype=float)
    rows = []

    for node in nodes:
        true  = holdout[node].values
        errs  = np.array([np.mean(np.abs(f[node].values - true)) for f in val_fc])
        order = np.argsort(errs)[:max(1, min(top_k, len(errs)))]

        if top_k > 1 and len(order) > 1:
            m  = errs[order]
            ww = 1.0 / np.maximum(m, 1e-6) ** 1.25
            ww = ww / ww.sum()
            fc = sum(w * test_fc[j][node].values for j, w in zip(order, ww))
            mae = float((ww * m).sum())
        else:
            j   = int(order[0])
            fc  = test_fc[j][node].values
            mae = float(errs[j])

        out[node] = fc
        rows.append({"Node": node, "CV_MAE": mae})

    report = pd.DataFrame(rows).sort_values("CV_MAE", ascending=False).reset_index(drop=True)
    return out.clip(lower=0.0), report




def clip_to_anchor(pred, anchor, max_frac, max_abs, node_frac=None, node_abs=None):
    out = pred.copy()
    for col in out.columns:
        mf = node_frac[col] if (node_frac and col in node_frac) else max_frac
        ma = node_abs[col]  if (node_abs  and col in node_abs)  else max_abs
        lo = np.maximum(0.0, anchor[col].values * (1.0 - mf) - ma)
        hi = anchor[col].values * (1.0 + mf) + ma
        out[col] = np.clip(out[col].values, lo, hi)
    return out.clip(lower=0.0)


def horizon_nodewise_blend(base, donor, nodes, start_w, end_w,
                           top_mult, hard_mult, mid_mult, cap):
    weights = np.linspace(start_w, end_w, H)
    out     = base.copy()
    for n in nodes:
        ww = weights.copy()
        if n in TOP_BLACK:
            ww *= top_mult
        elif n in HARD:
            ww *= hard_mult
        elif n in MID_HARD:
            ww *= mid_mult
        out[n] = (1.0 - np.clip(ww, 0.0, cap)) * base[n].values + np.clip(ww, 0.0, cap) * donor[n].values
    return out.clip(lower=0.0)


def build_rank_boost(report, nodes, scalar, top_floor, hard_floor, mid_floor=1.0):
    boost = {n: 1.0 for n in nodes}
    maes  = report["CV_MAE"].values.astype(float)
    lo, hi = np.percentile(maes, 20), np.percentile(maes, 95)
    for _, row in report.iterrows():
        score = float(np.clip((row["CV_MAE"] - lo) / (hi - lo + EPS), 0.0, 1.0))
        boost[row["Node"]] = 1.0 + scalar * score
    for n in TOP_BLACK:
        boost[n] = max(boost.get(n, 1.0), top_floor)
    for n in HARD:
        boost[n] = max(boost.get(n, 1.0), hard_floor)
    for n in MID_HARD:
        boost[n] = max(boost.get(n, 1.0), mid_floor)
    return boost


def horizon_rank_blend(base, donor, nodes, rank_boost, start_w, end_w,
                       rank_power=1.0, cap=0.032):
    weights = np.linspace(start_w, end_w, H)
    out     = base.copy()
    for n in nodes:
        mult = rank_boost.get(n, 1.0) ** rank_power
        ww   = np.clip(weights * mult, 0.0, cap)
        out[n] = (1.0 - ww) * base[n].values + ww * donor[n].values
    return out.clip(lower=0.0)


def asym_rank_blend(base, donor, nodes, rank_boost, up_w, down_w,
                    rank_power=1.0, cap=0.022):
    diff = donor - base
    out  = base.copy()
    for n in nodes:
        mult = rank_boost.get(n, 1.0) ** rank_power
        ww   = np.where(diff[n].values >= 0, up_w, down_w) * mult
        ww   = np.clip(ww, 0.0, cap)
        out[n] = base[n].values + ww * diff[n].values
    return out.clip(lower=0.0)




def main():
    print("Loading data ...")
    train = load_csv("Kaggle_Data_Train.csv")
    r24   = load_csv("Round24.csv")
    nodes = list(r24.columns)

    r32 = load_csv(
        "round32_nodewise_cvseason_top0p1_mid0p055_base0p02.csv",
        expected=nodes
    )
    print("Train shape: {}  |  Nodes: {}".format(train.shape, len(nodes)))
    print("Anchor file loaded successfully.")

    print("\nBuilding candidate bank and running CV selection ...")
    candidates       = build_candidates()
    cvseason, report = cv_select(train, nodes, candidates, top_k=3)

    print("\nWorst 5 nodes by CV MAE:")
    for _, row in report.head(5).iterrows():
        print("  %-8s  %.2f" % (row["Node"], row["CV_MAE"]))

    boost_r34 = build_rank_boost(report, nodes, scalar=0.85,
                                 top_floor=1.95, hard_floor=1.45, mid_floor=1.0)
    boost_r35 = build_rank_boost(report, nodes, scalar=0.90,
                                 top_floor=2.05, hard_floor=1.50, mid_floor=1.15)

    # Step 1 — horizon-weighted blend, hard nodes corrected more via tier multipliers
    print("\n[Step 1] Horizon-nodewise blend ...")
    r33 = horizon_nodewise_blend(
        r32, cvseason, nodes,
        start_w=0.004, end_w=0.012,
        top_mult=2.4, hard_mult=1.7, mid_mult=1.25,
        cap=0.055,
    )
    r33 = clip_to_anchor(r33, r32, max_frac=0.022, max_abs=6.0)
    print("  done.")

    # Step 2 — same ramp but node weights come from data-driven rank boost
    print("\n[Step 2] Rank-horizon blend ...")
    r34 = horizon_rank_blend(
        r33, cvseason, nodes, boost_r34,
        start_w=0.0015, end_w=0.0045,
        rank_power=1.0, cap=0.032,
    )
    r34 = clip_to_anchor(r34, r33, max_frac=0.014, max_abs=4.0)
    print("  done.")

    # Step 3 — asymmetric correction, pulls down 3x more readily than pushing up
    print("\n[Step 3] Asymmetric rank blend ...")
    r35 = asym_rank_blend(
        r34, cvseason, nodes, boost_r35,
        up_w=0.001, down_w=0.003,
        rank_power=1.0, cap=0.022,
    )
    r35 = clip_to_anchor(r35, r34, max_frac=0.007, max_abs=2.2)

    out_name = "round35_asym_rank_cvseason_up0p001_down0p003_rp1p0.csv"
    save_csv(r35, out_name, nodes)

    print("\nSanity check (key nodes):")
    for n in ["Node 4", "Node 6", "Node 31"]:
        v = r35[n].values
        print("  %s: w1=%.1f  w13=%.1f  w26=%.1f  w52=%.1f" % (n, v[0], v[12], v[25], v[51]))

    print("\nDone.  Output -> " + str(OUTDIR / out_name))


if __name__ == "__main__":
    main()
