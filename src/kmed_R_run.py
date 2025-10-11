#!/usr/bin/env python3
# kmed_R_run.py
# ------------------------------------------------------------------------------
# KMED-R (Relationships) — Partner Dyad Simulator
# Conceptual simulations of epistemic intimacy as dynamics over EA, DT, D
# under stylised policies:
#   fiduciary-partner | intermittent-reassurance | avoidant-withholding |
#   coercive-silencing | therapeutic-repair | mutual-growth | sweep
# ------------------------------------------------------------------------------
# Author:          Peter Kahl
# First published: London, 07 October 2025
# Version:         0.9.9 (2025-10-11)
# License:         MIT (see below)
# Repository:      https://github.com/Peter-Kahl/KMED-R-relationships-partner-dyad-simulator
#
# © 2025 Peter Kahl / Lex et Ratio Ltd.
# ------------------------------------------------------------------------------
#
# Quick start (examples):
#   python src/kmed_R_run.py --policy fiduciary-partner --T 160 --tempo slow --smooth
#   python src/kmed_R_run.py --policy intermittent-reassurance --T 200 --tempo slow --seed 7 --smooth
#   python src/kmed_R_run.py --policy avoidant-withholding --T 200 --tempo slow --noise 0.003 --phi 0.30 --pi 0.08 --smooth
#   python src/kmed_R_run.py --policy coercive-silencing --T 160 --tempo slow --phi 0.05 --pi 0.05
#   python src/kmed_R_run.py --policy therapeutic-repair --T 200 --tempo slow --phi 0.70 --pi 0.65 --smooth
#   python src/kmed_R_run.py --policy mutual-growth --T 200 --tempo slow --smooth
#   python src/kmed_R_run.py --policy sweep --sweep_grid 31 --sweep_y suppression --T 120
#
# Arguments:
#   --policy     fiduciary-partner | intermittent-reassurance | avoidant-withholding |
#                coercive-silencing | therapeutic-repair | mutual-growth | sweep
#                (default: fiduciary-partner)
#   --T          number of time steps (default: 160)
#   --seed       RNG seed (default: 42)
#   --noise      Gaussian noise std for state updates (default: 0.005)
#
# Core coefficients (cf. paper §7.2):
#   --alpha      EA sensitivity to (ρ − σ)                    (default: 1.0)
#   --beta       EA sensitivity to (ϕ − D)                    (default: 1.0)
#   --gamma      DT sensitivity to (ϕ + ρ)                    (default: 1.0)
#   --delta      DT erosion by σ                              (default: 1.0)
#   --eps        D growth by (σ − ρ)                          (default: 1.0)
#   --zeta       D reduction by ϕ                             (default: 1.0)
#   --eta        momentum on ΔEA (path-dependency)            (default: 0.0)
#
# Policy overrides (optional):
#   --phi        override fiduciary coefficient ϕ ∈ [0,1]     (default: None → policy-defined)
#   --pi         override repair probability π ∈ [0,1]        (default: None → policy-defined)
#
# Visual/tempo controls:
#   --tempo      slow | medium | fast   (controls segment length; default: medium)
#   --smooth     apply moving-average smoothing to EA/DT/D in plots
#   --smooth_k   smoothing window (odd int; default: 3)
#
# Sweep (qualitative heatmaps of final EA×DT):
#   --sweep_grid odd grid size (e.g., 21 or 31)               (default: 0 = off)
#   --sweep_y    suppression | phi | noise | initEA | initDT  (default: suppression)
#
# Outputs:
#   - For policies:  states plot (...states.png), events plot (...events.png),
#                    run metadata (..._runmeta.json), series (..._series.json).
#   - For sweep:     heatmaps (...heatmaps.png) and run metadata.
#   - Optional raw arrays (.npy) with --save_raw.
# ------------------------------------------------------------------------------
# MIT License (short form)
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# Full license text: https://opensource.org/licenses/MIT
# ------------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import argparse, json, platform, getpass
import numpy as np
import matplotlib.pyplot as plt

__version__ = "0.9.9"

# -------- IO
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- Visual rhythm (set by --tempo)
SEG_LEN   = 24     # steps per segment; set in main()
JITTER    = 0.005  # tiny per-segment jitter; set in main()

# -------- Params & State ------------------------------------------------------

@dataclass
class Params:
    alpha: float = 1.0   # dEA/dt ← (ρ − σ)
    beta:  float = 1.0   # dEA/dt ← (ϕ − D)
    gamma: float = 1.0   # dDT/dt ← (ϕ + ρ)
    delta: float = 1.0   # dDT/dt ← −σ
    eps:   float = 1.0   # dD/dt  ← (σ − ρ)
    zeta:  float = 1.0   # dD/dt  ← −ϕ
    eta:   float = 0.0   # momentum on ΔEA
    noise: float = 0.005 # Gaussian noise

@dataclass
class State:
    EA: float = 0.55
    DT: float = 0.55
    D:  float = 0.45
    R:  float = 0.0
    S:  float = 0.0
    dEA_prev: float = 0.0
    Rc: int = 0  # recognition cooldown
    Sc: int = 0  # suppression cooldown

# -------- Utilities -----------------------------------------------------------

def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))

def bernoulli(p: float) -> int:
    return 1 if np.random.random() < p else 0

def movavg(x: np.ndarray, k: int = 3) -> np.ndarray:
    if k < 2:
        return x
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode='edge')
    kern = np.ones(k, dtype=float) / k
    return np.convolve(xpad, kern, mode='valid')

# -------- Policy schedule -----------------------------------------------------

def policy_profile(policy: str, t: int) -> dict:
    """
    Segment-based regimes: probabilities are piecewise-constant over SEG_LEN steps.
    """
    seg   = t // max(1, int(SEG_LEN))
    phase = seg % 2
    rngR  = np.random.normal(0.0, JITTER)
    rngS  = np.random.normal(0.0, JITTER * 0.8)

    def _clip(p): return float(np.clip(p, 0.02, 0.98))

    # Safe defaults for all keys
    pR, pS = 0.50, 0.20
    phi, pi = 0.50, 0.30
    rc, sc  = 0, 0

    if policy == "fiduciary-partner":
        pR, pS = 0.88 + rngR, 0.06 + rngS
        phi, pi = 0.80, 0.40
        rc, sc  = 2, 2

    elif policy == "intermittent-reassurance":
        if phase == 0:   # warmth
            pR, pS = 0.70 + rngR, 0.18 + rngS
        else:            # withdrawal
            pR, pS = 0.40 + rngR, 0.35 + rngS
        phi, pi = 0.50, 0.30
        rc, sc  = 3, 3

    elif policy == "avoidant-withholding":
        pR, pS = 0.16 + rngR, 0.24 + rngS
        phi, pi = 0.30, 0.08
        rc, sc  = 6, 6

    elif policy == "coercive-silencing":
        pR, pS = 0.06, 0.92
        phi, pi = 0.05, 0.05
        rc, sc  = 1, 1

    elif policy == "therapeutic-repair":
        if phase == 0:
            pR, pS = 0.62 + rngR, 0.22 + rngS
        else:
            pR, pS = 0.54 + rngR, 0.26 + rngS
        phi, pi = 0.70, 0.50
        rc, sc  = 3, 3

    elif policy == "mutual-growth":
        pR, pS = 0.90 + rngR, 0.05 + rngS
        phi, pi = 0.85, 0.55
        rc, sc  = 2, 2

    else:
        raise ValueError(f"Unknown policy: {policy}")

    return {
        "pR":  _clip(pR),
        "pS":  _clip(pS),
        "phi": float(np.clip(phi, 0, 1)),
        "pi":  float(np.clip(pi,  0, 1)),
        "rc":  int(rc),
        "sc":  int(sc),
    }

# -------- Dynamics (cf. §7.2) -------------------------------------------------

def step(state: State, params: Params, policy: str, t: int,
         phi_override: float | None, pi_override: float | None) -> State:
    sched = policy_profile(policy, t)
    pR, pS = sched["pR"], sched["pS"]
    phi = clip01(phi_override if phi_override is not None else sched["phi"])
    pi  = clip01(pi_override  if pi_override  is not None else sched["pi"])
    rc, sc = int(sched.get("rc", 0)), int(sched.get("sc", 0))

    # Cooldowns reduce rapid R/S flipping
    pR_eff = 0.0 if state.Rc > 0 else pR
    pS_eff = 0.0 if state.Sc > 0 else pS

    # Sample one-hot events (recognition first)
    R = bernoulli(pR_eff)
    S = 0 if R == 1 else bernoulli(pS_eff)

    # Update cooldown counters
    Rc_next = max(0, state.Rc - 1)
    Sc_next = max(0, state.Sc - 1)
    if R == 1: Rc_next = rc
    if S == 1: Sc_next = sc

    # Optional repair micro-boosts after S
    repair_DT = 0.0
    repair_EA = 0.0
    if S == 1 and bernoulli(pi):
        repair_DT = 0.04 * params.gamma * (0.5 + 0.5 * phi)
        repair_EA = 0.02 * params.alpha * (0.5 + 0.5 * phi)

    # Core kinetics
    dEA = params.alpha * (R - S) + params.beta * (phi - state.D)
    if params.eta != 0.0:
        dEA += params.eta * state.dEA_prev

    dDT = params.gamma * (phi + R) - params.delta * S
    dD  = params.eps   * (S - R)   - params.zeta  * phi

    # Small conceptual jitter
    nEA = np.random.normal(0.0, params.noise)
    nDT = np.random.normal(0.0, params.noise)
    nD  = np.random.normal(0.0, params.noise)

    EA = clip01(state.EA + dEA + repair_EA + nEA)
    DT = clip01(state.DT + dDT + repair_DT + nDT)
    D  = clip01(state.D  + dD  + nD)

    return State(EA=EA, DT=DT, D=D, R=float(R), S=float(S),
                 dEA_prev=dEA, Rc=Rc_next, Sc=Sc_next)

# -------- Simulation ----------------------------------------------------------

def run_sim(policy: str, T: int, seed: int, params: Params,
            phi: float | None, pi: float | None):
    np.random.seed(seed)
    s = State()
    series = {k: [] for k in ("EA","DT","D","R","S","phi","pi")}
    for t in range(T):
        sched = policy_profile(policy, t)
        phi_eff = clip01(phi if phi is not None else sched["phi"])
        pi_eff  = clip01(pi  if pi  is not None else sched["pi"])
        s = step(s, params, policy, t, phi_override=phi, pi_override=pi)

        series["EA"].append(s.EA)
        series["DT"].append(s.DT)
        series["D"].append(s.D)
        series["R"].append(s.R)
        series["S"].append(s.S)
        series["phi"].append(phi_eff)
        series["pi"].append(pi_eff)
    return series

# -------- Heatmap sweep (qualitative) ----------------------------------------

def sweep_heatmap(T: int, seed: int, params: Params, grid: int = 21, mode: str = "suppression"):
    """
    x-axis: recognition pR in [0.10..0.90]
    y-axis: varies by mode
      - suppression : pS in [0.05..0.95]
      - phi         : fiduciary coefficient ϕ in [0.00..1.00]
      - noise       : state-update noise σ in [0.00..0.03]
      - initEA      : initial EA₀ in [0.10..0.90] (DT fixed at 0.55)
      - initDT      : initial DT₀ in [0.10..0.90] (EA fixed at 0.55)
    Computes final EA, DT surfaces using a neutral policy backbone.
    """
    np.random.seed(seed)
    xs = np.linspace(0.10, 0.90, grid)
    if mode == "suppression":
        ys = np.linspace(0.05, 0.95, grid); y_label = "Suppression pS"
    elif mode == "phi":
        ys = np.linspace(0.00, 1.00, grid); y_label = "Fiduciary coefficient ϕ"
    elif mode == "noise":
        ys = np.linspace(0.00, 0.03, grid); y_label = "Noise σ"
    elif mode == "initEA":
        ys = np.linspace(0.10, 0.90, grid); y_label = "Initial EA₀"
    elif mode == "initDT":
        ys = np.linspace(0.10, 0.90, grid); y_label = "Initial DT₀"
    else:
        raise ValueError(f"Unknown sweep_y mode: {mode}")

    EA_end = np.zeros((grid, grid))
    DT_end = np.zeros((grid, grid))

    base_phi = 0.55
    base_pi  = 0.35

    for i, pR in enumerate(xs):
        for j, y in enumerate(ys):
            p = Params(**asdict(params))
            s = State()
            if mode == "initEA":
                s.EA = float(y)
            if mode == "initDT":
                s.DT = float(y)
            if mode == "noise":
                p.noise = float(y)

            if mode == "suppression":
                pS, phi = float(y), base_phi
            elif mode == "phi":
                pS, phi = 0.30, float(y)
            else:
                pS, phi = 0.30, base_phi

            np.random.seed(seed)  # deterministic per cell
            for t in range(T):
                R = bernoulli(pR)
                S = 0 if R == 1 else bernoulli(pS)

                repair_DT = 0.0
                repair_EA = 0.0
                if S == 1 and bernoulli(base_pi):
                    repair_DT = 0.04 * p.gamma * (0.5 + 0.5*phi)
                    repair_EA = 0.02 * p.alpha * (0.5 + 0.5*phi)

                dEA = p.alpha * (R - S) + p.beta * (phi - s.D)
                dDT = p.gamma * (phi + R) - p.delta * S
                dD  = p.eps   * (S - R)   - p.zeta  * phi

                if p.eta != 0.0:
                    dEA += p.eta * s.dEA_prev

                nEA = np.random.normal(0.0, p.noise)
                nDT = np.random.normal(0.0, p.noise)
                nD  = np.random.normal(0.0, p.noise)

                EA = clip01(s.EA + dEA + repair_EA + nEA)
                DT = clip01(s.DT + dDT + repair_DT + nDT)
                D  = clip01(s.D  + dD  + nD)
                s = State(EA=EA, DT=DT, D=D, R=float(R), S=float(S), dEA_prev=dEA)

            EA_end[j, i] = s.EA
            DT_end[j, i] = s.DT

    x_label = "Recognition pR"
    return xs, ys, EA_end, DT_end, x_label, y_label

# -------- Plotting ------------------------------------------------------------

def stamp_meta(ax, meta: dict, loc="lower right", fontsize=8):
    from matplotlib.offsetbox import AnchoredText
    lines = [
        f'{meta["version"]}  •  {meta["timestamp"]}',
        f'policy={meta["policy"]}, T={meta["T"]}, seed={meta["seed"]}',
        f'α={meta["alpha"]:.2f}, β={meta["beta"]:.2f}, γ={meta["gamma"]:.2f}, δ={meta["delta"]:.2f}',
        f'ε={meta["eps"]:.2f}, ζ={meta["zeta"]:.2f}, η={meta["eta"]:.2f}, noise={meta["noise"]:.3f}',
        f'ϕ={meta.get("phi","–")}, π={meta.get("pi","–")}'
    ]
    at = AnchoredText("\n".join(lines), prop=dict(size=fontsize),
                      frameon=True, loc=loc, borderpad=0.6)
    at.patch.set_alpha(0.45)
    ax.add_artist(at)

def plot_series(series: dict, meta: dict, out_prefix: Path):
    t = np.arange(len(series["EA"]))

    # Optional smoothing
    do_smooth = bool(meta.get("smooth", False))
    k = int(meta.get("smooth_k", 3))
    EA = np.array(series["EA"])
    DT = np.array(series["DT"])
    D  = np.array(series["D"])
    if do_smooth:
        k = k if k % 2 == 1 else k + 1
        EA = movavg(EA, k)
        DT = movavg(DT, k)
        D  = movavg(D,  k)

    # Stacked: top = EA/DT; bottom = D
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10.6, 7.2), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1]})
    ax_top.plot(t, EA, label="EA (autonomy)")
    ax_top.plot(t, DT, label="DT (tolerance)")
    ax_top.set_ylabel("State (0–1)")
    ax_top.set_title(f'KMED-R [{meta["policy"]}]: Epistemic states over time')
    ax_top.legend(loc="lower left")

    ax_bot.plot(t, D, label="D (dependence)")
    ax_bot.set_xlabel("Time")
    ax_bot.set_ylabel("Dependence (0–1)")
    ax_bot.legend(loc="upper left")

    plt.tight_layout()
    stamp_meta(ax_bot, meta, loc="lower right", fontsize=8)
    plt.savefig(out_prefix.with_suffix(".states.png"), dpi=220)
    plt.close()

    # Events + policy levels
    fig, ax1 = plt.subplots(figsize=(10.6, 5.8))
    ax1.step(t, series["R"], where="post", label="R (recognition)", linewidth=1.2)
    ax1.step(t, series["S"], where="post", label="S (suppression)", linewidth=1.2)
    ax1.set_xlabel("Time"); ax1.set_ylabel("Event (0/1)")
    ax2 = ax1.twinx()
    ax2.plot(t, series["phi"], alpha=0.65, label="ϕ (policy)", linestyle="--")
    ax2.plot(t, series["pi"],  alpha=0.65, label="π (policy)", linestyle=":")
    ax2.set_ylabel("Policy levels")

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labs1+labs2, loc="upper right")
    plt.title(f'KMED-R [{meta["policy"]}]: R/S events with ϕ & π profiles')
    plt.tight_layout()
    stamp_meta(ax1, meta, loc="lower left", fontsize=8)
    plt.savefig(out_prefix.with_suffix(".events.png"), dpi=220)
    plt.close()

def plot_heatmaps(xs, ys, EA_end, DT_end, out_prefix: Path, meta: dict, x_label: str, y_label: str, title_suffix: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    for ax, Z, title in zip(axes, [EA_end, DT_end], ["EA (final)", "DT (final)"]):
        im = ax.imshow(Z, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="auto")
        ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.9)
    plt.suptitle(f"KMED-R heatmaps: final EA and DT — {title_suffix or y_label}")
    stamp_meta(axes[-1], meta, loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".heatmaps.png"), dpi=220)
    plt.close()

# -------- Metadata ------------------------------------------------------------

def build_meta(script_name: str, policy: str, T: int, seed: int, params: Params,
               phi: float | None, pi: float | None):
    md = {
        "script": script_name,
        "version": __version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user": getpass.getuser(),
        "python": platform.python_version(),
        "policy": policy,
        "T": int(T),
        "seed": int(seed),
        **asdict(params)
    }
    if phi is not None: md["phi"] = float(phi)
    if pi  is not None: md["pi"]  = float(pi)
    return md

# -------- Main ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="KMED-R (Relationships) partner-dyad simulator — conceptual scaffolding"
    )
    ap.add_argument("--policy",
        choices=["fiduciary-partner","intermittent-reassurance","avoidant-withholding",
                 "coercive-silencing","therapeutic-repair","mutual-growth","sweep"],
        default="fiduciary-partner")
    ap.add_argument("--T",     type=int,   default=160)
    ap.add_argument("--seed",  type=int,   default=42)
    ap.add_argument("--noise", type=float, default=0.005)

    # coefficients (cf. §7.2)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta",  type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--eps",   type=float, default=1.0)
    ap.add_argument("--zeta",  type=float, default=1.0)
    ap.add_argument("--eta",   type=float, default=0.0)

    # policy overrides
    ap.add_argument("--phi", type=float, default=None, help="Override fiduciary coefficient ϕ ∈ [0,1]")
    ap.add_argument("--pi",  type=float, default=None, help="Override repair probability π ∈ [0,1]")

    # sweep
    ap.add_argument("--sweep_grid", type=int, default=0)
    ap.add_argument("--sweep_y",
        choices=["suppression","phi","noise","initEA","initDT"],
        default="suppression",
        help="Which y-axis to sweep (default: suppression).")

    # tempo/visuals
    ap.add_argument("--tempo", choices=["slow","medium","fast"], default="medium",
                    help="Controls segment length and visual rhythm (slow=cleanest).")
    ap.add_argument("--smooth", action="store_true",
                    help="Apply a small moving-average to EA/DT/D before plotting.")
    ap.add_argument("--smooth_k", type=int, default=3,
                    help="Window (odd) for moving-average when --smooth is set (default: 3).")

    # raw
    ap.add_argument("--save_raw", action="store_true")

    args = ap.parse_args()

    # set global visual rhythm
    global SEG_LEN, JITTER
    SEG_LEN = {"slow": 36, "medium": 24, "fast": 12}[args.tempo]
    JITTER = 0.005

    params = Params(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta,
        eps=args.eps, zeta=args.zeta, eta=args.eta, noise=args.noise
    )

    phi_override = None if args.phi is None else clip01(args.phi)
    pi_override  = None if args.pi  is None else clip01(args.pi)

    if args.policy != "sweep":
        series = run_sim(args.policy, args.T, args.seed, params, phi_override, pi_override)
        meta = build_meta("kmed_R_run.py", args.policy, args.T, args.seed, params, phi_override, pi_override)

        daystamp = datetime.now().strftime("%Y%m%d")
        prefix = OUTPUT_DIR / f"KMED-R_{args.policy}_{daystamp}"
        (OUTPUT_DIR / f"KMED-R_{args.policy}_{daystamp}_runmeta.json").write_text(json.dumps(meta, indent=2))
        (OUTPUT_DIR / f"KMED-R_{args.policy}_{daystamp}_series.json").write_text(json.dumps(series))

        if args.save_raw:
            for k, v in series.items():
                np.save(OUTPUT_DIR / f"KMED-R_{args.policy}_{daystamp}_{k}.npy", np.array(v))

        # include smoothing flags for plot stamp
        plot_series(series, meta | {"smooth": bool(args.smooth), "smooth_k": int(args.smooth_k)}, prefix)

    else:
        grid = int(args.sweep_grid) if args.sweep_grid else 21
        xs, ys, EA_end, DT_end, xlab, ylab = sweep_heatmap(args.T, args.seed, params, grid=grid, mode=args.sweep_y)
        meta = build_meta("kmed_R_run.py", "sweep", args.T, args.seed, params, phi_override, pi_override)
        daystamp = datetime.now().strftime("%Y%m%d")
        prefix = OUTPUT_DIR / f"KMED-R_SWEEP_{daystamp}_{args.sweep_y}"
        (OUTPUT_DIR / f"KMED-R_SWEEP_{daystamp}_{args.sweep_y}_runmeta.json").write_text(json.dumps(meta, indent=2))
        if args.save_raw:
            np.save(OUTPUT_DIR / f"KMED-R_SWEEP_{daystamp}_{args.sweep_y}_EA_end.npy", EA_end)
            np.save(OUTPUT_DIR / f"KMED-R_SWEEP_{daystamp}_{args.sweep_y}_DT_end.npy", DT_end)
        plot_heatmaps(xs, ys, EA_end, DT_end, prefix, meta, xlab, ylab, title_suffix=args.sweep_y)

if __name__ == "__main__":
    main()