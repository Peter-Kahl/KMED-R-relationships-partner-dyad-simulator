#!/usr/bin/env python3
# fbt_analyse.py
# ------------------------------------------------------------------------------
# KMED-R Toolkit — Fiduciary Boundary Test (FBT) Analyser
# ------------------------------------------------------------------------------
# Supplementary analysis utility for the KMED-R (Relationships) Partner Dyad
# Simulator. Computes the Fiduciary Boundary Test (FBT) — a post-hoc measure
# of epistemic alignment between recognition (ρ), suppression (σ), and fiduciary
# containment (ϕ) over time. The tool classifies each run as fiduciary,
# clientelist, or mixed based on local ratios and thresholds, writing results
# to a compact JSON summary.
#
# Typical use:
#   python tools/fbt_analyse.py ../outputs/KMED-R_fiduciary-partner_20251012_series.json
#
# Outputs:
#   • *_fbt.json — Fiduciary verdict, median ϕ, ρ/σ ratio, and windowed shares
#
# Integration:
#   Part of the KMED-R ecosystem (Lex et Ratio Ltd), this utility complements
#   conceptual simulations in *Epistemic Clientelism in Intimate Relationships*
#   (§ 3.5 Fiduciary Boundary Test) and subsequent computational extensions.
# ------------------------------------------------------------------------------
# Author:          Peter Kahl
# First published: London, 12 October 2025
# Version:         0.9.1 (2025-10-12)
# License:         MIT (see below)
# Repository:      https://github.com/Peter-Kahl/KMED-R-relationships-partner-dyad-simulator
# © 2025 Lex et Ratio Ltd.
# ------------------------------------------------------------------------------
# MIT License
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------

import json, sys, numpy as np, pathlib as p

def rolling_mean(x, w):  # simple safe window
    x = np.asarray(x, float); w = int(w); w = max(1, w)
    pad = w // 2; xp = np.pad(x, (pad, pad), mode='edge')
    k = np.ones(w) / w
    return np.convolve(xp, k, mode='valid')

def main(series_path: str, win=25, phi_hi=0.70, rs_hi=1.50, phi_lo=0.20, rs_lo=0.75):
    sp = p.Path(series_path); ser = json.loads(sp.read_text())
    R = np.array(ser["R"], float); S = np.array(ser["S"], float); PHI = np.array(ser["phi"], float)
    rbar = rolling_mean(R, win); sbar = rolling_mean(S, win); rs = np.divide(rbar, sbar + 1e-9)
    phi_med = rolling_mean(PHI, win)
    fid = (phi_med >= phi_hi) & (rs >= rs_hi)
    cli = (phi_med <= phi_lo) & (rs <= rs_lo)
    verdict = "FIDUCIARY" if fid.mean() > 0.8 else "CLIENTELIST" if cli.mean() > 0.8 else "MIXED"
    out = {"win": win, "phi_med≈": float(np.median(PHI)), "ρ/σ≈": float((R.sum()+1e-9)/(S.sum()+1e-9)),
           "verdict": verdict, "fid_share": float(fid.mean()), "cli_share": float(cli.mean())}
    op = sp.with_name(sp.stem.replace("_series","_fbt") + ".json"); op.write_text(json.dumps(out, indent=2))
    print(op)

if __name__ == "__main__":
    main(sys.argv[1])