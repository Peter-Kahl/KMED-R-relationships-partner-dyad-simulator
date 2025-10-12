# KMED-R (Relationships): Partner Dyad Simulator

[![Generic badge](https://img.shields.io/badge/ORCID-0009.0003.1616.4843-green.svg)](https://orcid.org/0009-0003-1616-4843)

![Two hands holding a delicate origami bird above a small flame of a candle. The bird is fragile yet emerging from fire.](https://github.com/Peter-Kahl/KMED-R-relationships-partner-dyad-simulator/blob/main/origami.jpg?raw=true)

KMED-R (Relationships) formalises epistemic dynamics within intimate partnerships and other dyadic relationships.
It extends the _Kahl Model of Epistemic Dissonance_ (KMED) from developmental to interpersonal psychology, showing how recognition, suppression, repair, and fiduciary quality shape epistemic life.

The model captures three interacting epistemic variables:
- **Epistemic Autonomy (EA)** – persistence in signalling contradiction
- **Dissonance Tolerance (DT)** – capacity to withstand contradiction
- **Dependence (D)** – reliance on suppression versus recognition

KMED-R thus provides a computational theatre for exploring epistemic clientelism, resilience, and trust, bridging intimate relationships with broader institutional and organisational applications.

### 📚 Table of Contents

1.	[Documentation & Conceptual Background](#1-documentation--conceptual-background)
2.	[Requirements](#2-requirements)
3.	[Installation](#3-installation)
4.	[Quick Start](#4-quick-start)
5.	[CLI Reference](#5-cli-reference)
6.	[Scenario Cheat-Sheet (§7.3)](#6-scenario-cheat-sheet--73)
7.	[Figure Glossary](#7-figure-glossary)
8.	[Tips for Clean Visuals](#8-tips-for-clean-visuals)
9.	[License](#9-license)
10.	[Citation](#10-citation)

---

## 1. Documentation & Conceptual Background

This repository contains the official Python implementation of _KMED-R (Relationships): Partner Dyad Simulator_, reproducing the simulations presented in the paper:

> Kahl, P. (2025). _Epistemic Clientelism in Intimate Relationships: The Kahl Model of Epistemic Dissonance (KMED) and the Foundations of Epistemic Psychology_. Lex et Ratio Ltd. GitHub: https://github.com/Peter-Kahl/Epistemic-Clientelism-in-Intimate-Relationships DOI: https://doi.org/10.13140/RG.2.2.33790.45122

Each simulation represents a stylised relational policy—fiduciary, inconsistent, avoidant, coercive, reparative, or mutual—modelled as a sequence of recognition (ρ), suppression (σ), fiduciary containment (ϕ), and repair (π) events.

These are not empirical data but qualitative epistemic archetypes, illustrating how ethical care modulates autonomy, tolerance, and dependence.

---

## 2. Requirements

- Python 3.9+
- `numpy`
- `matplotlib`

Install dependencies via:

```bash
# install dependencies
pip install -r requirements.txt
# or manually:
pip install numpy matplotlib
```

---

## 3. Installation

Clone the repository:

```bash
git clone https://github.com/Peter-Kahl/KMED-R-relationships-partner-dyad-simulator.git
cd KMED-R-relationships-partner-dyad-simulator/src
```

---

## 4. Quick Start

```bash
# Fiduciary baseline (secure, trust-rich)
python kmed_R_run.py --policy fiduciary-partner --T 160 --tempo slow --smooth

# Intermittent reassurance (oscillating warmth/withdrawal)
python kmed_R_run.py --policy intermittent-reassurance --T 200 --tempo slow --smooth

# Avoidant withholding (cool distance)
python kmed_R_run.py --policy avoidant-withholding --T 200 --tempo slow --phi 0.30 --pi 0.08 --smooth

# Coercive silencing (punitive suppression)
python kmed_R_run.py --policy coercive-silencing --T 160 --tempo slow --phi 0.05 --pi 0.05

# Therapeutic repair (rupture → restoration)
python kmed_R_run.py --policy therapeutic-repair --T 200 --tempo slow --phi 0.70 --pi 0.65 --smooth

# Mutual growth (reciprocal autonomy)
python kmed_R_run.py --policy mutual-growth --T 200 --tempo slow --smooth

# Surface mapping (EA×DT heatmaps)
python kmed_R_run.py --policy sweep --sweep_grid 31 --sweep_y suppression --T 120

# Fiduciary vs Clientelist trajectories (Figure 8.1)
python kmed_R_run.py --make_figure bifurcation --T 160 --tempo slow --smooth

# Recognition/Suppression event traces
python kmed_R_run.py --make_figure bifurcation-events --T 160 --tempo slow

# Epistemic Silencing trajectory (Figure 8.3)
python kmed_R_run.py --make_figure silencing --T 160 --tempo slow --smooth
```

Outputs appear in the `/outputs/` directory:

- `*_states.png` – stacked EA/DT and D trajectories
- `*_events.png` – recognition/suppression events and policy levels (ϕ, π)
- `*_series.json` – time-series data
- `*_runmeta.json` – parameters, metadata, and provenance
- `*_heatmaps.png` – final EA/DT surfaces (sweep only)
- `*_bifurcation.png` - composite comparison of fiduciary vs. clientelist trajectories (Figure 8.1 in Epistemic Clientelism in Intimate Relationships).
- `*_bifurcation_events.png` - side-by-side R/S event panels for both paths, illustrating recognition–suppression asymmetry.
- `*_silencing.png` - time-series plot of EA, DT, and D under coercive-silencing policy, including the computed Silencing Index (S) overlay (Figure 8.3).
- `*.npy` - optional raw numerical arrays for analytical reuse when invoked with `--save_raw`.

---

## 5. CLI Reference

| Flag	| Type / Range	| Default	|	Description	|
| -------- | ------- | -------- | -------- |
| `--policy` | fiduciary-partner \| intermittent-reassurance \| avoidant-withholding \| coercive-silencing \| therapeutic-repair \| mutual-growth \| sweep | fiduciary-partner | Selects relational policy or simulation mode |
| `--T` | int ≥ 1 | 160 | Number of simulation time steps |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--noise` | float ≥ 0 | 0.005 | Gaussian noise applied to state updates |
| `--alpha` | float | 1.0 | EA sensitivity to (ρ − σ) |
| `--beta` | float | 1.0 | EA sensitivity to (ϕ − D) |
| `--gamma` | float | 1.0 | DT sensitivity to (ϕ + ρ) |
| `--delta` | float | 1.0 | DT erosion by σ |
| `--eps` | float | 1.0 | D growth by (σ − ρ) |
| `--zeta` | float | 1.0 | D reduction by ϕ |
| `--eta` | float | 0.0 | Momentum on ΔEA (path-dependence) |
| `--phi` | float [0,1] or None | None | Override fiduciary coefficient ϕ (policy-defined if None) |
| `--pi` | float [0,1] or None | None | Override repair probability π (policy-defined if None) |
| `--tempo` | slow \| medium \| fast | medium | Controls segment length and visual rhythm |
| `--smooth` | flag | off | Apply moving-average smoothing to EA/DT/D |
| `--smooth_k` | odd int ≥ 3 | 3 | Window size for smoothing (when `--smooth` is set) |
| `--sweep_grid` | odd int | 0 | Grid size for parameter-sweep heatmaps (e.g. 21, 31) |
| `--sweep_y` | suppression \| phi \| noise \| initEA \| initDT | suppression | Y-axis variable for sweep mode |
| `--make_figure` | bifurcation \| bifurcation-events \| silencing | None | Generates composite figures (Figures 8.1, 8.3) |
| `--bif_policies` | str pair A,B | fiduciary-partner,coercive-silencing | Two policies to compare in bifurcation plots |
| `--bif_seeds` | int pair A,B | None (defaults to `--seed`) | Two random seeds for comparison runs |
| `--bif_phi` | float pair [0,1],[0,1] | None | Optional ϕ overrides for A,B |
| `--bif_pi` | float pair [0,1],[0,1] | None | Optional π overrides for A,B |
| `--save_raw` | flag | off | Save raw numerical arrays (.npy) for analysis


---

## 6. Scenario Cheat-Sheet

| Section | Policy | Essence | CLI Example |
| -------- | ------- | -------- | -------- |
| § 7.3.1 | Fiduciary-Partner | Stable, trust-rich reciprocity; epistemic analogue of secure attachment | `--policy fiduciary-partner --tempo slow --smooth` |
| § 7.3.2 | Intermittent-Reassurance | Oscillating warmth and withdrawal; autonomy and dependence alternate | `--policy intermittent-reassurance --T 200 --tempo slow --smooth` |
| § 7.3.3 | Avoidant-Withholding | Muted, low-recognition environment; trust underdeveloped | `--policy avoidant-withholding --phi 0.30 --pi 0.08 --smooth` |
| § 7.3.4 | Coercive-Silencing | Punitive suppression; dependence saturates, autonomy collapses | `--policy coercive-silencing --phi 0.05 --pi 0.05` |
| § 7.3.5 | Therapeutic-Repair | Rupture–repair dynamic; autonomy partially restored | `--policy therapeutic-repair --phi 0.70 --pi 0.65 --smooth` |
| § 7.3.6 | Mutual-Growth | High recognition, low suppression, shared autonomy | `--policy mutual-growth --tempo slow --smooth` |
| § 7.3.7 | Surface-Mapping | Recognition × suppression sweep; fiduciary plateau vs. clientelist basin | `--policy sweep --sweep_grid 31 --sweep_y suppression --T 120` |
| § 8.1 | Bifurcation (Figure 8.1 — Recognition–Suppression Bifurcation) | Simulation output corresponding to the vignette in § 8.1. The same dissonance event (δ) yields two outcomes: a fiduciary path (ρ > σ, ϕ high) enabling epistemic repair (EA ↑, DT ↑, D ↓) and a clientelist path (σ > ρ, ϕ low) leading to epistemic collapse (EA ↓, DT ↓, D ↑). | `python kmed_R_run.py --make_figure bifurcation --T 160 --tempo slow --smooth` |
| § 8.4 | Epistemic Silencing (Figure 8.3 — The Trajectory of Epistemic Silencing) | Modelled simulation under the coercive-silencing policy (σ ≫ ρ, low ϕ, π). Over time, EA and DT collapse toward 0 as D → 1. The Silencing Index (S) tracks the cumulative futility of expression — the system stabilises not through reconciliation but through epistemic paralysis. | `python kmed_R_run.py --make_figure silencing --T 160 --tempo slow --smooth` |

---

## 7. Figure Glossary

**States Plot (`*_states.png`)**\
Stacked subplots for clarity:
- _Top_: EA (autonomy) and DT (tolerance)
- _Bottom_: D (dependence)\
Optional smoothing (`--smooth`) applies a centred moving average with window `--smooth_k`.

**Events Plot (`*_events.png`)**\
Step series of R (recognition) and S (suppression) events, with dashed/dotted overlays for policy parameters ϕ and π.

**Heatmaps (`*_heatmaps.png`)**\
Generated in sweep mode. Final EA and DT values mapped over recognition × Y-parameter grid.\
Reveals the fiduciary plateau (stability) and clientelist basin (collapse) that bound all regimes.

---

## 8. Tips for Clean Visuals

- Use `--tempo` slow for publication-grade plots (minimal flicker).
- Add `--smooth` for conceptual clarity; adjust `--smooth_k` (3–7) for gentler trends.
- If R/S events appear too “busy,” lower `--noise` (e.g., 0.003) or lengthen `--T`.
- For deterministic replication, fix `--seed`.

---

## 9. License

- Code is released under the MIT License (see LICENSE).
- Accompanying paper and documentation are released under Creative Commons [BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

You may freely use, adapt, and extend the code for research and educational purposes. Please cite appropriately.

---

## 10. Citation

Please cite the paper and optionally the repository release tag:

> Kahl, P. (2025). _Epistemic Clientelism in Intimate Relationships: The Kahl Model of Epistemic Dissonance (KMED) and the Foundations of Epistemic Psychology_. Lex et Ratio Ltd. GitHub: https://github.com/Peter-Kahl/Epistemic-Clientelism-in-Intimate-Relationships DOI: https://doi.org/10.13140/RG.2.2.33790.45122

and

> Kahl, P. (2025). _KMED-R (Relationships): Partner Dyad Simulator_ (Version v1.0-preprint) [Computer software]. Lex et Ratio Ltd. GitHub: https://github.com/Peter-Kahl/KMED-R-relationships-partner-dyad-simulator/releases/tag/v1.0-preprint