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

## 📚 Table of Contents
1.	[Documentation & Conceptual Background](#1-documentation--conceptual-background)
2.	[Requirements](#2-requirements)
3.	[Installation](#3-installation)
4.	[Quick Start](#4-quick-start)
5.	[CLI Reference](#5-cli-reference)
6.	[Scenario Cheat-Sheet (§7.3)]()
7.	[Figure Glossary]()
8.	[Tips for Clean Visuals]()
9.	[License]()
10.	[Citation]()

---

## 1. Documentation & Conceptual Background

This repository contains the official Python implementation of KMED-R (Relationships): Partner Dyad Simulator, reproducing the simulations presented in the paper:

> Kahl, P. (2025). _Epistemic Clientelism in Intimate Relationships: The Kahl Model of Epistemic Dissonance (KMED) and the Foundations of Epistemic Psychology_. Lex et Ratio Ltd. https://github.com/Peter-Kahl/Epistemic-Clientelism-in-Intimate-Relationships

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
```

Outputs appear in the /outputs/ directory:
- `*_states.png` – stacked EA/DT and D trajectories
- `*_events.png` – recognition/suppression events and policy levels (ϕ, π)
- `*_series.json` – time-series data
- `*_runmeta.json` – parameters, metadata, and provenance
- `*_heatmaps.png` – final EA/DT surfaces (sweep only)

---

## 5. CLI Reference

| Flag	| Type / Range	| Default	|	Description	|
| -------- | ------- | -------- | -------- |
| --policy | one of fiduciary-partner, intermittent-reassurance, avoidant-withholding, coercive-silencing, therapeutic-repair, mutual-growth, sweep | fiduciary-partner | Selects relational regime |
| --T | int ≥ 1 | 160 | Number of time steps |
| --seed | int | 42 | Random seed |
| --noise | float ≥ 0 | 0.005 | Gaussian noise on updates |
| --phi | float [0,1]\|None | None | Override fiduciary coefficient ϕ |
| --pi | float [0,1]\|None | None | Override repair probability π |
| --tempo | slow|medium|fast | medium | Temporal rhythm and visual smoothness |
| --smooth | flag | off | Apply small moving-average to EA/DT/D |
| --smooth_k | odd int ≥ 3 | 3 | Smoothing window |
| --sweep_grid | odd int | 0 | Grid size for heatmap sweeps |
| --sweep_y | suppression\|phi\|noise\|initEA\|initDT | suppression | Y-axis parameter in sweeps |
| --save_raw | flag | off | Save raw .npy arrays |

---

## 6. Scenario Cheat-Sheet (§ 7.3)

| Section | Policy | Essence | CLI Example |
| -------- | ------- | -------- | -------- |
| § 7.3.1 | Fiduciary-Partner | Stable, trust-rich reciprocity; epistemic analogue of secure attachment | --policy fiduciary-partner --tempo slow --smooth |
| § 7.3.2 | Intermittent-Reassurance | Oscillating warmth and withdrawal; autonomy and dependence alternate | --policy intermittent-reassurance --T 200 --tempo slow --smooth |
| § 7.3.3 | Avoidant-Withholding | Muted, low-recognition environment; trust underdeveloped | --policy avoidant-withholding --phi 0.30 --pi 0.08 --smooth |
| § 7.3.4 | Coercive-Silencing | Punitive suppression; dependence saturates, autonomy collapses | --policy coercive-silencing --phi 0.05 --pi 0.05 |
| § 7.3.5 | Therapeutic-Repair | Rupture–repair dynamic; autonomy partially restored | --policy therapeutic-repair --phi 0.70 --pi 0.65 --smooth |
| § 7.3.6 | Mutual-Growth | High recognition, low suppression, shared autonomy | --policy mutual-growth --tempo slow --smooth |
| § 7.3.7 | Surface-Mapping | Recognition × suppression sweep; fiduciary plateau vs. clientelist basin | --policy sweep --sweep_grid 31 --sweep_y suppression --T 120 |

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
- [Accompanying paper and documentation](https://github.com/Peter-Kahl/The-Newborns-First-Cry-as-Epistemic-Claim-and-Foundation-of-Psychological-Development) are released under Creative Commons BY-NC-ND 4.0.

You may freely use, adapt, and extend the code for research and educational purposes. Please cite appropriately.

---

## 10. Citation

Please cite the paper and optionally the repository release tag:

> Kahl, P. (2025). _Epistemic Clientelism in Intimate Relationships: The Kahl Model of Epistemic Dissonance (KMED) and the Foundations of Epistemic Psychology_. Lex et Ratio Ltd. https://github.com/Peter-Kahl/Epistemic-Clientelism-in-Intimate-Relationships

and

> Kahl, P. (2025). KMED-R (Relationships): Partner Dyad Simulator (Version v1.0-preprint) [Computer software]. Lex et Ratio Ltd. GitHub. https://github.com/Peter-Kahl/KMED-R-relationships-partner-dyad-simulator/releases/tag/v1.0-preprint