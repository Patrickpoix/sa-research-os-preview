# SA Research OS

**Point-in-time quantitative research system for CN A-shares and US equities.**

SA Research OS is built around reproducible research, explicit observation-time semantics, bounded experimentation and statistical validation. The development corpus covers a **543-symbol CN research universe**, **22,597 official disclosure PDFs → 90,388 structured fact rows**, and a **17-year US research window with 3,349 securities and ~10 million point-in-time rows**.

## Core capabilities

- Point-in-time research and availability-aware data handling
- Walk-forward research with explicit train / validation / test boundaries
- Factor research and experiment tracking with bounded trial budgets
- Statistical validation including HAC/Newey-West inference, FDR control, block bootstrap and Deflated Sharpe
- Cost-aware multi-market research and reproducible research artifacts
- Explicit validation states instead of collapsing missing evidence into zero

## Open engineering modules

This repository contains runnable modules from the production research codebase that demonstrate the system's engineering style:

```text
alpha_research_os/
  research/statistics.py     auditable statistical inference primitives
  validation/hash.py        deterministic artifact hashing
  validation/io.py          bounded validation I/O
  validation/reporting.py   machine-readable validation state contract

tests/
  research/                 research-statistics regression tests
  validation/               hash/reporting contract tests
```

The statistical layer keeps assumptions explicit: sample size, annualization, lag selection, trial count and resampling parameters are caller-owned inputs rather than hidden defaults.

## Quick start

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Research approach

```text
question → point-in-time evidence → bounded experiment → statistical validation
         → walk-forward evaluation → retained evidence → research verdict
```

The emphasis is on research that can be reproduced, audited and challenged rather than optimized only for an attractive backtest.
