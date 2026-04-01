# Robust Portfolio Analytics and Model Governance under Distribution Shift

This project packages a portfolio-analytics notebook, reusable research modules, and a repeatable CLI workflow around a tractable Wasserstein-inspired robust optimization proxy. The current version is designed to be more honest and more reproducible than the original draft: fixed data horizon, cached raw prices, soft-feasibility diagnostics, window-correct stress summaries, and explicit monitoring limitations.

## Paper alignment

- The main allocator is a **Wasserstein-inspired proxy**, not an exact implementation of the general Wasserstein DRO programs in the theory papers.
- The noisy-data section is a **corruption-aware stress-testing extension**, not an implementation of a convolution-based noisy-observation ambiguity set.
- The optional Kelly-style appendix is built on **log-return samples** and is explicitly presented as a tractable proxy, not the paper's exact Wasserstein-Kelly convex reformulation.

## What changed

- The robust allocator is now labeled as a **Wasserstein proxy** instead of a full DRO implementation.
- The optimizer uses a **soft return constraint with slack** instead of frequent hard-feasibility failures.
- The backtest now includes **no-trade bands, partial execution blending, epsilon smoothing, and composite hyperparameter scoring** so the robust strategy behaves more like an operational research system than a one-shot optimizer.
- The project uses a **fixed end date** and reads from `data/raw_prices.parquet` by default.
- Stress summaries now filter both **returns and governance metrics** to the requested window.
- Monitoring is framed as a **prototype governance workflow**, with separate targets and calibration metrics.
- The regime block is framed as a **heuristic market-state tagging layer**, not a standalone predictive ML result.

## Layout

```text
.
|- Robust Portfolio Analytics and Model Governance under Distribution Shift.ipynb
|- configs/
|  |- base.yaml
|- data/
|  |- README.md
|- outputs/
|- scripts/
|  |- generate_notebook.py
|  |- run_backtest.py
|- src/
|  |- __init__.py
|  |- backtest.py
|  |- baselines.py
|  |- config.py
|  |- corruption.py
|  |- data.py
|  |- features.py
|  |- monitoring.py
|  |- regime.py
|  |- reporting.py
|  |- robust.py
|  |- validation.py
|- tests/
|- environment.yml
|- pyproject.toml
`- requirements.txt
```

## Quick start

1. Create an environment with one of the following:

```bash
conda env create -f environment.yml
conda activate robust-portfolio-analytics
```

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
pip install -e .
```

2. Run `python scripts/run_backtest.py --config configs/base.yaml` to generate CSV, JSON, weights, figures, and regime diagnostics under `outputs/cli_run/`.
3. Open `Robust Portfolio Analytics and Model Governance under Distribution Shift.ipynb` and run it top to bottom. The notebook metadata is set to the `Python (robust-portfolio-analytics)` kernel.

## Notes

- The workflow is wired to reuse `data/raw_prices.parquet` and `data/large_universe_raw_prices.parquet` if they are already present in the working archive, and to refresh them only on request.
- `data/large_universe_raw_prices.parquet` is used for the larger-universe notebook extension when available.
- The notebook is generated from `scripts/generate_notebook.py`, so code and narrative stay aligned.
- The monitoring section is intentionally modest: it is a governance-oriented instability prototype, not an alpha model.
- The regime-classification extension is also intentionally modest: it labels heuristic market states from trailing diagnostics and uses ML only to support conditional governance analysis.
