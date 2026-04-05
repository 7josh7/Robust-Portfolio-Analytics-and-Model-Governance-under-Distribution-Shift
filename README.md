# Robust Portfolio Analytics and Model Governance under Distribution Shift

This project packages a portfolio-analytics notebook, reusable research modules, and a repeatable CLI workflow around two complementary robust-allocation branches: a tractable Wasserstein-inspired proxy baseline and a paper-aligned DR mean-variance branch. The current version is designed to be more honest and more reproducible than the original draft: fixed data horizon, cached raw prices, soft-feasibility diagnostics, window-correct stress summaries, and explicit monitoring limitations.

## Paper alignment

- The main allocator is a **Wasserstein-inspired proxy**, not an exact implementation of the general Wasserstein DRO programs in the theory papers.
- The DR mean-variance branch is closer to the Blanchet-Chen-Zhou regularized formulation, but it is still implemented as a practical research workflow rather than the paper's full inference stack.
- The noisy-data section is a **corruption-aware stress-testing extension**, not an implementation of a convolution-based noisy-observation ambiguity set.
- The regime engine now includes both a **lightweight mixture benchmark** and a **two-state HMM-style market-factor branch**, which is closer to Costa-Kwon without claiming their full factor-model implementation.
- The optional Kelly-style appendix now includes both a **log-return proxy** and a closer-to-paper **exact-`p=2` Wasserstein-Kelly branch**.

## What changed

- The robust allocator is now labeled as a **Wasserstein proxy** instead of a full DRO implementation.
- A new **DRMV regularized min-variance branch** has been added so the project can compare a practical proxy against a closer-to-paper robust allocation model.
- A **paper-reference DRMV calibration mode** is available alongside the tuned production mode so the notebook can compare practical versus literature-facing calibration.
- The optimizer uses a **soft return constraint with slack** instead of frequent hard-feasibility failures.
- The backtest now includes **no-trade bands, partial execution blending, epsilon smoothing, and composite hyperparameter scoring** so the robust strategy behaves more like an operational research system than a one-shot optimizer.
- The project now includes **mixture** and **HMM-style regime-conditioned DRMV variants**, with covariance-conditioned versions called out explicitly.
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
|  |- covariance.py
|  |- data.py
|  |- features.py
|  |- monitoring.py
|  |- regime.py
|  |- reporting.py
|  |- robust.py
|  |- selection.py
|  |- targets.py
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
- The new regime-conditioned allocation inputs are separate from that heuristic tagging block: they now support both a lightweight mixture benchmark and a two-state HMM-style market-factor engine for the DRMV branch.
- A frozen control snapshot of the previous fast/stable results is stored under `outputs/baseline_snapshots/v_current_fast_stable/`.
