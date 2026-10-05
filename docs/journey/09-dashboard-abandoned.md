# 09 — Streamlit Dashboard (Planned & Abandoned)

> **Status:** abandoned · **Effect:** 0.0 · **Date:** 2026-09-07
>
> ← [08-toggles-and-combo.md](08-toggles-and-combo.md) · [timeline](README.md) · [next](10-vocab-upsweep.md) →

## Question
Should we build a live Streamlit dashboard to monitor training trajectories and sweep metrics in real-time?

## Reasoning
Phase 9 was originally planned to create an interactive visualization tool (`tools/dashboard.py`) for live loss curves, learning rate tracking, and automated multi-run comparisons directly from structured JSON log files.

## Approach
An implementation plan was drafted (`docs/_source/plans/2026-09-07-phase9-dashboard.md`) covering:
- Streamlit application reading log files from `mainrun/logs/`.
- Dynamic Altair / Plotly charts for train/val loss curves.
- Multi-run overlay comparison tables.

## What we learned & Why it was abandoned
- **Zero impact on ML performance:** The challenge objective is purely to minimize validation loss within 7 epochs. Building and maintaining a UI added zero predictive power.
- **Workflow reality on remote GPU pods:** All experiments were orchestrated headless over SSH using lightweight Python driver scripts and analyzed with `analyze.py`. Opening browser ports and serving web UI from ephemeral RunPod instances introduced friction without benefit.
- **Decision:** The dashboard code (`tools/`) was completely removed from the repository, and UI verification tasks were dropped to keep the codebase lean and focused entirely on ML training dynamics.

## Open threads
- Address the vocabulary size question prior to advanced optimizer work ([10-vocab-upsweep.md](10-vocab-upsweep.md)).

## Sources
- Original plan: `docs/_source/plans/2026-09-07-phase9-dashboard.md`
- Decision record: Git commits `ee57d84` ("Remove the Streamlit dashboard entirely") and `aab648b`.
