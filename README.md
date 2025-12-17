# Retail Forecasting Admin Dashboard (Streamlit)

A Streamlit-based admin dashboard for scenario-driven retail sales forecasting and automated planning report generation (PDF) using a local SLM (Ollama).

## What this app does
- Operations dashboard: KPIs + sales trends + actual vs predicted charts (per-store selection)
- Forecasting & planning: admin inputs → forecast generation → persisted JSON artifact
- Report generation: forecast JSON → prompt template → local SLM (Ollama) → downloadable PDF
- Model insights: dataset + model + explainability overview loaded from config files

## Tech stack
- UI: Streamlit + Plotly
- Model execution: PyTorch TorchScript (`flowx_jit.pt`)
- Local SLM: Ollama (`/api/generate`)
- PDF generation: ReportLab
- Data/config: JSON + CSV

## Project structure
.
├── streamlit_app.py
├── config/
│ ├── dashboard_meta.json
│ ├── kpis.json
│ ├── prompt_template.txt
│ └── model.json
├── data/
│ ├── sales_over_time.csv
│ └── actual_vs_predicted.csv
├── artifacts/
│ ├── flowx_jit.pt
│ ├── model.pt
│ └── latest_forecast.json
└── requirements.txt



## How forecasting works (high level)
- Admin enters scenario parameters (store, family, item, promotion, oil price, holiday description).
- Inputs are transformed into model-compatible tensors that match training feature dimensions.
- Predictions are generated for a fixed horizon (16 days).
- Output is saved to artifacts/latest_forecast.json.
- Report generation reads latest_forecast.json only (single source of truth).
- Reporting (SLM → PDF) ==> The forecast JSON is injected into config/prompt_template.txt.
- The prompt is sent to Ollama’s HTTP API.
- The SLM response is rendered and exported as PDF.