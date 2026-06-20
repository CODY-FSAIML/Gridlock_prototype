# Ravis _AI.control — Event Operations Command Center

Ravis _AI.control helps traffic-control teams plan for congestion caused by planned and unplanned events. It converts an event scenario into an explainable operational recommendation: closure risk, disruption severity, suggested officers, barricades, diversion points, and staging time.

## The problem

Political rallies, festivals, sports events, construction work, accidents, and sudden gatherings can create local gridlock. Today, resource deployment is largely experience-driven and post-event learning is limited.

## What the prototype does

- Accepts event, timing, location, crowd, duration, and lane-impact inputs.
- Uses a CatBoost model trained on 8,173 historical Bengaluru incident records to estimate road-closure risk.
- Converts the risk and operational conditions into a practical deployment plan.
- Shows location context, comparable historical records, and plain-language reasoning.
- Captures post-event feedback during the session to demonstrate the learning loop.

## Model validation

Evaluation uses a chronological future-period holdout, not a random split. This is important because it better represents real deployment.

| Metric | Result |
| --- | ---: |
| F1 | 0.3953 |
| Recall | 0.5155 |
| Precision | 0.3205 |
| PR-AUC | 0.3020 |
| ROC-AUC | 0.8024 |

Road closures are only about 8% of the dataset, so F1, recall, precision, and PR-AUC are used instead of accuracy as the primary measures.

## Run locally

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Repository contents

- `app.py` — Streamlit command-center dashboard
- `train_model.py` — reproducible CatBoost training and evaluation
- `event_closure_model.cbm` — trained model used by the app
- `gridlock_dataset.csv` — supplied historical event dataset
- `model_metrics.json` — saved evaluation metrics

## Responsible-use note

Ravis _AI.control is a decision-support prototype. Its resource recommendations are starting points for a traffic control room, which retains final authority based on the live ground situation.
