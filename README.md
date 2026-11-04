# NEXUS — Unified Business Intelligence & Decision Platform

A GitHub-ready Streamlit application that connects Marketing, Supply Chain, Finance and Digital Business Management (DBM) through one evidence-led workflow.

## What is included

- **Overview:** operating dashboard and shared departmental decision loop.
- **Data Studio:** upload operational CSVs, map source columns, inspect completeness and download templates.
- **Forecast Lab:** ARIMA demand forecasts, confidence intervals and comparison against the Marketing 40% / 30% lecture scenario.
- **Algorithm Lab:**
  - Regression — sales / revenue drivers and test metrics (R², MAE, RMSE).
  - Classification — risk labels, confusion matrix and class-level precision / recall / F1.
  - Clustering — K-Means segments with PCA visualisation and profiles.
  - Association Rule Mining — product affinity using support, confidence and lift.
- **Decision Hub:** demand input → stock-aware procurement quantity → Finance budget / margin check → recorded human decision.
- **Governance & Export:** RACI, KPI catalogue, model governance and downloadable ZIP decision pack.
- **Demo data:** synthetic operational sales and transaction-basket datasets so the app works immediately without an upload.

## Run locally

Python 3.10–3.13 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy through GitHub + Streamlit Community Cloud

1. Extract this ZIP file.
2. Create a new GitHub repository, for example `nexus-business-intelligence`.
3. Upload the **contents** of this folder to the repository root. `streamlit_app.py`, `requirements.txt` and `.streamlit/config.toml` must be at the repository root / expected paths.
4. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
5. Choose **Create app**, select your repository and branch, and set the main file path to `streamlit_app.py`.
6. Select **Deploy**. Streamlit installs the dependencies from `requirements.txt` and publishes the app at a `streamlit.app` address.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Data format

### Operational sales CSV

The main file should ideally contain one row per date × product × region. Recommended fields:

| Field | Meaning | Required for |
|---|---|---|
| `date` | Observation / sales date | ARIMA and weekly demand |
| `product` | SKU or product identifier | Product planning and segmentation |
| `sales_units` | Units sold / demand | Regression and forecasting |
| `marketing_spend` | Campaign expenditure | Demand-driver regression |
| `price` | Selling price | Regression and margin checks |
| `discount_pct` | Discount percentage | Regression |
| `inventory_on_hand` | Usable stock | Procurement / risk analysis |
| `incoming_units` | Confirmed incoming supply | Procurement calculation |
| `lead_time_days` | Supplier replenishment lead time | Supply Chain assessment |
| `unit_cost` | Purchase cost per unit | Finance assessment |
| `category`, `region` | Product / market dimensions | Segmentation and analysis |

The Data Studio includes column mapping, so source headers do not have to match every canonical name. Date, product and sales units are the minimum required fields for core operation.

### Transaction basket CSV

Association Rule Mining needs a separate transaction-grain file with one row per product in a basket:

```csv
transaction_id,product
TX-001,SKU-101
TX-001,SKU-422
TX-002,SKU-205
```

A daily product-sales file is not a substitute for transaction-level basket data.

## Algorithm use

| Algorithm | NEXUS business use | Output |
|---|---|---|
| Regression | Estimate relationships between demand and price, promotion, marketing and product attributes | Predicted continuous value, R², MAE, RMSE, feature signals |
| ARIMA | Forecast future weekly sales from historical time series | Forecast and model-derived interval |
| Classification | Categorise stockout / business risk | Predicted class, accuracy, confusion matrix, precision / recall |
| Clustering | Find similar customers, products or demand observations | Segment assignments and profiles |
| Association Rule Mining | Identify products that co-occur in baskets | Support, confidence and lift |

## Important demo / production boundary

- The bundled sample data is synthetic and is only for demonstrating the application.
- Marketing's lecture inputs (40%, 30%, R² 0.7 and ±2,000) are treated as assumptions to review, not verified model results.
- R² is not forecast accuracy. ±2,000 is not treated as a confidence interval until its definition is confirmed.
- Algorithms provide decision support; they do not automatically approve procurement or financial expenditure.
- Decision history is currently stored in Streamlit session state and is not a durable shared audit log. Production use requires authentication, role-based access, a database, durable audit logging, data retention controls, monitoring and security review.
- Do not upload confidential or personal data to a public demonstration deployment.

## Quick analytics smoke tests

```bash
python -m unittest discover -s tests
```

## Project structure

```text
NEXUS_Business_Intelligence/
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── tests/
│   └── test_analytics.py
├── nexus/
│   ├── __init__.py
│   ├── data.py
│   └── analytics.py
└── data/
    ├── demo_sales.csv
    └── demo_transactions.csv
```
