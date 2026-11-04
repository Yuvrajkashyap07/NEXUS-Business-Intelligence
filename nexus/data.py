"""Data utilities and reproducible demo data for NEXUS."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_data(seed: int = 41, days: int = 420) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create realistic, synthetic operational data plus transaction baskets."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    products = [
        ("SKU-101", "Smart Home", 149.0, 79.0, 31),
        ("SKU-205", "Audio", 89.0, 42.0, 24),
        ("SKU-310", "Wearables", 219.0, 118.0, 18),
        ("SKU-422", "Accessories", 39.0, 13.0, 46),
        ("SKU-518", "Home Office", 179.0, 91.0, 15),
        ("SKU-630", "Gaming", 299.0, 177.0, 12),
    ]
    regions = ["Dubai", "Abu Dhabi", "Sharjah"]
    rows: list[dict] = []
    for sku, category, price, unit_cost, base in products:
        for region in regions:
            region_factor = {"Dubai": 1.22, "Abu Dhabi": 0.92, "Sharjah": 0.78}[region]
            product_phase = rng.uniform(0, 2 * np.pi)
            for idx, date in enumerate(dates):
                weekend = 1.14 if date.dayofweek in (4, 5) else 0.94
                annual = 1 + 0.13 * np.sin(2 * np.pi * idx / 365.25 + product_phase)
                campaign = int((idx % 56) in range(18, 25) or (idx % 91) in range(55, 62))
                discount = float(rng.choice([0, 0, 0, 5, 10, 15], p=[.34, .20, .12, .14, .12, .08])) if campaign else 0.0
                marketing_spend = max(0, rng.normal(230 + 185 * campaign + base * 2.1, 75))
                price_realised = price * (1 - discount / 100)
                demand_mean = base * region_factor * weekend * annual * (1 + 0.028 * discount) * (1 + 0.00065 * marketing_spend)
                sales = int(max(0, rng.poisson(max(demand_mean, 1))))
                # A snapshot-like inventory position; deliberately creates all risk classes.
                cover_days = float(rng.choice([2.2, 4.8, 8.5, 13.0, 19.0], p=[.12, .18, .25, .27, .18]))
                inventory = max(0, int(sales * cover_days + rng.normal(0, max(2, sales * .55))))
                incoming = int(rng.choice([0, 0, 0, 20, 40, 80], p=[.34, .18, .12, .14, .12, .10]))
                lead_time = int(max(2, rng.normal(13 if region == "Dubai" else 16, 4)))
                coverage = inventory / max(sales, 1)
                risk = "High" if coverage < 3.5 else ("Medium" if coverage < 8 else "Low")
                rows.append({
                    "date": date,
                    "product": sku,
                    "category": category,
                    "region": region,
                    "marketing_spend": round(marketing_spend, 2),
                    "price": round(price_realised, 2),
                    "list_price": price,
                    "discount_pct": discount,
                    "sales_units": sales,
                    "inventory_on_hand": inventory,
                    "incoming_units": incoming,
                    "lead_time_days": lead_time,
                    "unit_cost": unit_cost,
                    "stockout_risk": risk,
                    "campaign_active": "Yes" if campaign else "No",
                })
    sales_df = pd.DataFrame(rows)

    # Synthetic basket data is separate because transaction-level data has a different grain.
    basket_rng = np.random.default_rng(seed + 100)
    basket_products = [p[0] for p in products]
    base_weights = np.array([.18, .20, .12, .24, .13, .13])
    transactions: list[dict] = []
    for tx in range(1, 3201):
        n_items = int(basket_rng.choice([1, 2, 3, 4], p=[.22, .43, .27, .08]))
        # Add repeat co-purchase patterns to make association rules discoverable.
        if basket_rng.random() < .36:
            pair = basket_rng.choice([[0, 3], [1, 5], [2, 4], [0, 4]], p=[.32, .28, .22, .18])
            chosen = list(pair)
            while len(chosen) < n_items:
                extra = int(basket_rng.choice(len(basket_products), p=base_weights))
                if extra not in chosen:
                    chosen.append(extra)
        else:
            chosen = list(basket_rng.choice(len(basket_products), size=min(n_items, len(basket_products)), replace=False, p=base_weights))
        for item_idx in chosen:
            transactions.append({"transaction_id": f"TX-{tx:05d}", "product": basket_products[int(item_idx)]})
    tx_df = pd.DataFrame(transactions)
    return sales_df, tx_df


def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise common header variations to the app's canonical column names."""
    out = df.copy()
    out.columns = [str(c).strip().lower().replace("%", "pct").replace(" ", "_").replace("-", "_") for c in out.columns]
    aliases = {
        "sales": "sales_units", "units_sold": "sales_units", "quantity_sold": "sales_units",
        "qty_sold": "sales_units", "units": "sales_units", "revenue_units": "sales_units",
        "sku": "product", "product_id": "product", "item": "product", "product_name": "product",
        "order_id": "transaction_id", "invoice_id": "transaction_id", "basket_id": "transaction_id",
        "date_time": "date", "order_date": "date", "sales_date": "date", "timestamp": "date",
        "stock": "inventory_on_hand", "on_hand": "inventory_on_hand", "current_stock": "inventory_on_hand",
        "incoming_stock": "incoming_units", "purchase_cost": "unit_cost", "cost": "unit_cost",
        "selling_price": "price", "ad_spend": "marketing_spend", "marketing_budget": "marketing_spend",
        "discount": "discount_pct", "lead_time": "lead_time_days", "supplier_lead_time": "lead_time_days",
    }
    out = out.rename(columns={c: aliases[c] for c in out.columns if c in aliases})
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["sales_units", "marketing_spend", "price", "list_price", "discount_pct", "inventory_on_hand", "incoming_units", "lead_time_days", "unit_cost"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # A user may map a non-standard source header to a canonical field in Data Studio.
    # Apply type coercion again after mapping so downstream time-series / ML code is safe.
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["sales_units", "marketing_spend", "price", "list_price", "discount_pct", "inventory_on_hand", "incoming_units", "lead_time_days", "unit_cost"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "stockout_risk" not in out.columns and {"inventory_on_hand", "sales_units"}.issubset(out.columns):
        cover = out["inventory_on_hand"].fillna(0) / out["sales_units"].fillna(0).clip(lower=1)
        out["stockout_risk"] = np.select([cover < 3.5, cover < 8], ["High", "Medium"], default="Low")
    if "revenue" not in out.columns and {"sales_units", "price"}.issubset(out.columns):
        out["revenue"] = out["sales_units"].fillna(0) * out["price"].fillna(0)
    if "gross_profit" not in out.columns and {"sales_units", "price", "unit_cost"}.issubset(out.columns):
        out["gross_profit"] = out["sales_units"].fillna(0) * (out["price"].fillna(0) - out["unit_cost"].fillna(0))
    return out


def expected_sales_columns() -> list[str]:
    return ["date", "product", "category", "region", "marketing_spend", "price", "discount_pct", "sales_units", "inventory_on_hand", "incoming_units", "lead_time_days", "unit_cost"]


def expected_transaction_columns() -> list[str]:
    return ["transaction_id", "product"]
