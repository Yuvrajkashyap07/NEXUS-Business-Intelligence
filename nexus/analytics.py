"""Machine-learning and forecasting functions used by NEXUS."""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def _build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop")


def _feature_summary(pipe: Pipeline, model_name: str) -> pd.DataFrame:
    try:
        names = pipe.named_steps["prep"].get_feature_names_out()
        model = pipe.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            values = np.abs(np.ravel(model.coef_))
        else:
            return pd.DataFrame(columns=["feature", "importance"])
        result = pd.DataFrame({"feature": names, "importance": values})
        return result.sort_values("importance", ascending=False).head(15)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


def run_regression(df: pd.DataFrame, target: str, features: list[str], model_name: str = "Random Forest") -> dict[str, Any]:
    data = df[[target] + features + (["date"] if "date" in df.columns and "date" not in features else [])].copy()
    data = data.dropna(subset=[target])
    if len(data) < 25:
        raise ValueError("Regression needs at least 25 complete target rows. Add more historical observations.")
    X = data[features].copy()
    y = pd.to_numeric(data[target], errors="coerce")
    valid = y.notna()
    X, y, data = X.loc[valid], y.loc[valid], data.loc[valid]
    if len(data) < 25:
        raise ValueError("The selected target must be numeric and have at least 25 valid rows.")
    if "date" in data.columns and data["date"].notna().sum() > len(data) * .7:
        order = np.argsort(pd.to_datetime(data["date"]).values)
        X, y = X.iloc[order], y.iloc[order]
        split = max(1, int(len(X) * .8))
        X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    if len(X_test) < 2:
        raise ValueError("Not enough test rows. Add more observations or use a larger dataset.")
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Ridge Regression":
        model = Ridge(alpha=1.0)
    else:
        model = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", _build_preprocessor(X_train)), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    return {
        "model": model_name, "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)), "rmse": rmse,
        "actual": y_test.reset_index(drop=True), "predicted": pd.Series(pred),
        "features": _feature_summary(pipe, model_name), "rows": len(data),
        "test_rows": len(X_test), "split": "Time-ordered 80/20" if "date" in data.columns else "Random 80/20",
    }


def run_arima(series: pd.Series, periods: int = 6, order: tuple[int, int, int] = (2, 1, 2)) -> dict[str, Any]:
    from statsmodels.tsa.arima.model import ARIMA
    y = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(y) < 18:
        raise ValueError("ARIMA needs at least 18 observations. Use a longer history or a coarser time grain.")
    if (y < 0).any():
        raise ValueError("Sales units cannot be negative for this forecasting view.")
    # Hold out the last 20% as a simple chronological backtest. This is more
    # informative for forecasting than a shuffled train/test split.
    split = max(12, int(len(y) * .8))
    backtest = {}
    if len(y) - split >= 2:
        try:
            train, test = y.iloc[:split], y.iloc[split:]
            test_fit = ARIMA(train, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
            test_pred = test_fit.get_forecast(steps=len(test)).predicted_mean
            test_pred = np.maximum(test_pred.to_numpy(), 0)
            actual = test.to_numpy()
            mae = float(mean_absolute_error(actual, test_pred))
            rmse = float(np.sqrt(mean_squared_error(actual, test_pred)))
            wape = float(np.abs(actual - test_pred).sum() / max(np.abs(actual).sum(), 1e-9) * 100)
            naive_pred = np.repeat(float(train.iloc[-1]), len(test))
            naive_wape = float(np.abs(actual - naive_pred).sum() / max(np.abs(actual).sum(), 1e-9) * 100)
            backtest = {"mae": mae, "rmse": rmse, "wape_pct": wape, "naive_wape_pct": naive_wape, "test_periods": len(test)}
        except Exception:
            backtest = {}

    fitted = ARIMA(y, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
    forecast_obj = fitted.get_forecast(steps=periods)
    mean = forecast_obj.predicted_mean.clip(lower=0)
    ci = forecast_obj.conf_int()
    lower = ci.iloc[:, 0].clip(lower=0)
    upper = ci.iloc[:, 1].clip(lower=0)
    index = pd.RangeIndex(1, periods + 1, name="period")
    forecast = pd.DataFrame({"period": index, "forecast": mean.to_numpy(), "lower_95": lower.to_numpy(), "upper_95": upper.to_numpy()})
    return {"forecast": forecast, "fitted_aic": float(fitted.aic), "observations": len(y), "order": order, "backtest": backtest}


def run_classification(df: pd.DataFrame, target: str, features: list[str], model_name: str = "Random Forest") -> dict[str, Any]:
    data = df[[target] + features].dropna(subset=[target]).copy()
    y = data[target].astype(str)
    counts = y.value_counts()
    keep = counts[counts >= 2].index
    data = data[y.isin(keep)]
    y = data[target].astype(str)
    if len(data) < 30 or y.nunique() < 2:
        raise ValueError("Classification needs at least 30 usable rows and two classes with repeated examples.")
    X = data[features]
    stratify = y if y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=stratify)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1500, class_weight="balanced")
    else:
        model = RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", _build_preprocessor(X_train)), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, pred, labels=labels)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    return {"accuracy": float(accuracy_score(y_test, pred)), "labels": labels,
            "confusion": pd.DataFrame(cm, index=labels, columns=labels),
            "report": pd.DataFrame(report).T, "features": _feature_summary(pipe, model_name),
            "rows": len(data), "model": model_name}


def run_clustering(df: pd.DataFrame, features: list[str], clusters: int = 4) -> dict[str, Any]:
    data = df[features].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.select_dtypes(include=np.number)
    data = data.dropna(axis=1, how="all")
    if data.shape[1] < 2:
        raise ValueError("Choose at least two numeric features for clustering.")
    data = data.fillna(data.median()).dropna()
    if len(data) < max(20, clusters * 3):
        raise ValueError("Not enough usable rows for the selected number of clusters.")
    sample = data.sample(min(len(data), 10000), random_state=42) if len(data) > 10000 else data
    scaler = StandardScaler()
    scaled = scaler.fit_transform(sample)
    model = KMeans(n_clusters=clusters, n_init=10, random_state=42)
    labels = model.fit_predict(scaled)
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
    plot = pd.DataFrame({"component_1": coords[:, 0], "component_2": coords[:, 1], "cluster": labels.astype(str)})
    profile = sample.assign(cluster=labels).groupby("cluster")[list(data.columns)].mean().reset_index()
    return {"plot": plot, "profile": profile, "inertia": float(model.inertia_), "rows": len(sample), "features": list(data.columns)}


def run_association_rules(transactions: pd.DataFrame, transaction_col: str, product_col: str,
                          min_support: float = .025, min_confidence: float = .25,
                          min_lift: float = 1.0) -> pd.DataFrame:
    data = transactions[[transaction_col, product_col]].dropna().astype(str)
    baskets = data.groupby(transaction_col)[product_col].apply(lambda x: sorted(set(x))).tolist()
    baskets = [set(b) for b in baskets if len(set(b)) > 0]
    n = len(baskets)
    if n < 10 or sum(1 for basket in baskets if len(basket) >= 2) < 5:
        raise ValueError("Association mining needs at least 10 transactions and at least 5 multi-product baskets.")
    item_counts: dict[str, int] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    for basket in baskets:
        for item in basket:
            item_counts[item] = item_counts.get(item, 0) + 1
        for pair in combinations(sorted(basket), 2):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    rows = []
    for (a, b), count in pair_counts.items():
        support = count / n
        if support < min_support:
            continue
        for antecedent, consequent in [(a, b), (b, a)]:
            confidence = count / max(item_counts[antecedent], 1)
            consequent_support = item_counts[consequent] / n
            lift = confidence / max(consequent_support, 1e-12)
            if confidence >= min_confidence and lift >= min_lift:
                rows.append({"if_bought": antecedent, "likely_also_bought": consequent,
                             "support": support, "confidence": confidence, "lift": lift,
                             "transactions": count})
    if not rows:
        return pd.DataFrame(columns=["if_bought", "likely_also_bought", "support", "confidence", "lift", "transactions"])
    return pd.DataFrame(rows).sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
