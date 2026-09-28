"""Yield-anomaly labels, data splits and the msIndex regressor."""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "rain_mm", "rain_flowering_mm", "water_balance_mm", "water_balance_flowering_mm",
    "dry_spell_days", "heat_stress_days", "greenup_doy",
    "ndvi_peak", "ndvi_integral", "ndvi_anomaly",
]
SPLIT_TRAIN = "Training (in-sample)"
SPLIT_TEST = "Test (out-of-sample)"
SPLIT_CURRENT = "Current season (forecast)"


def is_final(df: pd.DataFrame) -> pd.Series:
    return df["yield_tha"].notna() & df["yield_status"].astype(str).str.startswith("final")


def add_yield_anomaly(df: pd.DataFrame, test_start: int) -> pd.DataFrame:
    """Yield anomaly = reported yield vs a per-province linear technology trend.

    The trend is fitted on training seasons only, so test and current seasons carry no look-ahead.
    """
    df = df.copy()
    prov = df[is_final(df) & (df["season"] < test_start)].drop_duplicates(["province", "season"])
    coefs = {p: np.polyfit(g["season"], g["yield_tha"], 1) for p, g in prov.groupby("province")}
    df["yield_trend"] = [np.polyval(coefs[p], s) for p, s in zip(df["province"], df["season"])]
    df["yield_anomaly"] = (df["yield_tha"] / df["yield_trend"] - 1).round(4)
    return df


def assign_split(df: pd.DataFrame, test_start: int) -> pd.Series:
    final = is_final(df)
    return pd.Series(
        np.select([final & (df["season"] < test_start), final], [SPLIT_TRAIN, SPLIT_TEST], SPLIT_CURRENT),
        index=df.index,
    )


def make_model(seasons: pd.Series | None = None):
    """Ridge regression with its penalty chosen by leave-one-season-out CV.

    The districts of a province share one yield label, so rows within a season are not independent;
    holding out whole seasons keeps the penalty choice from rewarding memorised provincial labels.
    """
    cv = None
    if seasons is not None and seasons.nunique() >= 3:
        cv = list(LeaveOneGroupOut().split(np.zeros(len(seasons)), groups=seasons.to_numpy()))
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30), cv=cv))


def fit_model(rows: pd.DataFrame):
    return make_model(rows["season"]).fit(rows[FEATURE_COLS], rows["yield_anomaly"])


def fill_features(df: pd.DataFrame, test_start: int | None = None) -> pd.DataFrame:
    """Green-up is occasionally undetectable (persistent cloud); fall back to the district's usual date.

    The usual date is taken from training seasons only, so a test season never borrows from its own era.
    """
    df = df.copy()
    ref = df if test_start is None else df[df["season"] < test_start]
    by_district = df["district"].map(ref.groupby("district")["greenup_doy"].median())
    df["greenup_doy"] = df["greenup_doy"].fillna(by_district).fillna(ref["greenup_doy"].median())
    return df


def _rescaled(model, fit_rows: pd.DataFrame, rows: pd.DataFrame) -> tuple[np.ndarray, float]:
    """A regularised model shrinks its predictions toward the mean, which is right for accuracy but wrong
    for an index: a payout ladder reading in yield-shortfall units would almost never reach its lower
    steps. Rescaling to the spread of the rows the model was fitted on keeps the ranking and makes the
    index mean what it says. Each model is rescaled with its own fitted predictions."""
    fitted = model.predict(fit_rows[FEATURE_COLS])
    scale = fit_rows["yield_anomaly"].std() / fitted.std() if fitted.std() > 0 else 1.0
    pred = fit_rows["yield_anomaly"].mean() + (model.predict(rows[FEATURE_COLS]) - fitted.mean()) * scale
    return pred, float(scale)


def score(df: pd.DataFrame, test_start: int) -> pd.DataFrame:
    """Adds pred_yield_anomaly.

    Training and test seasons are scored by a model fitted on training seasons only (honest evaluation).
    The current season is scored by a model refitted on every season with a final yield.
    """
    df = add_yield_anomaly(fill_features(df, test_start), test_start)
    df["Data split"] = assign_split(df, test_start)
    train = df[df["Data split"] == SPLIT_TRAIN]
    labelled = df[df["Data split"] != SPLIT_CURRENT]
    current = (df["Data split"] == SPLIT_CURRENT).to_numpy()

    eval_pred, eval_scale = _rescaled(fit_model(train), train, df)
    prod_pred, prod_scale = _rescaled(fit_model(labelled), labelled, df)
    df["pred_yield_anomaly"] = np.round(np.where(current, prod_pred, eval_pred), 4)
    df["index_scale"] = np.where(current, round(prod_scale, 3), round(eval_scale, 3))
    df["pred_yield_tha"] = (df["yield_trend"] * (1 + df["pred_yield_anomaly"])).round(2)
    return df
