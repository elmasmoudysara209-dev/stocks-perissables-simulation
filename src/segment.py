from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

master = pd.read_csv(DATA / "sku_master.csv")
demand = pd.read_csv(DATA / "demand_daily.csv", parse_dates=["date"])
d24 = demand[demand.date.dt.year == 2024]

stats = d24.groupby("sku_id").demand.agg(mu="mean", sigma="std")
stats["cv"] = stats.sigma / stats.mu
stats = stats.join(master.set_index("sku_id")[["sku_name", "family", "unit_cost_mad"]])
stats["annual_value"] = stats.mu * 366 * stats.unit_cost_mad

def abc_classes(value):
    v = value.sort_values(ascending=False)
    cum_before = v.cumsum().shift(fill_value=0) / v.sum()
    cls = np.where(cum_before < 0.80, "A", np.where(cum_before < 0.95, "B", "C"))
    return pd.Series(cls, index=v.index)

def xyz_classes(cv):
    return pd.Series(np.where(cv <= 0.40, "X", np.where(cv <= 0.50, "Y", "Z")), index=cv.index)

stats["abc"] = abc_classes(stats.annual_value)
stats["xyz"] = xyz_classes(stats.cv)
stats["classe"] = stats.abc + stats.xyz
stats = stats.sort_values("annual_value", ascending=False)

print(stats[["sku_name", "mu", "sigma", "cv", "annual_value", "abc", "xyz", "classe"]].round(2).to_string())
print()
print(stats.abc.value_counts().sort_index().to_string())
print(stats.xyz.value_counts().sort_index().to_string())
print()
print(pd.crosstab(stats.abc, stats.xyz))
share = stats.groupby("abc").annual_value.sum() / stats.annual_value.sum()
print((share * 100).round(1).to_string())
stats.to_csv(DATA / "segmentation.csv", index_label="sku_id")
