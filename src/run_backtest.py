"""Étape 4 - Rétro-test : paramètres estimés sur 2024, simulation sur 2025, 30 articles x 3 politiques."""
from pathlib import Path

import pandas as pd

from policies import FixedCover, PeriodicReview, ReorderPointQ, Sku
from simulate import check_balance, initial_stock_for, kpis, metrics, simulate_sku

DATA = Path(__file__).resolve().parents[1] / "data"
START, END = "2025-01-01", "2025-12-31"

master = pd.read_csv(DATA / "sku_master.csv").set_index("sku_id")
seg = pd.read_csv(DATA / "segmentation.csv").set_index("sku_id")
demand = pd.read_csv(DATA / "demand_daily.csv", parse_dates=["date"]).pivot(
    index="date", columns="sku_id", values="demand")
mu = pd.read_csv(DATA / "rolling_mu.csv", index_col=0, parse_dates=True)
sigma = pd.read_csv(DATA / "rolling_sigma.csv", index_col=0, parse_dates=True)

POLICIES = {
    "Couverture fixe (C=3 j)": lambda: FixedCover(3),
    "(s,Q) 95 %, Q=2 j": lambda: ReorderPointQ(0.95, 2),
    "(R,S) 95 %, R=2 j": lambda: PeriodicReview(0.95, 2),
}

rows = []
worst_balance = 0.0
for sku_id, m in master.iterrows():
    sku = Sku(int(m.lead_time_days), int(m.shelf_life_at_receipt), int(m.order_multiple))
    margin = m.unit_price_mad - m.unit_cost_mad
    d, mu_s, sg_s = (x.loc[START:END, sku_id].to_numpy() for x in (demand, mu, sigma))
    init = initial_stock_for(mu_s[0], sku)
    for name, make in POLICIES.items():
        df = simulate_sku(d, mu_s, sg_s, sku, make(), init)
        worst_balance = max(worst_balance, abs(check_balance(df, init)))
        rows.append(dict(sku_id=sku_id, abc=seg.loc[sku_id, "abc"], policy=name,
                         **metrics(df, m.unit_cost_mad, margin)))

res = pd.DataFrame(rows)
print(f"Écart maximal du bilan de stock sur {len(res)} simulations : {worst_balance:.6f} (doit être 0)\n")

summary = []
for (abc, pol), g in res.groupby(["abc", "policy"]):
    totals = {c: g[c].sum() for c in ["demand", "sold", "expired", "received", "stock_sum",
                                      "stockout_days", "cost_total"]}
    totals["days"] = g.days.iloc[0]          # même nombre de jours pour tous les articles
    k = kpis(totals)
    summary.append(dict(classe=abc, politique=pol, service_pct=k["fill_rate"] * 100,
                        pertes_pct=k["waste_rate"] * 100, couverture_j=k["coverage_days"],
                        cout_total_MAD=k["cost_total"]))
summary = pd.DataFrame(summary)
pd.options.display.float_format = "{:,.1f}".format
print(summary.to_string(index=False))
res.to_csv(DATA / "backtest_2025.csv", index=False)
