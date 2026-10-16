"""Étape 5 - Expériences Monte Carlo.

Pour chaque réplication : on tire une nouvelle demande 2025 à partir d'un modèle ajusté sur 2024,
puis on fait tourner TOUTES les configurations de politiques sur cette même demande
(comparaison équitable : la différence vient de la politique, pas de la chance).

Usage :  python src/experiments.py 10      (essai rapide, 10 réplications)
         python src/experiments.py 100     (version finale)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from policies import FixedCover, PeriodicReview, ReorderPointQ, Sku
from simulate import initial_stock_for, metrics, simulate_sku

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT / "data", ROOT / "results"
SEED = 42
WARMUP = 28

# ---- Grille d'expériences (volontairement petite) --------------------------------------------
CONFIGS = (
    [("Couverture fixe", f"C={c} j", lambda c=c: FixedCover(c)) for c in (2, 3, 4, 5, 7)]
    + [("(s,Q)", f"service {s * 100:g} %, Q={q} j", lambda s=s, q=q: ReorderPointQ(s, q))
       for s in (0.90, 0.95, 0.98, 0.99) for q in (0.5, 1, 2, 3)]
    + [("(R,S)", f"service {s * 100:g} %, R={r} j", lambda s=s, r=r: PeriodicReview(s, r))
       for s in (0.80, 0.90, 0.95, 0.98) for r in (1, 2, 3)]
)


# ---- Modèle de demande ajusté sur 2024 --------------------------------------------------------
def fit_demand_model(wide2024: pd.DataFrame) -> dict:
    """Par article : demande moyenne, indice jour de semaine, indice mensuel, variabilité du bruit.

    demande attendue = mu x indice_jour x indice_mois ;  demande tirée = Poisson(attendue x bruit Gamma)
    cv² du bruit estimé par la méthode des moments : somme((y-e)² - e) / somme(e²).
    """
    model = {}
    for sku_id in wide2024.columns:
        y = wide2024[sku_id]
        mu = y.mean()
        dow, month = y.index.dayofweek, y.index.month
        wd = (y.groupby(dow).mean() / mu).to_numpy()
        y_wd = y / wd[dow]
        mi = (y_wd.groupby(month).mean() / y_wd.mean()).to_numpy()
        e = mu * wd[dow] * mi[month - 1]
        cv2 = max(float(((y - e) ** 2 - e).sum() / (e ** 2).sum()), 0.01)
        model[sku_id] = dict(mu=mu, wd=wd, mi=mi, cv2=cv2)
    return model


def draw_demand(model: dict, dates: pd.DatetimeIndex, rng) -> pd.DataFrame:
    cols = {}
    for sku_id, p in model.items():
        e = p["mu"] * p["wd"][dates.dayofweek] * p["mi"][dates.month - 1]
        k = 1.0 / p["cv2"]
        cols[sku_id] = rng.poisson(e * rng.gamma(k, 1.0 / k, size=len(dates)))
    return pd.DataFrame(cols, index=dates).astype(float)


def main(n_reps: int) -> None:
    master = pd.read_csv(DATA / "sku_master.csv").set_index("sku_id")
    seg = pd.read_csv(DATA / "segmentation.csv").set_index("sku_id")
    wide = pd.read_csv(DATA / "demand_daily.csv", parse_dates=["date"]).pivot(
        index="date", columns="sku_id", values="demand")
    wide2024 = wide.loc["2024"]
    tail2024 = wide2024.iloc[-WARMUP:]
    dates25 = pd.date_range("2025-01-01", "2025-12-31", freq="D")

    model = fit_demand_model(wide2024)
    skus = {i: Sku(int(m.lead_time_days), int(m.shelf_life_at_receipt), int(m.order_multiple))
            for i, m in master.iterrows()}
    margins = master.unit_price_mad - master.unit_cost_mad

    RESULTS.mkdir(exist_ok=True)
    rows, t0 = [], time.time()
    for rep in range(n_reps):
        rng = np.random.default_rng(SEED + rep)
        sim = draw_demand(model, dates25, rng)
        full = pd.concat([tail2024, sim])                       # 28 jours réels + 2025 simulé
        mu = full.rolling(WARMUP).mean().shift(1).iloc[WARMUP:]
        sg = full.rolling(WARMUP).std().shift(1).iloc[WARMUP:]
        for sku_id, sku in skus.items():
            d, mu_s, sg_s = sim[sku_id].to_numpy(), mu[sku_id].to_numpy(), sg[sku_id].to_numpy()
            init = initial_stock_for(mu_s[0], sku)
            for family, label, make in CONFIGS:
                df = simulate_sku(d, mu_s, sg_s, sku, make(), init)
                t = metrics(df, master.loc[sku_id, "unit_cost_mad"], margins[sku_id])
                rows.append(dict(rep=rep, family=family, params=label, abc=seg.loc[sku_id, "abc"],
                                 sku_id=sku_id, **t))
        done = rep + 1
        print(f"réplication {done}/{n_reps} ({time.time() - t0:.0f} s, reste ~{(time.time() - t0) / done * (n_reps - done):.0f} s)")

    res = pd.DataFrame(rows)
    keep = ["days", "demand", "sold", "lost", "expired", "received", "stock_sum", "stockout_days",
            "n_orders", "cost_holding", "cost_lost", "cost_expired", "cost_total"]
    agg = res.groupby(["rep", "family", "params", "abc"], sort=False)[keep].sum().reset_index()
    agg["days"] = agg["days"] / res.groupby(["rep", "family", "params", "abc"], sort=False).sku_id.nunique().to_numpy()
    agg.to_csv(RESULTS / "mc_results.csv", index=False)
    print(f"\nRésultats enregistrés : {RESULTS / 'mc_results.csv'} ({len(agg)} lignes)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
