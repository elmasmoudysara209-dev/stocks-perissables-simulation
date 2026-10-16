"""Étape 6 - Test de sensibilité : la décision tient-elle si les hypothèses changent ?

On garde les configurations retenues à l'étape 5 (couverture fixe C=3 et C=5, (s,Q) 99 % Q=1 et Q=2,
(R,S) 90 % et 98 % avec R=1) et on les refait tourner sous des hypothèses modifiées :
variabilité +20 %, coût de possession 15 % et 25 % par an, durée de vie résiduelle -1 jour.

Usage :  python src/sensitivity.py 30
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import RESULTS, DATA, SEED, WARMUP, draw_demand, fit_demand_model
from policies import FixedCover, PeriodicReview, ReorderPointQ, Sku
from simulate import initial_stock_for, metrics, simulate_sku

CONFIGS = {
    "Couverture fixe C=3": ("Couverture fixe", lambda: FixedCover(3)),
    "Couverture fixe C=5": ("Couverture fixe", lambda: FixedCover(5)),
    "(s,Q) 99 %, Q=1 j": ("(s,Q)", lambda: ReorderPointQ(0.99, 1)),
    "(s,Q) 99 %, Q=2 j": ("(s,Q)", lambda: ReorderPointQ(0.99, 2)),
    "(R,S) 90 %, R=1 j": ("(R,S)", lambda: PeriodicReview(0.90, 1)),
    "(R,S) 98 %, R=1 j": ("(R,S)", lambda: PeriodicReview(0.98, 1)),
}
# configuration de chaque famille retenue par classe (étape 5)
CHOICE = {"A": ("Couverture fixe C=3", "(s,Q) 99 %, Q=1 j", "(R,S) 90 %, R=1 j"),
          "B": ("Couverture fixe C=5", "(s,Q) 99 %, Q=2 j", "(R,S) 98 %, R=1 j"),
          "C": ("Couverture fixe C=5", "(s,Q) 99 %, Q=2 j", "(R,S) 98 %, R=1 j")}
SCENARIOS = {                      # nom : (facteur sur le CV², coût de possession annuel, jours de durée de vie en moins)
    "Base": (1.0, 0.20, 0),
    "Variabilité +20 %": (1.2 ** 2, 0.20, 0),
    "Possession 15 %": (1.0, 0.15, 0),
    "Possession 25 %": (1.0, 0.25, 0),
    "Durée de vie -1 jour": (1.0, 0.20, 1),
}


def main(n_reps: int) -> None:
    master = pd.read_csv(DATA / "sku_master.csv").set_index("sku_id")
    seg = pd.read_csv(DATA / "segmentation.csv").set_index("sku_id")
    wide = pd.read_csv(DATA / "demand_daily.csv", parse_dates=["date"]).pivot(
        index="date", columns="sku_id", values="demand")
    wide2024, tail2024 = wide.loc["2024"], wide.loc["2024"].iloc[-WARMUP:]
    dates25 = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    base_model = fit_demand_model(wide2024)
    margins = master.unit_price_mad - master.unit_cost_mad

    rows = []
    for scen, (cv_mult, h_annual, life_delta) in SCENARIOS.items():
        model = copy.deepcopy(base_model)
        for p in model.values():
            p["cv2"] *= cv_mult
        skus = {i: Sku(int(m.lead_time_days), int(m.shelf_life_at_receipt) - life_delta, int(m.order_multiple))
                for i, m in master.iterrows()}
        for rep in range(n_reps):
            rng = np.random.default_rng(SEED + rep)
            sim = draw_demand(model, dates25, rng)
            full = pd.concat([tail2024, sim])
            mu = full.rolling(WARMUP).mean().shift(1).iloc[WARMUP:]
            sg = full.rolling(WARMUP).std().shift(1).iloc[WARMUP:]
            for sku_id, sku in skus.items():
                d, mu_s, sg_s = sim[sku_id].to_numpy(), mu[sku_id].to_numpy(), sg[sku_id].to_numpy()
                init = initial_stock_for(mu_s[0], sku)
                for name, (_, make) in CONFIGS.items():
                    df = simulate_sku(d, mu_s, sg_s, sku, make(), init)
                    t = metrics(df, master.loc[sku_id, "unit_cost_mad"], margins[sku_id], h_annual=h_annual)
                    rows.append(dict(scenario=scen, rep=rep, config=name, abc=seg.loc[sku_id, "abc"],
                                     cost=t["cost_total"], demand=t["demand"], sold=t["sold"],
                                     expired=t["expired"], received=t["received"]))
        print(f"scénario terminé : {scen}")

    res = pd.DataFrame(rows)
    agg = res.groupby(["scenario", "abc", "config", "rep"], sort=False)[["cost", "demand", "sold", "expired", "received"]].sum().reset_index()
    agg.to_csv(RESULTS / "sensibilite_detail.csv", index=False)

    out = []
    for scen in SCENARIOS:
        for abc, (ref, sq, rs) in CHOICE.items():
            sub = agg[(agg.scenario == scen) & (agg.abc == abc)]
            cost = sub.pivot(index="rep", columns="config", values="cost")
            fill = sub.groupby("config").apply(lambda x: x.sold.sum() / x.demand.sum(), include_groups=False)
            waste = sub.groupby("config").apply(lambda x: x.expired.sum() / x.received.sum(), include_groups=False)
            row = dict(scenario=scen, classe=abc)
            for label, cfg in (("(s,Q)", sq), ("(R,S)", rs)):
                saving = (cost[ref] - cost[cfg]) / cost[ref] * 100
                row[f"eco_{label}"] = saving.mean()
                row[f"meilleur_{label}_pct"] = (saving > 0).mean() * 100
            row["service_ref"] = fill[ref] * 100
            row["service_(R,S)"] = fill[rs] * 100
            row["pertes_ref"] = waste[ref] * 100
            row["pertes_(R,S)"] = waste[rs] * 100
            out.append(row)
    out = pd.DataFrame(out)
    out.to_csv(RESULTS / "sensibilite.csv", index=False)

    pd.options.display.float_format = "{:,.1f}".format
    print("\nÉconomie de coût (%) par rapport à la couverture fixe retenue ; positif = la politique est moins chère")
    print(out[["scenario", "classe", "eco_(s,Q)", "eco_(R,S)", "meilleur_(R,S)_pct", "pertes_ref", "pertes_(R,S)"]]
          .rename(columns={"meilleur_(R,S)_pct": "(R,S) meilleur dans (%) des rép.",
                           "pertes_ref": "pertes réf (%)", "pertes_(R,S)": "pertes (R,S) (%)"}).to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
