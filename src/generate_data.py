"""Étape 1 - Génération du jeu de données simulé (produits laitiers frais).

Produit trois fichiers dans data/ :
- sku_master.csv    : fiche de chaque article (ce que l'entreprise connaît)
- demand_daily.csv  : ventes journalières observées (ce que l'entreprise enregistre)
- ground_truth.csv  : vrais paramètres du générateur de demande, réservés à la
                      validation de l'étape 2 (comparer tes estimations à la vérité)

Toutes les valeurs sont des HYPOTHÈSES indicatives (voir le cahier des charges, section 4).
Même graine (SEED) = mêmes fichiers à l'identique.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START, END = "2024-01-01", "2025-12-31"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Hypothèses par famille (cahier des charges, section 4)
FAMILIES = {
    "Lait pasteurisé": dict(n=4, shelf_life=7, lead_time=1, demand=(600, 1500), cost=(4, 6), season=0.03, ramadan=0.10),
    "Lben et raïb": dict(n=6, shelf_life=12, lead_time=2, demand=(300, 900), cost=(4, 7), season=0.08, ramadan=0.35),
    "Yaourts": dict(n=10, shelf_life=24, lead_time=2, demand=(150, 1200), cost=(2, 5), season=0.15, ramadan=0.15),
    "Desserts lactés": dict(n=6, shelf_life=21, lead_time=2, demand=(100, 500), cost=(3, 6), season=0.08, ramadan=0.25),
    "Fromages frais": dict(n=4, shelf_life=14, lead_time=3, demand=(60, 300), cost=(5, 12), season=0.05, ramadan=0.10),
}

# Effet jour de la semaine, lundi (0) à dimanche (6) ; la moyenne vaut exactement 1
WEEKDAY = np.array([0.95, 0.90, 0.90, 0.95, 1.15, 1.20, 0.95])

# Dates approximatives du Ramadan (début, fin incluses)
RAMADAN = [("2024-03-11", "2024-04-09"), ("2025-03-01", "2025-03-30")]

PROMOS_PER_YEAR = 6
PROMO_DAYS = (3, 7)          # durée d'une promotion, en jours (bornes incluses)
PROMO_UPLIFT = (0.30, 0.60)  # hausse de demande pendant une promotion
CV_NOISE = (0.25, 0.50)      # variabilité aléatoire autour de la demande attendue
SHELF_LIFE_LOSS_AT_RECEIPT = 2  # jours de durée de vie perdus avant la réception
ORDER_MULTIPLES = [12, 24, 48]
MARGIN = (0.25, 0.40)


def build_catalog(rng: np.random.Generator) -> pd.DataFrame:
    """Une ligne par article : fiche article + vrais paramètres de demande."""
    rows = []
    k = 0
    for family, p in FAMILIES.items():
        for i in range(p["n"]):
            k += 1
            lo, hi = p["demand"]
            demand_base = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))  # répartition log-uniforme
            cost = round(float(rng.uniform(*p["cost"])), 2)
            margin = float(rng.uniform(*MARGIN))
            rows.append(dict(
                sku_id=f"SKU{k:02d}",
                sku_name=f"{family} {i + 1}",
                family=family,
                shelf_life_days=p["shelf_life"],
                shelf_life_at_receipt=p["shelf_life"] - SHELF_LIFE_LOSS_AT_RECEIPT,
                lead_time_days=p["lead_time"],
                order_multiple=int(rng.choice(ORDER_MULTIPLES)),
                unit_cost_mad=cost,
                unit_price_mad=round(cost * (1 + margin), 2),
                # --- vraies valeurs du générateur (ground_truth) ---
                demand_base=round(demand_base, 1),
                cv_noise=round(float(rng.uniform(*CV_NOISE)), 3),
                season_amp=p["season"],
                ramadan_effect=p["ramadan"],
            ))
    return pd.DataFrame(rows)


def promo_multiplier(rng: np.random.Generator, dates: pd.DatetimeIndex) -> np.ndarray:
    """Multiplicateur de demande dû aux promotions (1 hors promotion)."""
    mult = np.ones(len(dates))
    for year in sorted(set(dates.year)):
        idx = np.where(dates.year == year)[0]
        starts = rng.choice(idx, size=PROMOS_PER_YEAR, replace=False)
        for s in starts:
            duration = int(rng.integers(PROMO_DAYS[0], PROMO_DAYS[1] + 1))
            uplift = float(rng.uniform(*PROMO_UPLIFT))
            sl = slice(s, min(s + duration, idx[-1] + 1))
            mult[sl] = np.maximum(mult[sl], 1 + uplift)
    return mult


def simulate_sku(rng: np.random.Generator, sku: pd.Series, dates: pd.DatetimeIndex,
                 ramadan_flag: np.ndarray) -> pd.DataFrame:
    """Demande journalière d'un article = Poisson(demande attendue x bruit Gamma)."""
    weekday = WEEKDAY[dates.dayofweek]
    season = 1 + sku.season_amp * np.sin(2 * np.pi * (dates.dayofyear.to_numpy() - 80) / 365)
    promo = promo_multiplier(rng, dates)
    ramadan = 1 + sku.ramadan_effect * ramadan_flag
    expected = sku.demand_base * weekday * season * promo * ramadan

    shape = 1.0 / sku.cv_noise ** 2                      # bruit de moyenne 1 et de CV = cv_noise
    noise = rng.gamma(shape, 1.0 / shape, size=len(dates))
    demand = rng.poisson(expected * noise)

    return pd.DataFrame(dict(
        date=dates.strftime("%Y-%m-%d"),
        sku_id=sku.sku_id,
        demand=demand.astype(int),
        promo=(promo > 1).astype(int),
        ramadan=ramadan_flag.astype(int),
    ))


def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, END, freq="D")
    ramadan_flag = np.zeros(len(dates))
    for a, b in RAMADAN:
        ramadan_flag[(dates >= a) & (dates <= b)] = 1

    catalog = build_catalog(rng)
    demand = pd.concat([simulate_sku(rng, sku, dates, ramadan_flag) for _, sku in catalog.iterrows()],
                       ignore_index=True)

    master_cols = ["sku_id", "sku_name", "family", "shelf_life_days", "shelf_life_at_receipt",
                   "lead_time_days", "order_multiple", "unit_cost_mad", "unit_price_mad"]
    truth_cols = ["sku_id", "demand_base", "cv_noise", "season_amp", "ramadan_effect"]

    DATA_DIR.mkdir(exist_ok=True)
    catalog[master_cols].to_csv(DATA_DIR / "sku_master.csv", index=False)
    demand.to_csv(DATA_DIR / "demand_daily.csv", index=False)
    catalog[truth_cols].to_csv(DATA_DIR / "ground_truth.csv", index=False)
    print(f"{len(catalog)} articles, {len(dates)} jours, {len(demand)} lignes -> {DATA_DIR}")


if __name__ == "__main__":
    main()
