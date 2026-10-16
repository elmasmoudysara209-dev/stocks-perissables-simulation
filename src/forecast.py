from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
WINDOW = 28

demand = pd.read_csv(DATA / "demand_daily.csv", parse_dates=["date"])
seg = pd.read_csv(DATA / "segmentation.csv")

# Tableau large : une ligne par jour, une colonne par article
wide = demand.pivot(index="date", columns="sku_id", values="demand")

# Moyenne et écart-type des 28 derniers jours, DÉCALÉS d'un jour (shift(1)) :
# la prévision du jour t n'utilise que les jours jusqu'à t-1.
mu = wide.rolling(WINDOW, min_periods=WINDOW).mean().shift(1)
sigma = wide.rolling(WINDOW, min_periods=WINDOW).std().shift(1)

print("Jour 28 (index 27) : prévision NaN ?", mu.iloc[27].isna().all())
print("Jour 29 (index 28) : prévision NaN ?", mu.iloc[28].isna().all())
print("Premier jour avec prévision :", mu.dropna().index[0].date())

# Qualité de la prévision, par classe ABC, sur 2024-02-01 -> 2025-12-31
ok = wide.index >= "2024-02-01"
err = (wide - mu)[ok]
real = wide[ok]
rows = []
for cls, ids in seg.groupby("abc").sku_id:
    e, r = err[ids], real[ids]
    rows.append(dict(classe=cls, biais_pct=e.sum().sum() / r.sum().sum() * 100,
                     erreur_abs_pct=e.abs().sum().sum() / r.sum().sum() * 100))
print(pd.DataFrame(rows).round(1).to_string(index=False))
e, r = err, real
print("TOTAL biais %.1f %% | erreur absolue %.1f %%" %
      (e.sum().sum() / r.sum().sum() * 100, e.abs().sum().sum() / r.sum().sum() * 100))

# Sauvegarde pour l'étape 3
mu.to_csv(DATA / "rolling_mu.csv")
sigma.to_csv(DATA / "rolling_sigma.csv")

# Exemple chiffré : SKU02 au 1er mars 2024
print("SKU02, 2024-03-01 : mu = %.0f, sigma = %.0f" % (mu.loc["2024-03-01", "SKU02"], sigma.loc["2024-03-01", "SKU02"]))
