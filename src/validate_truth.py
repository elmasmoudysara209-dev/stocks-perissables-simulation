from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

seg = pd.read_csv(DATA / "segmentation.csv").set_index("sku_id")
truth = pd.read_csv(DATA / "ground_truth.csv").set_index("sku_id")

# Mon estimation (moyenne 2024) contre la vraie demande de base du générateur
comp = seg[["sku_name", "family", "mu", "cv", "xyz"]].join(truth[["demand_base", "cv_noise"]])
comp["ratio"] = comp.mu / comp.demand_base

print(comp.sort_values("ratio")[["sku_name", "mu", "demand_base", "ratio"]].round(2).to_string())
print()
print("Ratio mu / demand_base : min %.2f | médiane %.2f | max %.2f"
      % (comp.ratio.min(), comp.ratio.median(), comp.ratio.max()))

# Les articles que je classe Z sont-ils vraiment les plus erratiques ?
print()
print("Bruit réel moyen (cv_noise) par classe XYZ :")
print(comp.groupby("xyz").cv_noise.mean().round(3).to_string())
print("Corrélation cv estimé / cv_noise réel : %.2f" % comp.cv.corr(comp.cv_noise))
