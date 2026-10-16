"""Étape 1 - Contrôles de qualité du jeu de données.

À lancer après generate_data.py. Le script s'arrête avec une erreur si un contrôle échoue,
et affiche les statistiques à comparer aux hypothèses du cahier des charges.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    master = pd.read_csv(DATA_DIR / "sku_master.csv")
    demand = pd.read_csv(DATA_DIR / "demand_daily.csv", parse_dates=["date"])

    # ---- 1. Structure ----------------------------------------------------------------
    n_sku, n_days = master.sku_id.nunique(), demand.date.nunique()
    assert n_sku == 30, f"30 articles attendus, {n_sku} trouvés"
    assert n_days == 731, f"731 jours attendus, {n_days} trouvés"
    assert len(demand) == n_sku * n_days, "il manque des lignes (un article x un jour)"
    assert not demand.isna().any().any(), "valeurs manquantes"
    assert (demand.demand >= 0).all(), "demande négative"
    assert (master.unit_price_mad > master.unit_cost_mad).all(), "prix de vente <= coût d'achat"
    assert (master.shelf_life_at_receipt > master.lead_time_days).all(), "durée de vie résiduelle trop courte"
    print(f"[OK] {n_sku} articles x {n_days} jours = {len(demand)} lignes, aucune valeur manquante ou négative\n")

    df = demand.merge(master[["sku_id", "family"]], on="sku_id")
    df["dow"] = df.date.dt.dayofweek
    sku_mean = df.groupby("sku_id").demand.transform("mean")
    df["index"] = df.demand / sku_mean  # demande rapportée à la moyenne de l'article

    # ---- 2. Demande moyenne par famille ------------------------------------------------
    fam = df.groupby("family").agg(demande_moyenne_jour=("demand", "mean"))
    print("Demande moyenne par article et par jour, par famille (unités) :")
    print(fam.round(0).to_string(), "\n")

    # ---- 3. Profil hebdomadaire --------------------------------------------------------
    jours = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]
    prof = df.groupby("dow")["index"].mean()
    prof.index = jours
    print("Profil hebdomadaire (1,00 = moyenne de l'article ; attendu : pic ven et sam) :")
    print(prof.round(2).to_string(), "\n")
    assert prof["sam"] > prof["mer"] and prof["ven"] > prof["mar"], "profil hebdomadaire incohérent"

    # ---- 4. Effets promotion et Ramadan ------------------------------------------------
    promo = df.groupby("promo")["index"].mean()
    print(f"Promotion : demande x{promo[1] / promo[0]:.2f} en promotion (attendu : entre 1,30 et 1,60)")
    assert 1.2 < promo[1] / promo[0] < 1.7, "effet promotion hors de la plage attendue"
    print(f"Jours en promotion par article et par an : {df.promo.mean() * 365:.0f} en moyenne\n")

    # Effet Ramadan mesuré article par article, hors jours de promotion, puis moyenné par famille
    base = df[(df.ramadan == 0) & (df.promo == 0)].groupby("sku_id").demand.mean()
    rama_sku = df[(df.ramadan == 1) & (df.promo == 0)].groupby("sku_id").demand.mean() / base
    truth = pd.read_csv(DATA_DIR / "ground_truth.csv").merge(master[["sku_id", "family"]], on="sku_id")
    rama = pd.DataFrame({
        "mesuré": rama_sku.groupby(truth.set_index("sku_id").family).mean(),
        "attendu": 1 + truth.groupby("family").ramadan_effect.first(),
    })
    print("Ramadan : demande relative à la période normale, hors promotions (attendu : lben/raïb en tête) :")
    print(rama.round(2).to_string(), "\n")

    # ---- 5. Variabilité par article -----------------------------------------------------
    cv = df.groupby("sku_id").demand.agg(lambda s: s.std() / s.mean())
    print("Coefficient de variation de la demande journalière (écart-type / moyenne) :")
    print(f"  minimum {cv.min():.2f} | médiane {cv.median():.2f} | maximum {cv.max():.2f}")
    print("\n[OK] tous les contrôles sont passés")


if __name__ == "__main__":
    main()
