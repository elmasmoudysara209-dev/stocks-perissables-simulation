"""Étape 5 - Analyse des résultats Monte Carlo : tableau comparatif, décision par classe, courbes.

Lit results/mc_results.csv (produit par experiments.py) et écrit :
    results/decision_par_classe.csv, results/frontiere_service_pertes.png, results/cout_par_politique.png
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})

RESULTS = Path(__file__).resolve().parents[1] / "results"
TARGET = {"A": 0.95, "B": 0.92, "C": 0.90}         # taux de service cible par classe (hypothèse à justifier)
FAMILIES = ["Couverture fixe", "(s,Q)", "(R,S)"]

mc = pd.read_csv(RESULTS / "mc_results.csv")
n_reps = mc.rep.nunique()
mc["fill"] = mc.sold / mc.demand
mc["waste"] = mc.expired / mc.received
mc["cov_days"] = (mc.stock_sum / mc.days) / (mc.demand / mc.days)

p5 = lambda s: np.percentile(s, 5)
p95 = lambda s: np.percentile(s, 95)
stat = (mc.groupby(["abc", "family", "params"], sort=False)
          .agg(fill=("fill", "mean"), fill_p5=("fill", p5), waste=("waste", "mean"),
               waste_p95=("waste", p95), cost=("cost_total", "mean"),
               cost_p5=("cost_total", p5), cost_p95=("cost_total", p95), cov=("cov_days", "mean"))
          .reset_index())

pd.options.display.float_format = "{:,.2f}".format
print(f"{n_reps} réplications Monte Carlo, demande 2025 simulée, mêmes aléas pour toutes les configurations\n")

# ---- 1. Référence équitable : la meilleure couverture fixe qui atteint la cible ----------------
decision = []
for abc, target in TARGET.items():
    s = stat[stat.abc == abc]
    best = {}
    for fam in FAMILIES:
        ok = s[(s.family == fam) & (s.fill >= target)]
        if len(ok):
            best[fam] = ok.sort_values("cost").iloc[0]
    ref = best.get("Couverture fixe")
    for fam, r in best.items():
        row = dict(classe=abc, cible=target * 100, politique=fam, parametres=r.params,
                   service_pct=r.fill * 100, service_p5=r.fill_p5 * 100, pertes_pct=r.waste * 100,
                   cout=r.cost, cout_p5=r.cost_p5, cout_p95=r.cost_p95, couverture_j=r["cov"])
        if ref is not None and fam != "Couverture fixe":
            a = mc[(mc.abc == abc) & (mc.family == fam) & (mc.params == r.params)].set_index("rep").cost_total
            b = mc[(mc.abc == abc) & (mc.family == "Couverture fixe") & (mc.params == ref.params)].set_index("rep").cost_total
            diff = (a - b) / b * 100                       # écart apparié (même demande à chaque réplication)
            row.update(gain_cout_pct=-diff.mean(), gain_p5=-np.percentile(diff, 95), gain_p95=-np.percentile(diff, 5),  # gain > 0 = économie
                       reps_meilleures_pct=(diff < 0).mean() * 100)
        decision.append(row)
    if ref is None:
        print(f"Classe {abc} : aucune couverture fixe n'atteint la cible de {target:.0%}")
dec = pd.DataFrame(decision)
dec.to_csv(RESULTS / "decision_par_classe.csv", index=False)

for abc in TARGET:
    d = dec[dec.classe == abc]
    print(f"=== Classe {abc} : cible de service {TARGET[abc]:.0%}, configuration la moins chère de chaque famille ===")
    print(d[["politique", "parametres", "service_pct", "pertes_pct", "couverture_j", "cout"]].to_string(index=False))
    if "gain_cout_pct" in d:
        g = d.dropna(subset=["gain_cout_pct"])
        for _, r in g.iterrows():
            verbe = "économie" if r.gain_cout_pct > 0 else "surcoût"
            print(f"  -> {r.politique} contre la meilleure couverture fixe : {verbe} de {abs(r.gain_cout_pct):.1f} % "
                  f"(intervalle de {r.gain_p5:+.1f} à {r.gain_p95:+.1f} %, positif = économie), "
                  f"moins cher dans {r.reps_meilleures_pct:.0f} % des réplications")
    print()

# ---- 2. Courbe service contre pertes, classe A -------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(10, 4.0), sharey=False)
colors = {"Couverture fixe": "#444444", "(s,Q)": "#1d4e89", "(R,S)": "#c0592b"}
for a, abc in zip(ax, TARGET):
    s = stat[stat.abc == abc]
    for fam in FAMILIES:
        f = s[s.family == fam]
        a.scatter(f.fill * 100, f.waste * 100, label=fam, color=colors[fam], s=28)
    a.axvline(TARGET[abc] * 100, color="#999999", ls="--", lw=1)
    a.set_title(f"Classe {abc}")
    a.set_xlabel("Service moyen (%)")
    a.set_ylabel("Pertes (%)")
    a.grid(alpha=0.25)
ax[0].legend()
fig.suptitle("Chaque point est une configuration de politique", fontsize=12)
fig.tight_layout()
fig.savefig(RESULTS / "frontiere_service_pertes.png", dpi=170)

# ---- 3. Coût des configurations retenues, avec intervalles -------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(10, 3.9))
for a, abc in zip(ax, TARGET):
    d = dec[dec.classe == abc].reset_index(drop=True)
    a.bar(d.politique, d.cout, yerr=[d.cout - d.cout_p5, d.cout_p95 - d.cout], capsize=4,
          color=[colors[p] for p in d.politique])
    a.set_title(f"Classe {abc}, service ≥ {TARGET[abc]:.0%}", fontsize=11)
    a.set_ylabel("Coût total moyen (MAD)")
    a.set_xticks(range(len(d)))
    a.set_xticklabels(["Fixe", "(s,Q)", "(R,S)"][:len(d)] if len(d) == 3 else d.politique)
fig.suptitle("Coût total moyen et intervalle 5 à 95 % sur les réplications", fontsize=12)
fig.tight_layout()
fig.savefig(RESULTS / "cout_par_politique.png", dpi=170)
print(f"Graphiques enregistrés dans {RESULTS}")
