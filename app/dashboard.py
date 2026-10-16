"""Étape 6 - Tableau de bord Streamlit.

Lancer (depuis le dossier du projet) :   streamlit run app/dashboard.py
Lit results/mc_results.csv (étape 5) et data/segmentation.csv (étape 2).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"A": 95, "B": 92, "C": 90}
COLORS = {"Couverture fixe": "#555555", "(s,Q)": "#1d4e89", "(R,S)": "#c0592b"}
FAMILIES = list(COLORS)

st.set_page_config(page_title="Stocks de produits périssables", layout="wide")


@st.cache_data
def load():
    mc = pd.read_csv(ROOT / "results" / "mc_results.csv")
    mc["fill"] = mc.sold / mc.demand
    mc["waste"] = mc.expired / mc.received
    mc["cov_days"] = mc.stock_sum / mc.demand
    seg = pd.read_csv(ROOT / "data" / "segmentation.csv")
    return mc, seg


@st.cache_data
def summarize(mc: pd.DataFrame) -> pd.DataFrame:
    p5 = lambda s: np.percentile(s, 5)
    p95 = lambda s: np.percentile(s, 95)
    return (mc.groupby(["abc", "family", "params"], sort=False)
              .agg(service=("fill", "mean"), pertes=("waste", "mean"), cout=("cost_total", "mean"),
                   cout_p5=("cost_total", p5), cout_p95=("cost_total", p95), couverture=("cov_days", "mean"))
              .reset_index())


mc, seg = load()
stat = summarize(mc)
n_reps = mc.rep.nunique()

# ------------------------------------------------------------------ en-tête
st.title("Pilotage des stocks de produits périssables")
st.caption(f"Simulation de 30 articles laitiers frais sur 2025, {n_reps} réplications Monte Carlo, "
           "demande simulée à partir d'un modèle ajusté sur 2024.")
st.info("**Données simulées.** Toutes les valeurs (demande, coûts, durées de vie, délais) sont des hypothèses "
        "documentées, pas des données d'entreprise. Les résultats valent pour ces hypothèses.")

# ------------------------------------------------------------------ filtres
with st.sidebar:
    st.header("Filtres")
    abc = st.radio("Classe ABC", list(TARGETS), horizontal=True)
    target = st.slider("Taux de service cible (%)", 80, 99, TARGETS[abc])
    fams = st.multiselect("Politiques affichées", FAMILIES, default=FAMILIES)
    st.caption("A = articles qui font 80 % de la valeur, B = les 15 % suivants, C = les 5 % restants.")

s = stat[(stat.abc == abc) & stat.family.isin(fams)]

# Meilleure configuration de chaque famille : la moins chère qui atteint la cible
best = {}
for fam in fams:
    ok = s[(s.family == fam) & (s.service * 100 >= target)]
    if len(ok):
        best[fam] = ok.sort_values("cout").iloc[0]

tab1, tab2, tab3 = st.tabs(["Recommandation", "Toutes les configurations", "Articles"])

# ------------------------------------------------------------------ onglet 1
with tab1:
    if not best:
        st.warning("Aucune configuration n'atteint cette cible de service. Baisse la cible.")
    else:
        ref = best.get("Couverture fixe")
        winner = min(best.values(), key=lambda r: r.cout)
        c1, c3, c4 = st.columns(3)
        c1.metric("Politique la moins chère", winner.family, help=f"Paramètres : {winner.params}")
        c3.metric("Taux de service obtenu", f"{winner.service * 100:.1f} %")
        if ref is not None and winner.family != "Couverture fixe":
            a = mc[(mc.abc == abc) & (mc.family == winner.family) & (mc.params == winner.params)].set_index("rep").cost_total
            b = mc[(mc.abc == abc) & (mc.family == "Couverture fixe") & (mc.params == ref.params)].set_index("rep").cost_total
            saving = (b - a) / b * 100
            c4.metric("Économie de coût vs référence", f"{saving.mean():.1f} %",
                      help=f"Intervalle 5 à 95 % : {np.percentile(saving, 5):.1f} à {np.percentile(saving, 95):.1f} %. "
                           f"Moins cher dans {(saving > 0).mean() * 100:.0f} % des réplications.")
            st.caption(f"Paramètres retenus : {winner.params}. Écart apparié (même demande à chaque réplication) : économie de {saving.mean():.1f} %, "
                       f"intervalle {np.percentile(saving, 5):.1f} à {np.percentile(saving, 95):.1f} %, "
                       f"politique moins chère dans {(saving > 0).mean() * 100:.0f} % des réplications.")
        else:
            c4.metric("Référence", "meilleure couverture fixe")
            st.caption(f"Paramètres retenus : {winner.params}.")

        left, right = st.columns(2)
        with left:
            st.subheader("Coût total par politique")
            names = list(best)
            costs = [best[n].cout for n in names]
            lo = [best[n].cout - best[n].cout_p5 for n in names]
            hi = [best[n].cout_p95 - best[n].cout for n in names]
            fig, ax = plt.subplots(figsize=(5.4, 3.6))
            ax.bar(names, costs, yerr=[lo, hi], capsize=4, color=[COLORS[n] for n in names])
            ax.set_ylabel("Coût total moyen 2025 (MAD)")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)
        with right:
            st.subheader("Service et pertes, toutes configurations")
            fig, ax = plt.subplots(figsize=(5.4, 3.6))
            for fam in fams:
                f = s[s.family == fam]
                ax.scatter(f.service * 100, f.pertes * 100, label=fam, color=COLORS[fam], s=26)
            ax.axvline(target, color="#999999", ls="--", lw=1)
            ax.set_xlabel("Taux de service moyen (%)")
            ax.set_ylabel("Pertes de péremption (%)")
            ax.legend(frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)
        st.subheader("Configuration retenue pour chaque politique")
        out = pd.DataFrame([{"Politique": r.family, "Paramètres": r.params, "Service (%)": r.service * 100,
                             "Pertes (%)": r.pertes * 100, "Couverture (jours)": r.couverture,
                             "Coût total (MAD)": r.cout} for r in best.values()])
        st.dataframe(out.style.format({"Service (%)": "{:.2f}", "Pertes (%)": "{:.2f}",
                                       "Couverture (jours)": "{:.2f}", "Coût total (MAD)": "{:,.0f}"}),
                     hide_index=True, width="stretch")
        st.caption("Les pertes de péremption restent très faibles : le gain vient des ruptures évitées, "
                   "pas des pertes évitées.")

# ------------------------------------------------------------------ onglet 2
with tab2:
    st.subheader(f"Classe {abc} : les {len(s)} configurations simulées")
    t = s.rename(columns={"family": "Politique", "params": "Paramètres", "service": "Service (%)",
                          "pertes": "Pertes (%)", "couverture": "Couverture (jours)", "cout": "Coût total (MAD)"})
    t["Service (%)"] *= 100
    t["Pertes (%)"] *= 100
    t = t.sort_values("Coût total (MAD)")[["Politique", "Paramètres", "Service (%)", "Pertes (%)",
                                           "Couverture (jours)", "Coût total (MAD)"]]
    st.dataframe(t.style.format({"Service (%)": "{:.2f}", "Pertes (%)": "{:.3f}",
                                 "Couverture (jours)": "{:.2f}", "Coût total (MAD)": "{:,.0f}"}),
                 hide_index=True, width="stretch")

# ------------------------------------------------------------------ onglet 3
with tab3:
    st.subheader("Segmentation ABC/XYZ des 30 articles (demande 2024)")
    f1, f2 = st.columns(2)
    fam_sel = f1.multiselect("Famille", sorted(seg.family.unique()), default=sorted(seg.family.unique()))
    abc_sel = f2.multiselect("Classe ABC", ["A", "B", "C"], default=["A", "B", "C"])
    view = seg[seg.family.isin(fam_sel) & seg.abc.isin(abc_sel)]
    st.dataframe(view[["sku_id", "sku_name", "family", "mu", "cv", "annual_value", "abc", "xyz", "classe"]]
                 .rename(columns={"sku_name": "Article", "family": "Famille", "mu": "Demande moyenne/jour",
                                  "cv": "Coef. de variation", "annual_value": "Valeur annuelle (MAD)",
                                  "abc": "ABC", "xyz": "XYZ", "classe": "Classe"})
                 .style.format({"Demande moyenne/jour": "{:,.0f}", "Coef. de variation": "{:.2f}",
                                "Valeur annuelle (MAD)": "{:,.0f}"}),
                 hide_index=True, width="stretch")
    st.markdown("**Matrice ABC/XYZ (nombre d'articles)**")
    st.dataframe(pd.crosstab(seg.abc, seg.xyz), width="stretch")
