# Pilotage des stocks de produits périssables par simulation

Projet individuel de génie industriel. Question : pour 30 articles laitiers frais, quelle politique de
réapprovisionnement minimise le coût total pour un taux de service cible, en tenant compte de la durée de vie,
des délais et de la variabilité de la demande ?

> **Les données sont simulées** à partir d'hypothèses documentées. Elles ne viennent d'aucune entreprise, et les
> résultats valent pour ces hypothèses.

## Résultats principaux (100 réplications Monte Carlo, demande 2025)

| Classe | Politique la moins chère | Service | Économie de coût vs meilleure couverture fixe |
| --- | --- | --- | --- |
| A (95 % visé) | (R,S) quotidien, service réglé 90 % | 99,8 % | 70,2 % (intervalle 5-95 % : 64,4 à 74,7) |
| B (92 % visé) | (R,S) quotidien, service réglé 98 % | 99,9 % | 19,8 % (9,7 à 31,0) |
| C (90 % visé) | (R,S) quotidien, service réglé 98 % | 99,95 % | 12,6 % (3,9 à 21,2) |

- Le (s,Q) n'est meilleur que la référence qu'en classe A ; en B et C l'écart est dans l'intervalle de variation.
- Le gain vient des ruptures évitées, pas des pertes de péremption (au plus 0,84 % des unités reçues).
- La décision tient sous variabilité +20 %, coût de possession 15 % ou 25 %, durée de vie -1 jour.

Détails, limites et méthode : `report/Rapport_Pilotage_Stocks_Perissables.docx`.

## Structure

```
projet_stock_perissables/
├── data/        sku_master.csv, demand_daily.csv, ground_truth.csv, segmentation.csv, rolling_*.csv
├── src/
│   ├── generate_data.py    génère les données simulées (graine 42)
│   ├── check_data.py       contrôles de qualité des données
│   ├── segment.py          segmentation ABC/XYZ (demande 2024)
│   ├── forecast.py         moyenne et écart-type glissants sur 28 jours, décalés d'un jour
│   ├── validate_truth.py   compare les estimations à la vérité du générateur
│   ├── policies.py         couverture fixe, (s,Q), (R,S), plafond de péremption
│   ├── simulate.py         simulateur journalier (délai, FEFO, péremption, bilan de stock)
│   ├── run_backtest.py     rétro-test 2025 avec paramètres par défaut
│   ├── experiments.py      Monte Carlo : 33 configurations x 100 réplications
│   ├── analyze.py          tableau de décision et graphiques
│   └── sensitivity.py      test de sensibilité
├── tests/       test_policies.py, test_simulate.py (24 tests)
├── app/         dashboard.py (Streamlit)
├── results/     mc_results.csv, decision_par_classe.csv, sensibilite.csv, graphiques
├── report/      rapport de 3 pages
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduire les résultats (depuis la racine du projet)

```bash
python src/generate_data.py          # 1. données simulées
python src/check_data.py             #    contrôles
python src/segment.py                # 2. segmentation ABC/XYZ
python src/forecast.py               #    prévision glissante
python -m pytest -q tests            # 3-4. politiques et simulateur : 24 tests
python src/run_backtest.py           # 4. rétro-test 2025
python src/experiments.py 100        # 5. Monte Carlo (environ 10 à 15 minutes)
python src/analyze.py                #    décision par classe et graphiques
python src/sensitivity.py 30         # 6. sensibilité (environ 5 minutes)
streamlit run app/dashboard.py       #    tableau de bord
```

Avec la graine fixe, les chiffres se retrouvent à l'identique.

## Conventions du simulateur

- Un lot reçu le jour d avec une durée résiduelle r est vendable du jour d au jour d + r - 1.
- Une commande passée en fin de jour d arrive le jour d + L.
- Ordre dans la journée : réception, vente (FEFO), retrait des périmés, position de stock, commande.
- Ventes manquées perdues ; durée de vie résiduelle à la réception = durée totale - 2 jours.
- Coût total = possession (20 %/an) + marge sur ventes perdues + coût d'achat des périmés.

## Limites

Données simulées ; délai fixe ; un seul site ; une vente perdue coûte seulement sa marge ; grille de paramètres
limitée ; cibles de service par classe choisies, pas calculées.
