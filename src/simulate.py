"""Étape 4 - Simulateur journalier d'un article sous une politique de réapprovisionnement.

Conventions (identiques pour toutes les politiques) :
- Un lot reçu le jour d avec une durée résiduelle r est vendable du jour d au jour d + r - 1.
  À la fin du jour d + r - 1, ce qui reste est périmé.
- Une commande passée en fin de jour d arrive le jour d + L et est vendable ce jour-là.
- Les lots sont vendus dans l'ordre de péremption (FEFO).
- La demande non servie est perdue (pas de commande en retard).
- Ordre dans la journée : 1. réception, 2. vente, 3. retrait des périmés, 4. position, 5. commande.
- La prévision (mu, sigma) du jour d est celle de la fenêtre glissante décalée d'un jour.
"""
from collections import deque

import numpy as np
import pandas as pd

from policies import Sku, round_up_to_multiple, shelf_life_cap


def simulate_sku(demand, mu, sigma, sku: Sku, policy, initial_stock: int = 0) -> pd.DataFrame:
    """Simule jour après jour. demand, mu, sigma : tableaux de même longueur (un jour = une case).

    Renvoie un DataFrame avec une ligne par jour : demand, received, sold, lost, expired,
    end_stock, position, order_qty.
    """
    n = len(demand)
    batches = deque()                       # lots (dernier_jour_de_vente, quantité), du plus proche de la date au plus loin
    pipeline = {}                           # jour d'arrivée -> quantité commandée
    if initial_stock > 0:
        batches.append([sku.residual_life - 1, initial_stock])

    out = {k: np.zeros(n) for k in ["demand", "received", "sold", "lost", "expired",
                                    "end_stock", "position", "order_qty"]}

    for d in range(n):
        # 1. Réception des commandes qui arrivent aujourd'hui
        arriving = pipeline.pop(d, 0)
        if arriving > 0:
            batches.append([d + sku.residual_life - 1, arriving])
        out["received"][d] = arriving

        # 2. Vente en FEFO
        need = demand[d]
        sold = 0
        while need > 0 and batches:
            take = min(batches[0][1], need)
            batches[0][1] -= take
            need -= take
            sold += take
            if batches[0][1] == 0:
                batches.popleft()
        out["demand"][d], out["sold"][d], out["lost"][d] = demand[d], sold, demand[d] - sold

        # 3. Retrait des unités périmées (dernier jour de vente atteint)
        expired = 0
        while batches and batches[0][0] <= d:
            expired += batches[0][1]
            batches.popleft()
        out["expired"][d] = expired

        # 4. Position de stock = stock physique + commandes en cours
        stock = sum(b[1] for b in batches)
        on_order = sum(pipeline.values())
        out["end_stock"][d] = stock
        out["position"][d] = stock + on_order

        # 5. Commande
        if not (np.isnan(mu[d]) or np.isnan(sigma[d])):
            qty = policy.order_qty(d, stock + on_order, mu[d], sigma[d], sku)
            if qty > 0:
                pipeline[d + sku.lead_time] = pipeline.get(d + sku.lead_time, 0) + qty
                out["order_qty"][d] = qty

    return pd.DataFrame(out)


def check_balance(df: pd.DataFrame, initial_stock: int) -> float:
    """Bilan de stock : stock initial + réceptions - ventes - périmés - stock final (doit valoir 0)."""
    return initial_stock + df.received.sum() - df.sold.sum() - df.expired.sum() - df.end_stock.iloc[-1]


def initial_stock_for(mu0: float, sku: Sku, days: float = 2.0) -> int:
    """Stock initial : `days` jours de demande moyenne, plafonné par la durée de vie, au multiple supérieur."""
    level = min(days * mu0, shelf_life_cap(mu0, sku.residual_life))
    return round_up_to_multiple(level, sku.multiple)


def metrics(df: pd.DataFrame, unit_cost: float, margin: float,
            h_annual: float = 0.20, k_order: float = 0.0, warmup: int = 28) -> dict:
    """Totaux et indicateurs, 28 premiers jours (chauffe) exclus."""
    m = df.iloc[warmup:]
    h_day = h_annual / 365
    tot = dict(
        days=len(m),
        demand=m.demand.sum(), sold=m.sold.sum(), lost=m.lost.sum(),
        expired=m.expired.sum(), received=m.received.sum(),
        stock_sum=m.end_stock.sum(), n_orders=int((m.order_qty > 0).sum()),
        stockout_days=int((m.lost > 0).sum()),
    )
    tot["cost_holding"] = h_day * unit_cost * tot["stock_sum"]
    tot["cost_lost"] = margin * tot["lost"]
    tot["cost_expired"] = unit_cost * tot["expired"]
    tot["cost_orders"] = k_order * tot["n_orders"]
    tot["cost_total"] = (tot["cost_holding"] + tot["cost_lost"]
                         + tot["cost_expired"] + tot["cost_orders"])
    return tot


def kpis(t: dict) -> dict:
    """Indicateurs à partir des totaux (agrégeables par classe : on additionne les totaux)."""
    avg_stock = t["stock_sum"] / t["days"]
    avg_demand = t["demand"] / t["days"]
    return dict(
        fill_rate=t["sold"] / t["demand"],
        stockout_rate=t["stockout_days"] / t["days"],
        waste_rate=t["expired"] / t["received"] if t["received"] else 0.0,
        avg_stock=avg_stock,
        coverage_days=avg_stock / avg_demand,
        turnover=(t["sold"] / t["days"] * 365) / avg_stock if avg_stock else 0.0,
        cost_total=t["cost_total"],
    )
