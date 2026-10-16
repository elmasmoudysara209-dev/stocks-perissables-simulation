"""Tests de l'étape 4 : cas simples calculables à la main."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from policies import FixedCover, PeriodicReview, ReorderPointQ, Sku
from simulate import check_balance, simulate_sku


class NeverOrder:
    def order_qty(self, day, position, mu, sigma, sku):
        return 0


class Scripted:
    """Commande des quantités fixées à l'avance : {jour: quantité}."""
    def __init__(self, orders):
        self.orders = orders

    def order_qty(self, day, position, mu, sigma, sku):
        return self.orders.get(day, 0)


def run(demand, sku, policy, initial):
    n = len(demand)
    return simulate_sku(np.array(demand, dtype=float), np.full(n, 100.0), np.full(n, 30.0), sku, policy, initial)


def test_rupture():
    df = run([130], Sku(1, 5, 12), NeverOrder(), initial=100)
    assert df.sold[0] == 100 and df.lost[0] == 30 and df.end_stock[0] == 0


def test_peremption_le_bon_jour():
    # lot de 50, durée résiduelle 3 : vendable aux jours 0, 1, 2 -> périmé à la fin du jour 2
    df = run([0, 0, 0, 0], Sku(1, 3, 12), NeverOrder(), initial=50)
    assert list(df.end_stock) == [50, 50, 0, 0]
    assert list(df.expired) == [0, 0, 50, 0]


def test_fefo_le_plus_ancien_est_vendu_en_premier():
    # lot initial 100 (dernier jour de vente 2) ; lot commandé au jour 0 : arrive jour 1, dernier jour de vente 3
    # jour 1 : demande 60 -> prise sur le lot initial (FEFO) -> il reste 40 d'ancien + 100 de nouveau
    df = run([0, 60, 0, 0], Sku(1, 3, 12), Scripted({0: 100}), initial=100)
    assert df.end_stock[1] == 140
    assert df.expired[2] == 40            # l'ancien lot (40) périme le jour 2
    assert df.expired[3] == 100           # le nouveau lot périme le jour 3


def test_delai():
    # commande de 100 passée au jour 0 avec un délai de 2 : rien aux jours 0 et 1, reçue au jour 2
    df = run([0, 0, 0], Sku(2, 5, 12), Scripted({0: 100}), initial=0)
    assert list(df.received) == [0, 0, 100]
    assert list(df.end_stock) == [0, 0, 100]


def test_commande_en_cours_comptee_dans_la_position():
    # position mesurée AVANT de passer commande : 0 le jour 0, puis 100 (en route, puis en stock)
    df = run([0, 0, 0], Sku(2, 5, 12), Scripted({0: 100}), initial=0)
    assert list(df.position) == [0, 100, 100]


@pytest.mark.parametrize("policy", [FixedCover(3), ReorderPointQ(0.95, 2), PeriodicReview(0.95, 2)])
def test_bilan_exact_sur_serie_aleatoire(policy):
    rng = np.random.default_rng(0)
    n = 200
    demand = rng.poisson(300, n).astype(float)
    sku = Sku(2, 8, 24)
    initial = 600
    df = simulate_sku(demand, np.full(n, 300.0), np.full(n, 90.0), sku, policy, initial)
    assert check_balance(df, initial) == pytest.approx(0, abs=1e-9)
    assert (df.end_stock >= 0).all()
