"""Étape 3 - Les trois politiques de réapprovisionnement.

Chaque politique répond à la question : « aujourd'hui, combien commander ? »
    order_qty(day, position, mu, sigma, sku) -> int (unités, multiple du conditionnement)

position = stock physique + commandes en cours (jamais le stock physique seul).
"""
import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass(frozen=True)
class Sku:
    lead_time: int      # délai de livraison L (jours)
    residual_life: int  # durée de vie résiduelle à la réception (jours)
    multiple: int       # multiple de commande (unités par conditionnement)


def z_value(service_level: float) -> float:
    """Coefficient z pour un niveau de service (0.95 -> 1.645)."""
    return float(norm.ppf(service_level))


def round_up_to_multiple(q: float, multiple: int) -> int:
    return int(math.ceil(q / multiple) * multiple) if q > 0 else 0


def round_down_to_multiple(q: float, multiple: int) -> int:
    return int(math.floor(q / multiple) * multiple) if q > 0 else 0


def shelf_life_cap(mu: float, residual_life: int) -> float:
    """Niveau de stock maximal : ce qui peut se vendre avant la date limite."""
    return mu * (residual_life - 1)


def order_up_to(level: float, position: float, sku: Sku, cap: float) -> int:
    """Commande qui ramène la position à `level`, arrondie au multiple supérieur,
    sans jamais dépasser le plafond de péremption `cap`."""
    raw = level - position
    if raw <= 0:
        return 0
    qty = round_up_to_multiple(raw, sku.multiple)
    if position + qty > cap:                       # l'arrondi dépasse le plafond
        qty = round_down_to_multiple(cap - position, sku.multiple)
    return max(qty, 0)


class FixedCover:
    """Référence : chaque jour, ramener la position à C jours de demande moyenne."""

    def __init__(self, cover_days: float):
        self.cover_days = cover_days

    def order_qty(self, day, position, mu, sigma, sku: Sku) -> int:
        cap = shelf_life_cap(mu, sku.residual_life)
        level = min(self.cover_days * mu, cap)
        return order_up_to(level, position, sku, cap)


class ReorderPointQ:
    """(s,Q) : si la position est <= s, commander des lots de Q = q_days x mu."""

    def __init__(self, service: float, q_days: float):
        self.z = z_value(service)
        self.q_days = q_days

    def reorder_point(self, mu, sigma, sku: Sku) -> float:
        L = sku.lead_time
        return mu * L + self.z * sigma * math.sqrt(L)

    def order_qty(self, day, position, mu, sigma, sku: Sku) -> int:
        s = self.reorder_point(mu, sigma, sku)
        if position > s:
            return 0
        cap = shelf_life_cap(mu, sku.residual_life)
        lot = round_up_to_multiple(self.q_days * mu, sku.multiple)
        if lot == 0:
            return 0
        n_lots = int((s - position) // lot) + 1    # nombre de lots pour repasser au-dessus de s
        qty = n_lots * lot
        if position + qty > cap:
            qty = max(round_down_to_multiple(cap - position, sku.multiple), 0)
        return qty


class PeriodicReview:
    """(R,S) : tous les R jours, ramener la position au niveau S."""

    def __init__(self, service: float, review_days: int):
        self.z = z_value(service)
        self.R = review_days

    def order_up_to_level(self, mu, sigma, sku: Sku) -> float:
        T = self.R + sku.lead_time
        return mu * T + self.z * sigma * math.sqrt(T)

    def order_qty(self, day, position, mu, sigma, sku: Sku) -> int:
        if day % self.R != 0:
            return 0
        cap = shelf_life_cap(mu, sku.residual_life)
        level = min(self.order_up_to_level(mu, sigma, sku), cap)
        return order_up_to(level, position, sku, cap)
