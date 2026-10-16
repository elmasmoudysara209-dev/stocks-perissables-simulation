"""Tests de l'étape 3 : chaque nombre est calculable à la main (exemple : SKU02, lait pasteurisé)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from policies import (FixedCover, PeriodicReview, ReorderPointQ, Sku, order_up_to,
                      round_down_to_multiple, round_up_to_multiple, shelf_life_cap, z_value)

MU, SIGMA = 1552.0, 503.0
SKU02 = Sku(lead_time=1, residual_life=5, multiple=48)


def test_z_values():
    assert z_value(0.90) == pytest.approx(1.2816, abs=1e-3)
    assert z_value(0.95) == pytest.approx(1.6449, abs=1e-3)
    assert z_value(0.98) == pytest.approx(2.0537, abs=1e-3)


def test_rounding():
    assert round_up_to_multiple(100, 48) == 144
    assert round_up_to_multiple(96, 48) == 96
    assert round_up_to_multiple(0, 48) == 0
    assert round_down_to_multiple(100, 48) == 96


def test_shelf_life_cap():
    assert shelf_life_cap(MU, 5) == 1552 * 4          # 6 208


def test_reorder_point_s():
    s = ReorderPointQ(0.95, 3).reorder_point(MU, SIGMA, SKU02)
    assert s == pytest.approx(2379, abs=1)            # 1552 + 1.645 x 503


def test_order_up_to_level_S_R1_not_capped():
    S = PeriodicReview(0.95, 1).order_up_to_level(MU, SIGMA, SKU02)
    assert S == pytest.approx(4274, abs=1)            # 1552 x 2 + 1.645 x 503 x sqrt(2)
    assert S < shelf_life_cap(MU, 5)


def test_order_up_to_level_S_R3_exceeds_cap():
    S = PeriodicReview(0.95, 3).order_up_to_level(MU, SIGMA, SKU02)
    assert S == pytest.approx(7862, abs=1)
    assert S > shelf_life_cap(MU, 5)                  # le plafond va couper


def test_fixed_cover_basic():
    qty = FixedCover(3).order_qty(0, 1000, MU, SIGMA, SKU02)
    assert qty == 3696                                # (4656 - 1000) = 3656 -> 77 x 48
    assert qty % 48 == 0


def test_fixed_cover_never_exceeds_cap():
    qty = FixedCover(10).order_qty(0, 0, MU, SIGMA, SKU02)
    assert qty == 6192                                # niveau plafonné à 6 208 -> 129 x 48
    assert qty <= shelf_life_cap(MU, 5)


def test_fixed_cover_nothing_to_order():
    assert FixedCover(3).order_qty(0, 5000, MU, SIGMA, SKU02) == 0


def test_sQ_above_s_orders_nothing():
    assert ReorderPointQ(0.95, 3).order_qty(0, 3000, MU, SIGMA, SKU02) == 0


def test_sQ_below_s_orders_one_lot_capped():
    qty = ReorderPointQ(0.95, 3).order_qty(0, 2000, MU, SIGMA, SKU02)
    assert qty == 4176                                # lot 4 656 -> 2000 + 4656 dépasse 6 208 -> 87 x 48
    assert 2000 + qty <= shelf_life_cap(MU, 5)


def test_sQ_small_lot_needs_several_lots():
    sku = Sku(lead_time=1, residual_life=20, multiple=12)
    qty = ReorderPointQ(0.95, 0.5).order_qty(0, 0, 100.0, 30.0, sku)
    # s = 100 + 1.645 x 30 = 149.3 ; lot = 50 -> 60 ; il faut 3 lots (180) pour dépasser s
    assert qty == 180


def test_RS_only_on_review_days():
    p = PeriodicReview(0.95, 3)
    assert p.order_qty(1, 1000, MU, SIGMA, SKU02) == 0
    assert p.order_qty(2, 1000, MU, SIGMA, SKU02) == 0
    assert p.order_qty(3, 1000, MU, SIGMA, SKU02) > 0


def test_RS_R1_orders_up_to_S():
    qty = PeriodicReview(0.95, 1).order_qty(0, 1000, MU, SIGMA, SKU02)
    assert qty == 3312                                # S = 4274 -> 3274 -> 69 x 48
    assert 1000 + qty <= shelf_life_cap(MU, 5)


def test_RS_R3_is_capped():
    qty = PeriodicReview(0.95, 3).order_qty(3, 1000, MU, SIGMA, SKU02)
    assert qty == 5184                                # plafond 6 208 -> 5208 -> 108 x 48
    assert 1000 + qty <= shelf_life_cap(MU, 5)


def test_order_up_to_returns_zero_when_position_high():
    assert order_up_to(1000, 1200, SKU02, cap=6000) == 0
