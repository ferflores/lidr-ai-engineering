"""Tests del cálculo de coste por tokens."""

from app.services.pricing import ModelPricing, estimate_cost, find_pricing


def test_precio_exacto_y_por_prefijo():
    assert find_pricing("gpt-4o-mini") == ModelPricing(0.15, 0.60)
    assert find_pricing("gpt-4o-mini-2024-07-18") == ModelPricing(0.15, 0.60)   # el prefijo más largo gana
    assert find_pricing("gpt-4o") == ModelPricing(2.50, 10.00)
    assert find_pricing("claude-haiku-4-5-20251001") == ModelPricing(1.00, 5.00)


def test_modelo_desconocido_sin_precio():
    assert find_pricing("modelo-inventado") is None


def test_precio_sobrescrito_por_configuracion():
    assert find_pricing("modelo-inventado", 1.0, 2.0) == ModelPricing(1.0, 2.0)
    # Hace falta sobrescribir los dos; con uno solo se usa la tabla
    assert find_pricing("gpt-4o-mini", 1.0, None) == ModelPricing(0.15, 0.60)


def test_calculo_del_coste():
    cost = estimate_cost(input_tokens=1_000_000, output_tokens=500_000, pricing=ModelPricing(0.15, 0.60))
    assert (cost.input_usd, cost.output_usd, cost.total_usd) == (0.15, 0.30, 0.45)

    cost = estimate_cost(input_tokens=1929, output_tokens=341, pricing=ModelPricing(0.15, 0.60))
    assert (cost.input_usd, cost.output_usd, cost.total_usd) == (0.000289, 0.000205, 0.000494)
