"""Precios de lista por millón de tokens (USD) para calcular el coste de cada llamada.

Los precios cambian con el tiempo: revísalos en https://openai.com/api/pricing/ y en
https://www.anthropic.com/pricing. Para el modelo activo se pueden sobrescribir con
`LLM_INPUT_PRICE_PER_MTOK` y `LLM_OUTPUT_PRICE_PER_MTOK` en `.env` (por ejemplo, para un
modelo que no esté en la tabla).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_mtok: float   # USD por millón de tokens de entrada
    output_per_mtok: float  # USD por millón de tokens de salida


MODEL_PRICING_USD: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o-mini": ModelPricing(input_per_mtok=0.15, output_per_mtok=0.60),
    "gpt-4o": ModelPricing(input_per_mtok=2.50, output_per_mtok=10.00),
    # Anthropic
    "claude-haiku-4-5": ModelPricing(input_per_mtok=1.00, output_per_mtok=5.00),
    "claude-sonnet-5": ModelPricing(input_per_mtok=2.00, output_per_mtok=10.00),
    "claude-opus-5": ModelPricing(input_per_mtok=5.00, output_per_mtok=25.00),
}


@dataclass(frozen=True)
class CostEstimate:
    input_usd: float
    output_usd: float
    total_usd: float
    pricing: ModelPricing


def find_pricing(
    model: str,
    input_override: float | None = None,
    output_override: float | None = None,
) -> ModelPricing | None:
    """Precio del modelo: el sobrescrito por configuración, o el de la tabla (exacto o por prefijo).

    `gpt-4o-mini-2024-07-18` casa con `gpt-4o-mini`; si varios prefijos casan gana el más largo,
    así `gpt-4o-mini` no se confunde con `gpt-4o`. Devuelve None si no hay precio conocido.
    """
    if input_override is not None and output_override is not None:
        return ModelPricing(input_per_mtok=input_override, output_per_mtok=output_override)
    candidatos = [name for name in MODEL_PRICING_USD if model == name or model.startswith(name + "-")]
    if not candidatos:
        return None
    return MODEL_PRICING_USD[max(candidatos, key=len)]


def estimate_cost(input_tokens: int, output_tokens: int, pricing: ModelPricing) -> CostEstimate:
    """Coste en USD de una llamada, redondeado a 6 decimales (millonésimas de dólar)."""
    input_usd = input_tokens * pricing.input_per_mtok / 1_000_000
    output_usd = output_tokens * pricing.output_per_mtok / 1_000_000
    return CostEstimate(
        input_usd=round(input_usd, 6),
        output_usd=round(output_usd, 6),
        total_usd=round(input_usd + output_usd, 6),
        pricing=pricing,
    )
