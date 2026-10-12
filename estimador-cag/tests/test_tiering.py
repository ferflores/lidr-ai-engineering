"""Tier dinámico: reglas en orden, y el tier solo cambia de modelo si la configuración lo pide."""

from app.services.tiering import EXTENDED, STANDARD, TierContext, resolve_tier
from tests.conftest import make_settings


def test_por_defecto_es_standard_con_el_modelo_del_proveedor():
    decision = resolve_tier(TierContext(context_chars=12_000, attachments_chars=0, messages_in_window=1), make_settings())

    assert decision.tier == STANDARD and decision.rule == "default"
    assert decision.model == "gpt-4o-mini" and decision.max_tokens == 1234


def test_un_adjunto_grande_resuelve_extended_antes_que_el_contexto():
    settings = make_settings()
    decision = resolve_tier(TierContext(context_chars=100_000, attachments_chars=15_000, messages_in_window=1), settings)

    assert decision.tier == EXTENDED and decision.rule == "attachments_large"
    # Sin TIER_EXTENDED_MODEL el tier usa el mismo modelo: se observa sin cambiar las medidas
    assert decision.model == settings.model and decision.max_tokens == settings.llm_max_tokens


def test_el_contexto_grande_resuelve_extended_y_los_umbrales_son_configurables():
    settings = make_settings(tier_extended_min_context_chars=50_000)
    assert resolve_tier(TierContext(49_999, 0, 11), settings).tier == STANDARD
    assert resolve_tier(TierContext(50_000, 0, 11), settings).rule == "context_large"


def test_la_configuracion_puede_dar_otro_modelo_y_otros_tokens_al_tier_extended():
    settings = make_settings(tier_extended_model="gpt-4o", tier_extended_max_tokens=8000)
    decision = resolve_tier(TierContext(context_chars=90_000, attachments_chars=0, messages_in_window=11), settings)

    assert decision.model == "gpt-4o" and decision.max_tokens == 8000
    assert resolve_tier(TierContext(100, 0, 1), settings).model == "gpt-4o-mini"
