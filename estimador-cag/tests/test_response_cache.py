"""Caché de respuestas: acierto exacto, acierto semántico (Jaccard), expulsión LRU, caducidad y estadísticas."""

from app.services.llm_service import TokenUsage
from app.services.response_cache import EXACT, SEMANTIC, ResponseCache, jaccard, normalized_words

SYSTEM = "system prompt"
MENSAJES = [{"role": "user", "content": "El cliente quiere una tienda online con recogida en tienda y reparto local por el barrio."}]


def test_acierto_exacto_devuelve_el_texto_guardado():
    cache = ResponseCache()
    assert cache.lookup(SYSTEM, MENSAJES) is None

    cache.store(SYSTEM, MENSAJES, "## Estimación", TokenUsage(100, 20), "gpt-4o-mini")
    hit = cache.lookup(SYSTEM, MENSAJES)

    assert hit is not None and hit.kind == EXACT and hit.similarity == 1.0
    assert hit.entry.text == "## Estimación" and hit.entry.usage.input_tokens == 100
    assert cache.stats() == {"entries": 1, "exact_hits": 1, "semantic_hits": 0, "misses": 1}


def test_acierto_semantico_con_el_mismo_contexto_y_un_mensaje_casi_igual():
    cache = ResponseCache(semantic_threshold=0.9)
    cache.store(SYSTEM, MENSAJES, "respuesta", None, "m")

    casi_igual = [{"role": "user", "content": MENSAJES[0]["content"].replace("barrio.", "barrio").replace("El cliente", "el CLIENTE")}]
    hit = cache.lookup(SYSTEM, casi_igual)
    assert hit is not None and hit.kind == SEMANTIC and hit.similarity == 1.0  # solo cambian mayúsculas y puntuación

    distinto = [{"role": "user", "content": "El cliente quiere una app móvil de reservas con pagos y notificaciones push para clubes."}]
    assert cache.lookup(SYSTEM, distinto) is None

    otro_system = cache.lookup("otro system", casi_igual)  # otro contexto (memoria distinta): nunca acierta
    assert otro_system is None
    assert ResponseCache(semantic_threshold=1.0).lookup(SYSTEM, casi_igual) is None


def test_jaccard_y_normalizacion():
    a = normalized_words("Tienda Online, con reparto!")
    assert a == {"tienda", "online", "con", "reparto"}
    assert jaccard(a, normalized_words("tienda online con reparto")) == 1.0
    assert jaccard(a, normalized_words("tienda online con recogida")) == 0.6
    assert jaccard(frozenset(), frozenset()) == 1.0


def test_lru_expulsa_la_entrada_mas_antigua_y_el_ttl_caduca(monkeypatch):
    cache = ResponseCache(max_entries=2, ttl_seconds=10)
    m = lambda i: [{"role": "user", "content": f"mensaje número {i} con suficiente texto"}]  # noqa: E731
    cache.store(SYSTEM, m(1), "1", None, "m")
    cache.store(SYSTEM, m(2), "2", None, "m")
    cache.lookup(SYSTEM, m(1))  # 1 pasa a ser la más reciente
    cache.store(SYSTEM, m(3), "3", None, "m")

    assert cache.lookup(SYSTEM, m(2)) is None and cache.lookup(SYSTEM, m(1)) is not None and len(cache) == 2

    import app.services.response_cache as module

    now = module.time.monotonic()
    monkeypatch.setattr(module.time, "monotonic", lambda: now + 11)
    assert cache.lookup(SYSTEM, m(1)) is None and len(cache) == 0
    cache.clear()
    assert cache.stats()["misses"] == 0
