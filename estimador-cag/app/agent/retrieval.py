"""Adapt the session 10 hybrid retriever to the agent's historical-reference contract."""

from app.embedding_pipeline.retrieval import retrieve


def budget_search(session, settings, embedder, reranker):
    async def search(args):
        sector = args.filters.sector if args.filters else None
        result = await retrieve(session, query=args.query, embedder=embedder, mode="hybrid",
                                k=settings.search_candidates if sector else 5, rerank=True,
                                reranker=reranker, candidates=settings.search_candidates, rrf_k=settings.rrf_k)
        items = []
        for hit in result.results:
            m = hit.metadata
            if sector and sector.casefold() != m["client_sector"].casefold():
                continue
            items.append({"id": f"{m['budget_id']}::{m['component_id']}", "scope": hit.content,
                          "sector": m["client_sector"], "estimated_hours": m["estimated_hours"]})
        return items[:5]
    return search
